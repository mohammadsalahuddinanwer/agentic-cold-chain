#!/usr/bin/env python3
"""Main entry point for the cold‑chain anomaly detection system."""
import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_loader import ColdChainDataLoader
from core.llm_wrapper import get_llm
from monitor_agent.anomaly_predictor import AnomalyPredictor
from diagnosis_agent.rag_retriever import RAGRetriever
from diagnosis_agent.re_act_agent import ReActAgent
from resolver_agent.resolver_agent import ResolverAgent
from resolver_agent.evolutionary_opt import EvolutionaryOptimiser
from customer_agent.customer_agent import CustomerAgent
from orchestration.langgraph_workflow import ColdChainOrchestrator
from evaluation.experiment_runner import ExperimentRunner
from monitor_agent.monitor_runtime import MonitorRuntime

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset


# --- Mock LLM for fast evaluation ---
class MockLLM:
    """
    Mock LLM that returns a fixed diagnosis instantly.
    Used for evaluation to avoid slow real LLM calls on CPU.
    """
    def generate(self, prompt, **kwargs):
        return '{"diagnosis": {"anomaly_type": "compressor_failure", "severity": 0.8, "root_cause": "Simulated anomaly for evaluation", "recommended_action": "dispatch_backup"}, "confidence": 0.9}'


def main():
    parser = argparse.ArgumentParser(description="Cold‑Chain Anomaly Detection System")
    parser.add_argument(
        "--mode",
        choices=["generate_data", "train_monitor", "run", "evaluate", "test_llm"],
        default="run",
    )
    parser.add_argument("--shipments", type=int, default=100, help="Number of shipments to generate")
    parser.add_argument("--dataset", choices=["euro_crop", "smart", "mendeley"],
                        default="euro_crop", help="Dataset for training")
    parser.add_argument("--output", type=str, default="outputs/results.csv", help="Output path")
    parser.add_argument("--no_8bit", action="store_true", help="Disable 8-bit quantization")
    args = parser.parse_args()

    Config.create_dirs()

    if args.mode == "test_llm":
        print("Testing Llama 3.1 integration...")
        llm = get_llm(
            load_in_8bit=not args.no_8bit,
            temperature=0.2,
            max_new_tokens=256,
        )

        response = llm.generate(
            "What are the common causes of temperature excursions in cold-chain logistics?",
            system_prompt="You are a helpful logistics expert. Answer concisely.",
        )
        print("\n=== Llama 3.1 Response ===")
        print(response)
        return

    if args.mode == "generate_data":
        print("Generating synthetic data...")
        from data_generation.generator import ColdChainGenerator
        generator = ColdChainGenerator(seed=42)
        shipments = generator.generate_dataset(args.shipments)
        df = generator.to_dataframe(shipments)
        df.to_csv(Config.SYNTHETIC_DATA_DIR / "shipments.csv", index=False)
        print(f"Generated {len(shipments)} shipments, saved to {Config.SYNTHETIC_DATA_DIR / 'shipments.csv'}")
        return

    # Initialize real Llama 3.1 (for demo and test modes)
    print("Initialising Llama 3.1...")
    llm = get_llm(**Config.get_llama_kwargs())

    # Initialize Monitor Agent
    # monitor = AnomalyPredictor(
    #    n_steps=Config.MONITOR_WINDOW_SIZE,
    #    n_features=len(Config.EURO_CROP_FEATURES),
    #    device=Config.MONITOR_DEVICE,
    #)
    monitor = MonitorRuntime(device=Config.MONITOR_DEVICE)

    if args.mode == "train_monitor":
        print(f"Training Monitor Agent on {args.dataset}...")
        loader = ColdChainDataLoader(Config)

        if args.dataset == "euro_crop":
            train_loader, val_loader, _ = loader.load_euro_crop_dataloaders(
                batch_size=64,
                window_size=Config.MONITOR_WINDOW_SIZE,
                horizon=30
            )
        else:
            print("Currently only euro_crop is supported for Monitor training.")
            return

        monitor.train(train_loader, val_loader, epochs=50)
        monitor.save_checkpoint(str(Config.CHECKPOINT_DIR / "monitor_best.pt"))
        print("Training complete.")

    elif args.mode == "run":
        # Load trained monitor
        try:
            monitor.load_checkpoint(str(Config.CHECKPOINT_DIR / "monitor_best.pt"))
            print("Loaded Monitor Agent")
            for name, param in monitor.model.named_parameters():
                if torch.isnan(param).any():
                    print(f"⚠️ NaN found in {name}")
                if torch.isinf(param).any():
                    print(f"⚠️ Inf found in {name}")
        except FileNotFoundError:
            print("Monitor checkpoint not found. Please train first with --mode train_monitor")
            return

        # RAG Retriever
        rag = RAGRetriever()
        #knowledge_files = list(Config.KNOWLEDGE_BASE_DIR.glob("*.json"))
        knowledge_files = [f for f in sorted(Config.KNOWLEDGE_BASE_DIR.glob("*.json"))
                           if f.name not in Config.EXCLUDED_KNOWLEDGE_FILES]
        if knowledge_files:
            rag.build_index(knowledge_files)
            print(f"Loaded {len(knowledge_files)} knowledge base files")
        else:
            print("No knowledge base files found. RAG will return empty results.")

        # Agents with REAL LLM
        react_agent = ReActAgent(llm, rag)
        optimiser = EvolutionaryOptimiser()
        resolver = ResolverAgent(llm, optimiser)
        customer = CustomerAgent(llm)
        orchestrator = ColdChainOrchestrator(monitor, react_agent, resolver, customer)

        print("\n=== Running System on Sample Data ===")

        # --- Load and scale data exactly like training ---
        df = pd.read_csv(Config.EURO_CROP_PATH)
        feature_cols = Config.EURO_CROP_FEATURES

        # 1. Clean the data
        data_df = df[feature_cols].copy()
        for col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        data_df = data_df.replace([np.inf, -np.inf], np.nan)
        for col in data_df.columns:
            median_val = data_df[col].median()
            if np.isnan(median_val):
                data_df = data_df.drop(columns=[col])
            else:
                data_df[col] = data_df[col].fillna(median_val)
        data_df = data_df.dropna()
        for col in data_df.columns:
            q01 = np.nanpercentile(data_df[col].values, 1)
            q99 = np.nanpercentile(data_df[col].values, 99)
            if q01 != q99:
                data_df[col] = data_df[col].clip(lower=q01, upper=q99)

        # 2. Scale
        scaler = RobustScaler()
        data_scaled = scaler.fit_transform(data_df.values)

        # 3. Take first 60 rows
        sample_window = data_scaled[:60].copy()

        # 4. Inject anomaly in scaled space
        print("⚠️ Injecting anomaly in scaled space (value = 15.0)...")
        for i in range(25, 35):
            sample_window[i, 0] = 15.0 + (i - 25) * 0.2
        print(f"DEBUG: Scaled window max: {sample_window[:, 0].max():.2f}")

        # 5. Convert to list of dicts with Python floats
        sensor_dicts = [
            {col: float(sample_window[i][j]) for j, col in enumerate(feature_cols)}
            for i in range(sample_window.shape[0])
        ]

        # 6. Run orchestrator
        result = orchestrator.run(
            sensor_window=sensor_dicts,
            logs=["Driver: En route, temperature stable"],
            context={
                "product_type": "vaccines",
                "current_location": "Berlin",
                "product_value": 5000,
            }
        )

        print("\n=== RESULT ===")
        print(f"Monitor prediction: {result.get('monitor_prediction')}")
        print(f"Diagnosis: {result.get('diagnosis')}")
        print(f"Resolution plan: {result.get('resolution_plan')}")
        print(f"Customer messages: {result.get('customer_messages')}")

    elif args.mode == "evaluate":
        print("Running systematic evaluation with Mock LLM (fast)...")

        # --- USE MOCK LLM FOR EVALUATION ---
        mock_llm = MockLLM()

        # RAG Retriever
        rag = RAGRetriever()
        #knowledge_files = list(Config.KNOWLEDGE_BASE_DIR.glob("*.json"))
        knowledge_files = [f for f in sorted(Config.KNOWLEDGE_BASE_DIR.glob("*.json"))
                           if f.name not in Config.EXCLUDED_KNOWLEDGE_FILES]
        if knowledge_files:
            rag.build_index(knowledge_files)
            print(f"Loaded {len(knowledge_files)} knowledge base files")
        else:
            print("No knowledge base files found. RAG will return empty results.")

        # Agents with MOCK LLM
        react_agent = ReActAgent(mock_llm, rag)
        optimiser = EvolutionaryOptimiser()
        resolver = ResolverAgent(mock_llm, optimiser)
        customer = CustomerAgent(mock_llm)
        orchestrator = ColdChainOrchestrator(monitor, react_agent, resolver, customer)

        from data_generation.generator import ColdChainGenerator
        generator = ColdChainGenerator(seed=42)
        runner = ExperimentRunner(orchestrator, generator)

        try:
            monitor.load_checkpoint(str(Config.CHECKPOINT_DIR / "monitor_best.pt"))
            print("Loaded Monitor Agent")
        except FileNotFoundError:
            print("Monitor checkpoint not found. Please train first.")
            return

        results = runner.run_full_grid()
        runner.save_results(results, args.output)
        print(f"Evaluation complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()