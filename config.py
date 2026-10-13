"""Configuration management for the cold‑chain anomaly system."""
import os
from dotenv import load_dotenv
from pathlib import Path
import torch

load_dotenv()


class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
    KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
    OUTPUT_DIR = BASE_DIR / "outputs"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"

    # Dataset paths (place your downloaded CSV files here)
    EURO_CROP_PATH = RAW_DATA_DIR / "euro_crop_agricultural_logistics.csv"
    SMART_LOGISTICS_PATH = RAW_DATA_DIR / "smart_logistics_supply_chain.csv"
    MENDELEY_PATH = RAW_DATA_DIR / "mendeley_cold_storage.csv"

    # Column mappings for each dataset
    EURO_CROP_FEATURES = [
        "Storage_Temperature",
        #"Storage_Humidity",
        "IoT_Sensor_Reading_Temperature",
        #"IoT_Sensor_Reading_Humidity",
        "Vibration_Level",
        #"Route_Distance",
        "Fuel_Consumption",
        "Quality_Maintenance_Ratio"
    ]
    EURO_CROP_TARGET = "Efficiency_Ratio"

    SMART_LOGISTICS_FEATURES = [
        "Temperature",
        "Humidity",
        "Traffic_Status",
        "Waiting_Time",
    ]
    SMART_LOGISTICS_TARGET = "Logistics_Delay"

    MENDELEY_FEATURES = [
        "Temp",
        "Humid",
        "Light",
        "CO2",
    ]
    MENDELEY_TARGET = "Class"

    # Monitor Agent
    MONITOR_MODEL = "informer"
    MONITOR_WINDOW_SIZE = 60  # 60 time steps (5 hours)
    MONITOR_HORIZONS = [30, 60, 90]  # minutes
    MONITOR_THRESHOLD = 1.0  # °C for vaccines
    MONITOR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MONITOR_PROB_THRESHOLD = 0.80   # classifier probability gate (thesis Sect. 3.6)
    MONITOR_FEATURES = ["temperature_c", "humidity_pct", "vibration_mg", "door_open", "co2_ppm"]

    # Llama 3.1 Configuration
    LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    LLAMA_LOAD_IN_8BIT = True  # Reduces VRAM to ~6GB
    LLAMA_DEVICE_MAP = "auto"
    LLAMA_TORCH_DTYPE = "bfloat16"  # or "float16"
    #LLAMA_MAX_NEW_TOKENS = 512
    LLAMA_MAX_NEW_TOKENS = 256
    LLAMA_TEMPERATURE = 0.2
    LLAMA_TOP_P = 0.9
    LLAMA_USE_PIPELINE = True

    # RAG configuration
    RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_VECTOR_STORE = "faiss"
    RAG_TOP_K = 5
    RAG_SIMILARITY_THRESHOLD = 0.75
    EXCLUDED_KNOWLEDGE_FILES = {
        "diagnosis_test_set.json", "diagnosis_test_set_100.json",
        "anomaly_cases _bkp.json", "anomaly_cases_59.json",
        "anomaly_cases_100.json", "operational_logs.json",
        "resolution_strategies.json",
    }

    # Customer agent configuration
    CUSTOMER_TEMPERATURE = 0.7
    CUSTOMER_MAX_TOKENS = 200

    # Evolutionary optimisation configuration.
    #
    # NOTE: these are the defaults for resolver_agent/evolutionary_opt.py, the
    # runtime optimiser used by the live pipeline (app.py). They are NOT the
    # settings behind the A2 results in Sect. 5.6 of the paper. That experiment
    # is run by run_a2_experiment.py, which takes its own values on the command
    # line and was executed with --population 60 --generations 30.
    GA_POPULATION_SIZE = 100
    GA_GENERATIONS = 50
    GA_CROSSOVER_PROB = 0.8
    GA_MUTATION_PROB = 0.1
    GA_NUM_OBJECTIVES = 4

    # Evaluation configuration
    SEED = 42
    EXPERIMENT_RUNS = 30
    #EXPERIMENT_RUNS = 3
    TEST_SPLIT = 0.2
    VAL_SPLIT = 0.1

    @classmethod
    def create_dirs(cls):
        """Create all necessary directories."""
        for dir_path in [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.SYNTHETIC_DATA_DIR,
            cls.KNOWLEDGE_BASE_DIR,
            cls.OUTPUT_DIR,
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_api_key(cls, service):
        """Get API key from environment."""
        return os.getenv(f"{service}_API_KEY")

    @classmethod
    def get_llama_kwargs(cls):
        """Get Llama 3.1 configuration as kwargs."""
        return {
            "model_id": cls.LLAMA_MODEL_ID,
            "load_in_8bit": cls.LLAMA_LOAD_IN_8BIT,
            "device_map": cls.LLAMA_DEVICE_MAP,
            "torch_dtype": getattr(torch, cls.LLAMA_TORCH_DTYPE),
            "max_new_tokens": cls.LLAMA_MAX_NEW_TOKENS,
            "temperature": cls.LLAMA_TEMPERATURE,
            "top_p": cls.LLAMA_TOP_P,
            "use_pipeline": cls.LLAMA_USE_PIPELINE,
        }