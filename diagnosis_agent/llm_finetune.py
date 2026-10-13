"""Fine-tuning pipeline for Llama 3.1 on anomaly classification."""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset
from typing import List, Dict, Any
import json
import os


class LlamaFineTuner:
    """
    Fine-tunes Llama 3.1 on cold-chain anomaly classification.
    Uses LoRA for efficient parameter-efficient fine-tuning.
    """

    def __init__(self, config):
        self.config = config
        self.model_id = config.LLAMA_MODEL_ID
        self.lora_rank = config.DIAGNOSIS_LORA_RANK
        self.lora_alpha = config.DIAGNOSIS_LORA_ALPHA
        self.load_in_8bit = config.LLAMA_LOAD_IN_8BIT

    def prepare_data(self, data_path: str) -> Dataset:
        """Prepare training data from JSON file."""
        with open(data_path, 'r') as f:
            data = json.load(f)

        # Format data for instruction tuning
        formatted_data = []
        for item in data:
            text = item.get('text', '')
            metadata = item.get('metadata', {})
            anomaly_type = metadata.get('type', 'unknown')

            instruction = f"Classify the following anomaly description: {text}"
            response = f"The anomaly type is: {anomaly_type}"

            formatted_data.append({
                "instruction": instruction,
                "response": response
            })

        return Dataset.from_list(formatted_data)

    def setup_model(self):
        """Load and prepare model for fine-tuning."""
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=self.load_in_8bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, tokenizer

    def train(self, train_dataset: Dataset, val_dataset: Dataset = None,
              output_dir: str = "outputs/lora_checkpoint"):
        """Run fine-tuning."""
        model, tokenizer = self.setup_model()

        # Format function for training
        def format_example(examples):
            prompts = []
            for instruction, response in zip(examples["instruction"], examples["response"]):
                prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
                prompts.append(prompt)
            return tokenizer(prompts, truncation=True, padding=True, max_length=512)

        tokenized_train = train_dataset.map(lambda x: format_example(x), batched=True)
        if val_dataset:
            tokenized_val = val_dataset.map(lambda x: format_example(x), batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            logging_steps=10,
            save_steps=50,
            eval_steps=50 if val_dataset else None,
            evaluation_strategy="steps" if val_dataset else "no",
            report_to="none",
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val if val_dataset else None,
            data_collator=data_collator,
        )

        trainer.train()

        # Save final model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        return model

    def load_finetuned(self, checkpoint_path: str):
        """Load a fine-tuned model."""
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        return model, tokenizer