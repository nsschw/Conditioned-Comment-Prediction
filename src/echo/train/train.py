import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import wandb
from datasets import load_dataset
import json
from datetime import datetime
import dotenv
import os


from echo.train.utility import analyse_dataset
from echo.train.config import ExperimentConfig

def train(config: ExperimentConfig):
    """Main training function"""

    # Analyse dataset token lengths
    analyse_dataset(config.model.name, config.data.train_file)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.training.output_dir) / config.name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and initialize wandb
    ENV = dotenv.dotenv_values("../" * 2 + ".env")
    wandb.login(key=ENV["WANDB_TOKEN"])
    os.environ["WANDB_PROJECT"] = "echo-v1.0"

    # Save config
    config.to_yaml(output_dir / "config.yaml")
    print(f"Starting experiment: {config.name}")
    print(f"Output dir: {output_dir}")
    print(f"Model: {config.model.name}")
    print(f"Data: {config.data.language}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    tokenizer.truncation_side="left"
    tokenizer.padding_side="left"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    # Load dataset
    train_dataset = load_dataset("json", data_files=config.data.train_file, split="train")
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=config.training.max_length,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        optim=config.training.optim,
        assistant_only_loss=config.training.assistant_only_loss,
        completion_only_loss=config.training.completion_only_loss,  
        bf16=config.training.bf16,

        gradient_checkpointing=True,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=25,

        report_to="wandb",
        run_name=config.name,
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    trainer.train()
    
    
    # Save final model
    trainer.save_model(output_dir / "final")
    
    print(f"Training complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config = ExperimentConfig.from_yaml(args.config)
    train(config)

if __name__ == "__main__":
    main()