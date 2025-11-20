from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal
import yaml
import json


@dataclass
class ModelConfig:
    name: str


@dataclass
class DataConfig:
    language: Literal["english", "german", "luxembourgish", "mixed"]
    use_bio: bool
    use_history: bool
 
    def __post_init__(self):
        base_path = Path("../../data/processed")

        # Language-specific training files
        if self.language == "english":
            self.train_file = str(base_path / "eng_train.json")
        elif self.language == "german":
            self.train_file = str(base_path / "ger_train.json")
        elif self.language == "luxembourgish":
            self.train_file = str(base_path / "lux_train.json")
        elif self.language == "mixed":
            self.train_file = str(base_path / "mixed_train.json")
        # Modify filename based on use_bio and use_history
        if self.use_bio:
            self.train_file = self.train_file.replace(".json", "_bio.json")
        if self.use_history:
            self.train_file = self.train_file.replace(".json", "_history.json")


@dataclass
class TrainingConfig:
    output_dir: str = "../../models"
    max_length: int = 10_000
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    bf16: bool = True
    assistant_only_loss: bool = False
    optim: str = "paged_adamw_32bit"
    


@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    name: str
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML file"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        
        # Handle nested configs
        if "model" in config_dict:
            config_dict["model"] = ModelConfig(**config_dict["model"])
        if "data" in config_dict:
            config_dict["data"] = DataConfig(**config_dict["data"])
        if "training" in config_dict:
            config_dict["training"] = TrainingConfig(**config_dict["training"])
        
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        """Save config to YAML file"""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)