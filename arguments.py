from typing import Optional, List, Literal, Union
from dataclasses import dataclass, field
import transformers
from pathlib import Path


# ref: https://github.com/zjysteven/lmms-finetune/blob/main/arguments.py

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default='/content/dinov2-finetune/dinov2-base')


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default='vtab')
    subset_name: Optional[str] = field(default='caltech101')

    train_data_dir: Union[str, Path] = field(
        default='/content/dinov2-finetune/data/vtab-1k/caltech101/train800.txt',
        metadata={"help": "Path to the training data."}
    )
    eval_data_dir: Union[str, Path] = field(
        default='/content/dinov2-finetune/data/vtab-1k/caltech101/val200.txt',
        metadata={"help": "Path to the evaluation data."}
    )
    test_data_dir: Union[str, Path] = field(
        default='/content/dinov2-finetune/data/vtab-1k/caltech101/test.txt',
        metadata={"help": "Path to the test data."}
    )

    def __post_init__(self):
        if (self.train_data_dir is None and self.eval_data_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )

        # Convert string paths to Path objects
        self.train_data_dir = Path(self.train_data_dir).expanduser()
        self.eval_data_dir = Path(self.eval_data_dir).expanduser()
        self.test_data_dir = Path(self.test_data_dir).expanduser()


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    output_dir: Optional[str] = field(default='../output')

    checkpoint_prefix: str = field(
        default="checkpoint",
        metadata={"help": "Prefix for checkpoint directories"}
    )

    num_train_epochs: Optional[int] = field(default=1)
    logging_steps: Optional[int] = field(default=5)

    per_device_train_batch_size: int = field(default=16, metadata={"help": "batch size for training data."}, )
    per_device_eval_batch_size: int = field(default=4, metadata={"help": "batch size for eval data."}, )
    per_device_test_batch_size: int = field(default=4, metadata={"help": "batch size for test data."}, )

    # metric_for_best_model: str = field(default='accuracy')
    # load_best_model_at_end: bool = field(default='True')
    eval_strategy: Literal['no', 'steps', 'epoch'] = field(default='no')
    save_strategy: Literal['no', 'steps', 'epoch'] = field(default='epoch')
    do_eval: bool = field(default=False)

    label_names: List[str] = field(
        default_factory=lambda: [
            "labels"
        ])

    # Quantization setting
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True,
                               metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",
                            metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})

    use_lora: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False
        if self.bits not in [4, 8, 16, 32]:
            raise ValueError("bits must be one of 4, 8, 16, or 32")
        if self.quant_type not in ["fp4", "nf4"]:
            raise ValueError("quant_type must be either 'fp4' or 'nf4'")


@dataclass
class LoraArguments:
    q_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    target_modules: Optional[str] = field(default="query,value,key")
    init_lora_weights: Literal[True, "pissa"] = field(default=True,
                                                      metadata={"help": ("True -> LoRA; `pissa` -> PiSSA"), })