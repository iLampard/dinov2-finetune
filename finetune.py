from transformers import AutoImageProcessor, AutoModelForImageClassification, HfArgumentParser, Trainer
from sklearn.metrics import accuracy_score
import numpy as np
from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments, DistributedArguments
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import deepspeed
from pathlib import Path
import wandb
from omegaconf import OmegaConf
from transformers.integrations import WandbCallback
from utils.log_utils import default_logger as logger
from utils.dataset_utils import create_dataset
from dataset.vtab_dataset import VTAB_NUM_CLASSES, collate_fn


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logger.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


# In the get_last_checkpoint function:
def get_last_checkpoint(checkpoint_dir, prefix):
    if not Path(checkpoint_dir).is_dir():
        return None

    if Path(checkpoint_dir, 'completed').exists():
        return None  # already finished

    checkpoints = [d for d in Path(checkpoint_dir).iterdir()
                   if d.is_dir() and d.name.startswith(prefix)]

    if not checkpoints:
        return None

    return max(checkpoints, key=lambda d: int(d.name.split('-')[-1]))


def build_dataset(processor, data_args):
    # Create training dataset
    train_ds = create_dataset(
        image_processor=processor,
        dataset_name=data_args.dataset_name,
        data_dir=data_args.train_data_dir,
        split='train'
    )

    # Create evaluation dataset if eval_data_dir is provided
    eval_ds = None
    if data_args.eval_data_dir:
        eval_ds = create_dataset(
            image_processor=processor,
            dataset_name=data_args.dataset_name,
            data_dir=data_args.eval_data_dir,
            split='eval'
        )

    return {
        'train_dataset': train_ds,
        'eval_dataset': eval_ds
    }


def build_model(model_args, training_args, data_args, lora_args, checkpoint_dir):
    if not training_args.use_lora: assert training_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if training_args.bf16 else torch.float32)

    # llm quantization config (for q-lora)
    bnb_config = None
    if training_args.use_lora and lora_args.q_lora:
        from transformers import BitsAndBytesConfig
        logger.info("Quantization for LLM enabled...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )

    # Determine device_map
    device_map = None
    if lora_args.q_lora:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size != 1:
            device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}

        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            raise ValueError("FSDP or ZeRO3 are not compatible with QLoRA.")

    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config if training_args.bits in [4, 8] else None,
        torch_dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True,
        num_labels=VTAB_NUM_CLASSES[data_args.subset_name]
    )

    logger.info("Model loaded successfully")

    if compute_dtype == torch.float32 and training_args.bits == 4:
        if torch.cuda.is_bf16_supported():
            logger.info('=' * 80)
            logger.info('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            logger.info('=' * 80)
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    if training_args.use_lora and training_args.bits < 16:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.use_lora:
        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            # os.path.join(checkpoint_dir, 'adapter_model')
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        else:
            logger.info(f'Init LoRA modules...')
            peft_config = LoraConfig(
                target_modules=lora_args.target_modules.split(','),
                inference_mode=False,
                r=lora_args.lora_rank,
                lora_alpha=lora_args.lora_alpha,
                lora_dropout=lora_args.lora_dropout,
                init_lora_weights=lora_args.init_lora_weights,
                modules_to_save=["classifier"],  # hard code here
            )
            model = get_peft_model(model, peft_config)

    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)

    # print trainable parameters for inspection
    logger.info("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"\t{name}")

    return model


# Helper functions to keep the main function clean
def get_quantization_config(training_args, lora_args):
    if lora_args.q_lora and training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=get_compute_dtype(training_args),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    return None


def get_compute_dtype(training_args):
    return torch.bfloat16 if training_args.bf16 else torch.float32


def train(model_args, data_args, training_args, lora_args, distributed_args):
    # Set up distributed training
    if distributed_args.use_distributed:
        if distributed_args.local_rank != -1:
            torch.cuda.set_device(distributed_args.local_rank)
            torch.distributed.init_process_group(backend='nccl', world_size=distributed_args.world_size)

    # Update TrainingArguments with distributed training args
    training_args.local_rank = distributed_args.local_rank
    training_args.world_size = distributed_args.world_size
    training_args.deepspeed = distributed_args.deepspeed_config
    training_args.sharded_ddp = distributed_args.sharded_ddp

    if training_args.use_wandb:
        # Programmatic login to wandb
        if 'WANDB_API_KEY' in os.environ:
            wandb.login(key=os.environ['WANDB_API_KEY'])
        else:
            wandb.login(key=training_args.wandb_token)
        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            name=training_args.wandb_run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "dataset": data_args.dataset_name,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "num_epochs": training_args.num_train_epochs,
                # Add any other hyperparameters you want to track
            }
        )

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = "DEEPSPEED"

    processor = AutoImageProcessor.from_pretrained(model_args.model_name_or_path)

    resume_from_checkpoint_dir = get_last_checkpoint(training_args.output_dir, training_args.checkpoint_prefix)

    model = build_model(model_args, training_args, data_args, lora_args, resume_from_checkpoint_dir)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    train_eval_dataset = build_dataset(processor, data_args)

    args = TrainingArguments(
        model_args.model_name_or_path,
        remove_unused_columns=False,
        evaluation_strategy=training_args.evaluation_strategy,
        save_strategy=training_args.save_strategy,
        learning_rate=training_args.learning_rate,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        # fp16=script_args.fp16,
        bf16=training_args.bf16,
        num_train_epochs=training_args.num_train_epochs,
        logging_steps=training_args.logging_steps,
        load_best_model_at_end=training_args.load_best_model_at_end,
        metric_for_best_model=training_args.metric_for_best_model,
        # push_to_hub=True,
        label_names=["labels"],
    )

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        accuracy = accuracy_score(y_pred=predictions, y_true=eval_pred.label_ids)
        return {"accuracy": accuracy}

    callbacks = []
    if training_args.use_wandb:
        callbacks.append(WandbCallback())

    trainer = Trainer(model,
                      args=args,
                      train_dataset=train_eval_dataset['train_dataset'],
                      eval_dataset=train_eval_dataset['eval_dataset'],
                      tokenizer=processor,
                      compute_metrics=compute_metrics,
                      data_collator=collate_fn,
                      callbacks=callbacks)

    # `not training_args.use_lora` is a temporary workaround for the issue that there are problems with
    # loading the checkpoint when using LoRA with DeepSpeed.
    # Check this issue https://github.com/huggingface/peft/issues/746 for more information.
    if (
            list(Path(training_args.output_dir).glob("checkpoint-*"))
            and not training_args.use_lora
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if training_args.use_wandb:
        wandb.finish()

    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args, _ = parser.parse_known_args()

    config = OmegaConf.load(args.config)

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments, DistributedArguments))
    model_args, data_args, training_args, lora_args, distributed_args = parser.parse_dict(config)

    # Update local_rank from command line argument
    distributed_args.distributed_local_rank = int(os.environ.get('LOCAL_RANK', -1))

    # Set environment variables for distributed training
    if distributed_args.use_distributed:
        os.environ['MASTER_ADDR'] = distributed_args.master_addr
        os.environ['MASTER_PORT'] = distributed_args.master_port
        os.environ['WORLD_SIZE'] = str(distributed_args.world_size)
        os.environ['LOCAL_RANK'] = str(distributed_args.distributed_local_rank)

    train(model_args, data_args, training_args, lora_args, distributed_args)
