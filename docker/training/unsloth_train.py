# docker/training/unsloth_train.py
#
# Unsloth training script for the projectdavid training container.
# Called by entrypoint.sh when --framework unsloth is specified.
#
# Usage: python unsloth_train.py /mnt/training_data/configs/{job_id}/config.yml
#
# Config keys consumed:
#   base_model                    — HF model ID or local path
#   dataset_path                  — path to prepared JSONL under /mnt/training_data
#   output_dir                    — checkpoint output path under /mnt/training_data
#   max_seq_length                — default 2048
#   load_in_4bit                  — default True
#   lora_r                        — default 16
#   lora_alpha                    — default 32
#   lora_dropout                  — default 0.05
#   per_device_train_batch_size   — default 4
#   gradient_accumulation_steps   — default 4
#   num_epochs                    — default 3
#   learning_rate                 — default 2e-4
#   save_steps                    — default 100
#   logging_steps                 — default 10
#   bf16                          — default False (uses fp16)
#   dataset_text_field            — default "text"

import sys
from pathlib import Path

import yaml


def main():
    if len(sys.argv) < 2:
        print("[unsloth] ERROR: config path required", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    print(f"[unsloth] Loading config from: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"[unsloth] Base model  : {cfg['base_model']}")
    print(f"[unsloth] Dataset     : {cfg['dataset_path']}")
    print(f"[unsloth] Output dir  : {cfg['output_dir']}")

    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg.get("max_seq_length", 2048),
        load_in_4bit=cfg.get("load_in_4bit", True),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        use_gradient_checkpointing=True,
    )

    print(f"[unsloth] Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files=cfg["dataset_path"],
        split="train",
    )
    print(f"[unsloth] Dataset size: {len(dataset)} samples")

    output_dir = cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=cfg.get("dataset_text_field", "text"),
        max_seq_length=cfg.get("max_seq_length", 2048),
        args=TrainingArguments(
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
            num_train_epochs=cfg.get("num_epochs", 3),
            learning_rate=cfg.get("learning_rate", 2e-4),
            output_dir=output_dir,
            save_steps=cfg.get("save_steps", 100),
            logging_steps=cfg.get("logging_steps", 10),
            fp16=not cfg.get("bf16", False),
            bf16=cfg.get("bf16", False),
            report_to="none",
        ),
    )

    print("[unsloth] Starting training...")
    stats = trainer.train()
    print(f"[unsloth] Training stats: {stats}")

    print(f"[unsloth] Saving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[unsloth] Done.")


if __name__ == "__main__":
    main()
