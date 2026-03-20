import argparse
import os

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# ─── PROFILE DEFINITIONS ──────────────────────────────────────────────────
# Add new profiles here to handle different hardware targets.
PROFILES = {
    "laptop": {
        "max_seq_length": 1024,  # Massive VRAM saver
        "per_device_train_batch_size": 1,  # Minimal VRAM footprint
        "gradient_accumulation_steps": 8,  # Compels batch size 1 to act like 8
        "max_steps": 20,  # Quick smoke test / Small GPU safe
        "optim": "adamw_8bit",  # Essential for laptop GPUs
    },
    "standard": {
        "max_seq_length": 2048,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_steps": 60,
        "optim": "adamw_8bit",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace Base Model ID")
    parser.add_argument("--data", required=True, help="Path to local staged JSONL dataset")
    parser.add_argument("--out", required=True, help="Samba path to save fine-tuned adapters")
    parser.add_argument(
        "--profile",
        default="standard",
        choices=["standard", "laptop"],
        help="Hardware profile to use",
    )
    args = parser.parse_args()

    # Load parameters from the selected profile
    p = PROFILES[args.profile]

    print(f"🚀 Initializing Unsloth Fine-Tuning [Profile: {args.profile.upper()}]")
    print(f"📦 Base Model: {args.model}")
    print(f"📂 Dataset: {args.data}")

    # 1. Load Model and Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=p["max_seq_length"],
        load_in_4bit=True,
        token=os.getenv("HF_TOKEN"),
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Process Dataset
    dataset = load_dataset("json", data_files=args.data, split="train")

    def format_prompts(examples):
        texts = []
        for messages in examples["messages"]:
            texts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)

    # 4. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=p["max_seq_length"],
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=p["per_device_train_batch_size"],
            gradient_accumulation_steps=p["gradient_accumulation_steps"],
            warmup_steps=2,
            max_steps=p["max_steps"],
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=p["optim"],
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="/tmp/outputs",
        ),
    )

    # 5. Execute Training
    print(f"🔥 Starting GPU Training kernels (SeqLen: {p['max_seq_length']})...")
    trainer_stats = trainer.train()

    # 6. Save Artifacts to Samba Hub
    print(f"💾 Exporting fine-tuned adapters to {args.out}...")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print("✨ Training Complete. Weights registered on Samba share.")


if __name__ == "__main__":
    main()
