# 1. CRITICAL: Unsloth MUST be imported before everything else for the 4060 kernels
import argparse
import json
import os

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# ─── PROFILE DEFINITIONS ──────────────────────────────────────────────────
PROFILES = {
    "laptop": {
        "max_seq_length": 1024,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": 20,
        "optim": "adamw_8bit",
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", default="standard", choices=["standard", "laptop"])
    args = parser.parse_args()

    p = PROFILES[args.profile]

    print(f"🚀 Initializing Unsloth Fine-Tuning [Profile: {args.profile.upper()}]")

    # 2. Load Model & Tokenizer
    # We set the sequence length here; the trainer will inherit it automatically
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=p["max_seq_length"],
        load_in_4bit=True,
        token=os.getenv("HF_TOKEN"),
    )

    # 3. Add LoRA Adapters
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

    # 4. Process Dataset (THE VERSION-PROOF WAY)
    dataset = load_dataset("json", data_files=args.data, split="train")

    def format_prompts(examples):
        texts = []
        for messages in examples["messages"]:
            texts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )
        return {"text": texts}

    # By removing all columns except 'text', the SFTTrainer
    # needs zero configuration to find the right data.
    dataset = dataset.map(
        format_prompts, batched=True, num_proc=2, remove_columns=dataset.column_names
    )

    # 5. Initialize Trainer (Minimalist Signature)
    # This avoids the 'max_seq_length' and 'dataset_text_field' TypeErrors
    # by relying on the model and dataset defaults.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,  # Required for TRL 0.24.0+
        args=SFTConfig(
            # Standard Training parameters
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
            report_to="none",
            packing=False,
        ),
    )

    # 6. Execute Training
    print(f"🔥 Starting GPU Training kernels (Laptop Mode: {p['max_seq_length']} ctx)...")
    trainer.train()

    # 7. Save Artifacts
    print(f"💾 Exporting fine-tuned adapters to {args.out}...")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print("✨ Training Complete. Weights registered on Samba share.")


if __name__ == "__main__":
    main()
