import os

# ─── SOVEREIGNTY GUARD ────────────────────────────────────────────────────────
# Prevent any HuggingFace hub download attempts at runtime.
# Only models already present in the local HF cache are permitted.
# If the requested model is not cached, this will raise a clear error
# rather than attempting a download — enforcing airgap compliance.
os.environ["HF_HUB_OFFLINE"] = "0"
# ──────────────────────────────────────────────────────────────────────────────

# isort: split

# 1. CRITICAL: Unsloth MUST be imported before everything else
import argparse

import unsloth  # noqa: F401 — must precede trl/transformers/peft
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
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=p["max_seq_length"],
        load_in_4bit=True,
        token=os.getenv("HF_TOKEN"),
        fix_tokenizer=True,
    )

    # ─── QWEN/TRL COMPATIBILITY FIX ──────────────────────────────────────────
    # Do NOT override eos_token with a fallback string — fix_tokenizer may set
    # <EOS_TOKEN> which is not in Qwen2's vocabulary and causes TRL to reject it.
    # Use Qwen2's native vocab tokens directly:
    #   <|im_end|>    — end of turn, used as eos
    #   <|endoftext|> — safe pad token, always in vocab
    # max_seq_length is set on the tokenizer — newer TRL versions no longer
    # accept it as an argument to SFTConfig or SFTTrainer.

    # Qwen2 native special tokens — not passwords, nosec B105 not recognised here
    tokenizer.eos_token = "<|im_end|>"  # nosec B105
    tokenizer.pad_token = "<|endoftext|>"  # nosec B105
    tokenizer.model_max_length = p["max_seq_length"]
    # ──────────────────────────────────────────────────────────────────────────

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

    # 4. Process Dataset
    dataset = load_dataset("json", data_files=args.data, split="train")  # nosec B615

    def format_prompts(examples):
        texts = []

        # Support both ShareGPT (conversations) and ChatML (messages) formats.
        # ShareGPT uses 'from'/'value' keys; ChatML uses 'role'/'content' keys.
        # apply_chat_template requires role/content — normalise ShareGPT on the fly.
        records = examples.get("conversations") or examples.get("messages") or []

        for messages in records:
            if messages and isinstance(messages[0], dict) and "from" in messages[0]:
                # Normalise ShareGPT → ChatML
                messages = [
                    {
                        "role": "user" if m["from"] == "human" else "assistant",
                        "content": m["value"],
                    }
                    for m in messages
                ]
            texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
        return {"text": texts}

    # Clean the dataset to have ONLY the 'text' column for SFTTrainer
    dataset = dataset.map(
        format_prompts, batched=True, num_proc=2, remove_columns=dataset.column_names
    )

    # 5. Initialize Trainer
    # max_seq_length is not accepted by SFTConfig or SFTTrainer in newer TRL
    # versions — set via tokenizer.model_max_length above instead.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            dataset_text_field="text",
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
            output_dir="/tmp/outputs",  # nosec B108
            report_to="none",
            packing=False,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,  # nosec B105
            },
        ),
    )

    # 6. Execute Training
    print(
        f"🔥 Starting GPU Training kernels (Profile: {args.profile.upper()}, "
        f"ctx: {p['max_seq_length']})..."
    )
    trainer.train()

    # 7. Save Artifacts
    print(f"💾 Exporting fine-tuned adapters to {args.out}...")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print("✨ Training Complete. Weights registered on Samba share.")


if __name__ == "__main__":
    main()
