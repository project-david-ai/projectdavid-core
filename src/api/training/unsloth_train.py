import json
import os

# ─── SOVEREIGNTY GUARD ────────────────────────────────────────────────────────
# HF_HUB_OFFLINE = "1" enforces cache-only mode — no downloads permitted.
# Set to "0" only if you explicitly want to allow HuggingFace hub downloads.
# For production sovereign deployments this should be "1".
os.environ["HF_HUB_OFFLINE"] = "1"
# ──────────────────────────────────────────────────────────────────────────────

# isort: split

# 1. CRITICAL: Unsloth MUST be imported before everything else
import argparse

import unsloth  # noqa: F401 — must precede trl/transformers/peft
from datasets import load_dataset
from transformers import TrainerCallback
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


# ─── PROGRESS EMITTER ─────────────────────────────────────────────────────────
class ProgressEmitter(TrainerCallback):
    """
    Emits structured PROGRESS: lines to stdout on every logging step.
    The training worker parses these lines and writes them to job.metrics
    so users get live feedback during training instead of a black hole.

    Output format (one line per logging step):
        PROGRESS:{"step": 5, "total_steps": 20, "epoch": 0.25, "loss": 1.423, "learning_rate": 0.0002}

    Note on leading newline:
        HuggingFace Transformers uses tqdm for its progress bar, which writes
        progress updates to stdout without a trailing newline (it uses carriage
        returns to update in-place). When our PROGRESS print fires on the same
        logging step, its output ends up concatenated to the end of the tqdm
        line, e.g.:

            5%|▌ | 1/20 [00:03<01:08,  3.61s/it]PROGRESS:{"step": 1, ...}

        The downstream parser in worker.py uses line.startswith("PROGRESS:")
        which fails on that concatenated form — only the final, clean PROGRESS
        emit (after tqdm is done) gets captured.

        Prepending "\n" guarantees our PROGRESS line starts on its own line
        regardless of what tqdm has done to stdout. The worker's stdout reader
        then sees it as an independent line and matches cleanly.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        progress = {
            "step": state.global_step,
            "total_steps": state.max_steps,
            "epoch": round(state.epoch or 0, 3),
            "loss": round(logs.get("loss", 0), 4),
            "learning_rate": logs.get("learning_rate"),
        }
        print(f"\nPROGRESS:{json.dumps(progress)}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────


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

    # ─── TOKENIZER COMPATIBILITY ──────────────────────────────────────────────
    # Qwen2 uses native special tokens that differ from other model families:
    #   <|im_end|>    — end of turn, used as eos
    #   <|endoftext|> — safe pad token, always in Qwen2 vocab
    #
    # For all other model families (Llama, Mistral, Phi, Gemma, etc.),
    # fall back to the universal safe pattern: pad_token = eos_token.
    #
    # Do NOT hardcode <|endoftext|> globally — it is not in Llama-3.x vocab
    # and will cause SFTTrainer to reject the tokenizer at init.
    if "qwen" in args.model.lower():
        tokenizer.eos_token = "<|im_end|>"  # nosec B105
        tokenizer.pad_token = "<|endoftext|>"  # nosec B105
    else:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
        callbacks=[ProgressEmitter()],
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
