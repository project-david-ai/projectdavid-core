import json
import os

# ─── SOVEREIGNTY GUARD ────────────────────────────────────────────────────────
# HF_HUB_OFFLINE = "1" enforces cache-only mode — no downloads permitted.
os.environ["HF_HUB_OFFLINE"] = "1"
# ──────────────────────────────────────────────────────────────────────────────

# isort: split

# 1. CRITICAL: Unsloth MUST be imported before everything else
import argparse
from pathlib import Path

import unsloth  # noqa: F401 — must precede trl/transformers/peft
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# ─── TRAINER FALLBACKS ────────────────────────────────────────────────────────
# Safety net only. The service layer is expected to write a complete resolved
# config, in which case none of these are used. Retained so manual invocations
# with sparse config files still run.
#
# target_modules is fixed here (not exposed via the API in Phase 1) — Phase 2
# will add base-model-aware validation before it becomes user-tunable.
TRAINER_FALLBACKS = {
    "max_seq_length": 2048,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_steps": 60,
    "optim": "adamw_8bit",
    "learning_rate": 2e-4,
    "warmup_steps": 2,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "logging_steps": 50,
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "bias": "none",
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


# ─── PROGRESS EMITTER ─────────────────────────────────────────────────────────
class ProgressEmitter(TrainerCallback):
    """
    Emits structured PROGRESS: lines to stdout on every logging step.
    The training worker parses these lines and writes them to job.metrics.

    Leading newline: HuggingFace tqdm writes progress without trailing newline
    (uses carriage returns). Prepending "\\n" guarantees our PROGRESS line
    starts on its own line so the worker's line.startswith("PROGRESS:") parser
    matches cleanly.
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


def load_config(config_path: str) -> dict:
    """Load training config from the JSON file written by the worker."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # Merge fallbacks under the file contents so missing keys don't crash.
    # The service layer writes a complete dict; fallbacks are last-resort.
    merged = dict(TRAINER_FALLBACKS)
    merged.update(raw)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to JSON file containing the fully-resolved training config "
        "(written by the training worker at job start).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config_path)

    print(f"🚀 Initializing Unsloth Fine-Tuning [config: {args.config_path}]")
    print(
        f"   profile={cfg.get('_profile')}  "
        f"max_seq_length={cfg['max_seq_length']}  "
        f"max_steps={cfg['max_steps']}  lr={cfg['learning_rate']}"
    )
    print(
        f"   lora_r={cfg['lora_r']}  lora_alpha={cfg['lora_alpha']}  "
        f"batch={cfg['per_device_train_batch_size']}  "
        f"accum={cfg['gradient_accumulation_steps']}"
    )

    # 2. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
        token=os.getenv("HF_TOKEN"),
        fix_tokenizer=True,
    )

    # ─── TOKENIZER COMPATIBILITY ──────────────────────────────────────────────
    # Qwen2 uses native special tokens differing from other families:
    #   <|im_end|>    — end of turn (eos)
    #   <|endoftext|> — safe pad token, always in Qwen2 vocab
    #
    # For other families (Llama, Mistral, Phi, Gemma, etc.), fall back to
    # the universal safe pattern: pad_token = eos_token. Do NOT hardcode
    # <|endoftext|> globally — not in Llama-3.x vocab, SFTTrainer rejects.
    if "qwen" in args.model.lower():
        tokenizer.eos_token = "<|im_end|>"  # nosec B105
        tokenizer.pad_token = "<|endoftext|>"  # nosec B105
    else:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.model_max_length = cfg["max_seq_length"]
    # ──────────────────────────────────────────────────────────────────────────

    # 3. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    # 4. Process Dataset
    dataset = load_dataset("json", data_files=args.data, split="train")  # nosec B615

    def format_prompts(examples):
        texts = []

        # Support both ShareGPT (conversations) and ChatML (messages) formats.
        # ShareGPT uses 'from'/'value'; ChatML uses 'role'/'content'.
        # apply_chat_template requires role/content — normalise ShareGPT on the fly.
        records = examples.get("conversations") or examples.get("messages") or []

        for messages in records:
            if messages and isinstance(messages[0], dict) and "from" in messages[0]:
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

    dataset = dataset.map(
        format_prompts, batched=True, num_proc=2, remove_columns=dataset.column_names
    )

    # 5. Initialize Trainer
    # max_seq_length is not accepted by SFTConfig or SFTTrainer in newer TRL
    # versions — set via tokenizer.model_max_length above instead.
    sft_kwargs = dict(
        dataset_text_field="text",
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=cfg["warmup_steps"],
        max_steps=cfg["max_steps"],
        learning_rate=cfg["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=cfg["logging_steps"],
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        seed=cfg["seed"],
        output_dir="/tmp/outputs",  # nosec B108
        report_to="none",
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,  # nosec B105
        },
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[ProgressEmitter()],
        args=SFTConfig(**sft_kwargs),
    )

    # 6. Execute Training
    print(
        f"🔥 Starting GPU Training kernels "
        f"(ctx: {cfg['max_seq_length']}, max_steps: {cfg['max_steps']})..."
    )
    train_output = trainer.train()

    # ─── FINAL LOSS EMISSION ──────────────────────────────────────────────────
    # trainer.train() returns a TrainOutput with train_output.training_loss —
    # the MEAN loss across all training steps. That's the canonical HF summary
    # value. Separately, trainer.state.log_history carries the per-step losses
    # from logging_steps emissions; the last entry with a "loss" key is the
    # LAST STEP'S loss.
    #
    # Per-step chart trajectories are only honest if we emit the last step's
    # loss, not the run mean (means always look anomalous next to step values
    # — they sit at the average of the descending curve rather than at its
    # tail, which reads as a regression on a chart).
    #
    # We emit both:
    #   loss       — last-step value, continues the per-step trajectory cleanly
    #   mean_loss  — run summary (HuggingFace's canonical training_loss)
    #
    # Downstream consumers (SDK, chart tooling, DB metrics) can choose which
    # makes sense for their context.
    last_step_loss = None
    if trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry:
                last_step_loss = round(float(entry["loss"]), 4)
                break

    final_progress = {
        "step": trainer.state.global_step,
        "total_steps": trainer.state.max_steps,
        "epoch": round(trainer.state.epoch or 0, 3),
        "loss": last_step_loss,  # last-step loss — continues the per-step curve
        "mean_loss": round(float(train_output.training_loss), 4),  # run summary
        "learning_rate": None,  # scheduler has completed; no active LR
        "final": True,  # provenance: this is the summary, not a step log
    }
    print(f"\nPROGRESS:{json.dumps(final_progress)}", flush=True)
    # ──────────────────────────────────────────────────────────────────────────

    # 7. Save Artifacts
    print(f"💾 Exporting fine-tuned adapters to {args.out}...")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print("✨ Training Complete. Weights registered on Samba share.")


if __name__ == "__main__":
    main()
