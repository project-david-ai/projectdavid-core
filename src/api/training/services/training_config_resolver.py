# projectdavid-core — training config resolver
# Path suggestion: src/projectdavid/services/training_config_resolver.py

from typing import Any, Dict, Optional

from projectdavid_common.schemas.training_schema import TrainingConfig, TrainingProfile

# Canonical defaults. Represents the behaviour of the current codebase when
# no config is supplied — the values currently hardcoded in unsloth_train.py
# (SFTConfig + get_peft_model call sites). These are also the values baked
# into PROFILES["standard"] for profile-scoped fields, so an empty config
# reproduces the previous default-profile-standard behaviour.
BASE_DEFAULTS: Dict[str, Any] = {
    # Profile-scoped (overridable by profile preset):
    "max_seq_length": 2048,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_steps": 60,
    "optim": "adamw_8bit",
    # SFTConfig-scoped:
    "learning_rate": 2e-4,
    "warmup_steps": 2,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
    "logging_steps": 50,
    "num_train_epochs": 3,
    # PEFT-scoped:
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "bias": "none",
}

# Must match PROFILES in unsloth_train.py. Kept duplicated for Phase 1;
# Phase 2 cleanup should hoist this into a shared constants module imported
# by both the resolver and the trainer.
PROFILES: Dict[str, Dict[str, Any]] = {
    "laptop": {
        "max_seq_length": 1024,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_steps": 12500,
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


def resolve_training_config(user_config: Optional[TrainingConfig]) -> Dict[str, Any]:
    """
    Resolve a user-supplied TrainingConfig (or None) into a fully-specified
    config dict suitable for persistence on TrainingJob.config.

    Resolution order:
        1. BASE_DEFAULTS
        2. PROFILES[profile] (if profile is set)
        3. User field overrides (non-None fields only, excluding `profile`)
        4. lora_alpha = lora_r convention (if lora_r overridden and
           lora_alpha not explicitly set)

    The returned dict is the complete execution plan. Worker and trainer
    read from it without further resolution logic.
    """
    resolved: Dict[str, Any] = dict(BASE_DEFAULTS)

    if user_config is None:
        resolved["_profile"] = None
        return resolved

    # Apply profile preset
    profile_value: Optional[str] = None
    if user_config.profile is not None:
        profile_value = (
            user_config.profile.value
            if isinstance(user_config.profile, TrainingProfile)
            else str(user_config.profile)
        )
        resolved.update(PROFILES[profile_value])

    # Apply user overrides (non-None only). `profile` is not a config field.
    overrides = user_config.model_dump(exclude={"profile"}, exclude_none=True)
    resolved.update(overrides)

    # PEFT convention: lora_alpha defaults to lora_r when user bumps r but
    # leaves alpha unset. Checked against the raw user_config, not the
    # resolved dict (which always has lora_alpha from BASE_DEFAULTS).
    if user_config.lora_r is not None and user_config.lora_alpha is None:
        resolved["lora_alpha"] = user_config.lora_r

    # Audit provenance
    resolved["_profile"] = profile_value

    return resolved
