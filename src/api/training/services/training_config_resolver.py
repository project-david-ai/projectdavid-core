# projectdavid-core — training config resolver
# Path suggestion: src/projectdavid/services/training_config_resolver.py

from typing import Any, Dict, Optional

from projectdavid_common.constants import BASE_DEFAULTS, PROFILES
from projectdavid_common.schemas.training_schema import TrainingConfig, TrainingProfile


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

    BASE_DEFAULTS and PROFILES are the canonical dicts exported from
    projectdavid_common.constants — the trainer safety-net fallbacks in
    unsloth_train.py import the same objects, so there is no possible
    drift between resolver and trainer.
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
