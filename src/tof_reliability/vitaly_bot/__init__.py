"""Telegram simulator of a demanding TOF stakeholder named Vitaly.

The package deliberately keeps Telegram, persisted state, document preparation,
and model execution separate.  That makes the behavioural core testable without
access to Telegram or a ChatGPT subscription.
"""

from .profile import VitalyProfile, load_default_profile
from .storage import VitalyStore

__all__ = ["VitalyProfile", "VitalyStore", "load_default_profile"]
