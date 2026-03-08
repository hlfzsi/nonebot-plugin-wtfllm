__all__ = [
    "State",
    "HeatSnapshot",
    "TransitionEvent",
    "MachineConfig",
    "HeatMachine",
    "MachineController",
    "MachinePool",
    "SessionKey",
    "build_machine_config",
    "machine_pool",
]

from ....config import APP_CONFIG
from ._types import HeatSnapshot, MachineConfig, State, TransitionEvent
from .controller import MachineController
from .machine import HeatMachine
from .pool import MachinePool
from .utils import SessionKey


def _config_number(name: str, default: float) -> float:
    value = getattr(APP_CONFIG, name, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def build_machine_config() -> MachineConfig:
    return MachineConfig(
        half_life=_config_number("heat_half_life_seconds", 300.0),
        activate_threshold=_config_number("heat_activate_threshold", 2.0),
        deactivate_threshold=_config_number("heat_deactivate_threshold", 0.5),
        idle_timeout=_config_number("heat_idle_timeout_seconds", 1800.0),
        velocity_alpha=_config_number("heat_velocity_alpha", 0.3),
        base_increment=_config_number("heat_base_increment", 1.0),
        participant_decay_threshold=_config_number(
            "heat_participant_decay_threshold",
            0.1,
        ),
    )


machine_pool = MachinePool(build_machine_config())