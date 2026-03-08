import math
import random
from typing import TYPE_CHECKING

from .topic_interest import has_active_topic_interest_match
from .states import machine_pool, State
from ..utils import logger, APP_CONFIG

if TYPE_CHECKING:
    from .states.heat import HeatSnapshot


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _normalized_trend(
    velocity: float,
    *,
    activate: float,
    deactivate: float,
    half_life: float,
) -> float:
    heat_scale = max(activate - deactivate, deactivate, 1e-6)
    normalized = velocity * half_life / heat_scale
    return _clamp(normalized, -4.0, 4.0)


def _compute_proactive_probability(heat_snapshot: "HeatSnapshot") -> float:
    if heat_snapshot.msg_ema <= 0.0 or heat_snapshot.n_participants <= 0:
        return 0.0

    activate = max(float(APP_CONFIG.heat_activate_threshold), 1e-6)
    deactivate = max(float(APP_CONFIG.heat_deactivate_threshold), 1e-6)
    half_life = max(float(APP_CONFIG.heat_half_life_seconds), 1e-6)
    heat = max(0.0, heat_snapshot.heat)
    participants = max(1, heat_snapshot.n_participants)

    stage_progress = _clamp(heat / activate, 0.0, 3.0)
    cooling_progress = _clamp((activate - heat) / activate, 0.0, 1.0)
    trend = _normalized_trend(
        heat_snapshot.velocity,
        activate=activate,
        deactivate=deactivate,
        half_life=half_life,
    )
    rise_signal = _sigmoid(trend)
    cool_signal = _sigmoid(-trend)
    participant_pressure = _clamp((participants - 1) / 5.0, 0.0, 1.0)
    recency = _clamp(
        heat_snapshot.msg_ema / (deactivate * participants + 1e-6), 0.0, 2.5
    )

    opening_score = (1.0 - _clamp(stage_progress, 0.0, 1.0)) * (
        1.0 - participant_pressure
    )
    cold_start_bonus = (
        (1.0 if heat_snapshot.state is State.IDLE else 0.0)
        * (1.0 - _clamp(stage_progress / 0.8, 0.0, 1.0))
        * (1.0 - participant_pressure)
    )
    cooling_score = cooling_progress * cool_signal
    inactive_cooling_bonus = (
        (1.0 if heat_snapshot.state is State.INACTIVE else 0.0)
        * cooling_progress
        * cool_signal
    )
    hot_suppression = _clamp(stage_progress - 0.8, 0.0, 1.5) / 1.5
    growth_suppression = rise_signal * _clamp(stage_progress, 0.0, 1.0)

    probability = 0.008
    probability += 0.022 * opening_score
    probability += 0.001 * cold_start_bonus
    probability += 0.018 * cooling_score
    probability += 0.012 * inactive_cooling_bonus
    probability += 0.008 * (recency / 2.5)
    probability += 0.004 * (1.0 if heat_snapshot.state is State.INACTIVE else 0.0)
    probability -= 0.02 * participant_pressure
    probability -= 0.03 * hot_suppression
    probability -= 0.012 * growth_suppression

    return _clamp(probability, 0.0, 0.05)


async def should_proactively_respond(
    *,
    agent_id: str,
    user_id: str,
    group_id: str | None,
    plain_text: str,
) -> bool:
    """统一的主动发言判断入口。"""
    topic_interest = await has_active_topic_interest_match(
        agent_id=agent_id,
        user_id=user_id,
        group_id=group_id,
        plain_text=plain_text,
    )

    if topic_interest:
        return True

    heat_snapshot = machine_pool.get_snapshot(
        agent_id=agent_id, group_id=group_id, user_id=user_id
    )

    if not APP_CONFIG.heat_enable:
        return False

    probability = _compute_proactive_probability(heat_snapshot)
    logger.debug(
        f"Proactive probability computed as {probability:.4f} for state {heat_snapshot.state}"
    )
    return random.random() < probability
