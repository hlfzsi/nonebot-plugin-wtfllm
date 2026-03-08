import importlib
from types import SimpleNamespace


def test_machine_pool_uses_app_config(monkeypatch) -> None:
    import nonebot_plugin_wtfllm.config as config_module
    import nonebot_plugin_wtfllm.proactive.states.heat as heat_module
    original_config = config_module.APP_CONFIG

    custom_config = SimpleNamespace(
        heat_half_life_seconds=123.0,
        heat_activate_threshold=3.2,
        heat_deactivate_threshold=0.8,
        heat_idle_timeout_seconds=456.0,
        heat_velocity_alpha=0.6,
        heat_base_increment=1.7,
        heat_participant_decay_threshold=0.25,
    )

    monkeypatch.setattr(config_module, "APP_CONFIG", custom_config)
    heat_module = importlib.reload(heat_module)

    try:
        cfg = heat_module.machine_pool._cfg
        assert cfg.half_life == 123.0
        assert cfg.activate_threshold == 3.2
        assert cfg.deactivate_threshold == 0.8
        assert cfg.idle_timeout == 456.0
        assert cfg.velocity_alpha == 0.6
        assert cfg.base_increment == 1.7
        assert cfg.participant_decay_threshold == 0.25
    finally:
        monkeypatch.setattr(config_module, "APP_CONFIG", original_config)
        importlib.reload(heat_module)