"""_types.py 数据类型单元测试"""

import numpy as np
import pytest

from nonebot_plugin_wtfllm.proactive.inertia._types import (
    AnomalyPoint,
    AnomalyType,
    ActivityCurve,
    SessionKey,
)


# ===================== SessionKey =====================


class TestSessionKey:

    def test_group_session(self):
        sk = SessionKey(group_id="g1", sender="user_a")
        assert sk.group_id == "g1"
        assert sk.user_id is None
        assert sk.sender == "user_a"

    def test_private_session(self):
        sk = SessionKey(user_id="user_a")
        assert sk.group_id is None
        assert sk.user_id == "user_a"
        assert sk.sender == ""

    def test_is_group(self):
        assert SessionKey(group_id="g1", sender="u1").is_group is True
        assert SessionKey(user_id="u1").is_group is False

    def test_hash_eq_same(self):
        a = SessionKey(group_id="g1", sender="u1")
        b = SessionKey(group_id="g1", sender="u1")
        assert a == b
        assert hash(a) == hash(b)

    def test_hash_neq_different_group(self):
        a = SessionKey(group_id="g1", sender="u1")
        b = SessionKey(group_id="g2", sender="u1")
        assert a != b

    def test_hash_neq_group_vs_private(self):
        a = SessionKey(group_id="g1", sender="user_a")
        b = SessionKey(user_id="user_a")
        assert a != b

    def test_usable_as_dict_key(self):
        sk = SessionKey(group_id="g1", sender="u1")
        d = {sk: 42}
        assert d[SessionKey(group_id="g1", sender="u1")] == 42

    def test_eq_with_non_session_key(self):
        sk = SessionKey(group_id="g1", sender="u1")
        assert sk != "not a session key"

    def test_target_id_group(self):
        sk = SessionKey(group_id="g1", sender="u1")
        assert sk.target_id == "g1"

    def test_target_id_private(self):
        sk = SessionKey(user_id="u1")
        assert sk.target_id == "u1"

    def test_target_id_invalid(self):
        with pytest.raises(ValueError):
            SessionKey().target_id

    def test_group_user_key(self):
        assert SessionKey(group_id="g1", sender="u1").group_user_key == ("g1", "u1")
        assert SessionKey(user_id="u1").group_user_key == (None, "u1")


# ===================== AnomalyType =====================


class TestAnomalyType:

    def test_values(self):
        assert AnomalyType.UNEXPECTED_ACTIVITY.value == "unexpected_activity"
        assert AnomalyType.UNEXPECTED_ABSENCE.value == "unexpected_absence"

    def test_enum_members(self):
        assert len(AnomalyType) == 2


# ===================== ActivityCurve =====================


class TestActivityCurve:

    def _make_curve(self, **overrides):
        defaults = dict(
            session=SessionKey(group_id="g1", sender="u1"),
            predicted_median=np.zeros(1440, dtype=np.float32),
            predicted_lower=np.zeros(1440, dtype=np.float32),
            predicted_upper=np.ones(1440, dtype=np.float32),
            r_squared=0.85,
            data_quality=0.7,
        )
        defaults.update(overrides)
        return ActivityCurve(**defaults)

    def test_valid_construction(self):
        curve = self._make_curve()
        assert curve.r_squared == 0.85
        assert curve.data_quality == 0.7
        assert curve.predicted_median.shape == (1440,)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="1440"):
            self._make_curve(predicted_median=np.zeros(100, dtype=np.float32))

    def test_arrays_are_correct_dtype(self):
        curve = self._make_curve()
        assert curve.predicted_median.dtype == np.float32
        assert curve.predicted_lower.dtype == np.float32
        assert curve.predicted_upper.dtype == np.float32


# ===================== AnomalyPoint =====================


class TestAnomalyPoint:

    def test_construction(self):
        ap = AnomalyPoint(
            minute_of_day=150,
            anomaly_type=AnomalyType.UNEXPECTED_ACTIVITY,
            deviation_score=2.5,
            expected_median=0.1,
            expected_range=(0.0, 0.3),
            reason="测试原因",
        )
        assert ap.minute_of_day == 150
        assert ap.anomaly_type == AnomalyType.UNEXPECTED_ACTIVITY
        assert ap.deviation_score == 2.5
        assert ap.expected_range == (0.0, 0.3)
        assert ap.reason == "测试原因"

    def test_absence_type(self):
        ap = AnomalyPoint(
            minute_of_day=600,
            anomaly_type=AnomalyType.UNEXPECTED_ABSENCE,
            deviation_score=1.0,
            expected_median=5.0,
            expected_range=(3.0, 8.0),
            reason="缺席",
        )
        assert ap.anomaly_type == AnomalyType.UNEXPECTED_ABSENCE
