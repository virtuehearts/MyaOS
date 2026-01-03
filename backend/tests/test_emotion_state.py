import pytest

from backend.main import (
    EmotionState,
    PersonalityTraits,
    _deterministic_event_signal,
    update_emotion_state,
)


def test_event_signal_is_deterministic() -> None:
    assert _deterministic_event_signal("hello") == _deterministic_event_signal("hello")
    assert _deterministic_event_signal("") == 0.0


def test_update_emotion_state_influences_traits() -> None:
    current = EmotionState(valence=0.5, arousal=0.5, dominance=0.5)
    traits = PersonalityTraits(
        openness=0.6,
        conscientiousness=0.7,
        extraversion=0.8,
        agreeableness=0.4,
        neuroticism=0.2,
    )
    new_state = update_emotion_state(current, traits, "A")
    assert new_state.valence == pytest.approx(0.5975)
    assert new_state.arousal == pytest.approx(0.53)
    assert new_state.dominance == pytest.approx(0.52)


def test_update_emotion_state_clamps_range() -> None:
    current = EmotionState(valence=0.99, arousal=0.01, dominance=0.99)
    traits = PersonalityTraits(
        openness=1.0,
        conscientiousness=1.0,
        extraversion=1.0,
        agreeableness=0.0,
        neuroticism=0.0,
    )
    new_state = update_emotion_state(current, traits, "!!!!!")
    assert 0.0 <= new_state.valence <= 1.0
    assert 0.0 <= new_state.arousal <= 1.0
    assert 0.0 <= new_state.dominance <= 1.0
