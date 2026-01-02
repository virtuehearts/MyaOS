from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="MyaOS Backend", version="0.1.0")


class EmotionState(BaseModel):
    valence: float = Field(..., ge=0.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)
    dominance: float = Field(..., ge=0.0, le=1.0)


class PersonalityTraits(BaseModel):
    openness: float = Field(..., ge=0.0, le=1.0)
    conscientiousness: float = Field(..., ge=0.0, le=1.0)
    extraversion: float = Field(..., ge=0.0, le=1.0)
    agreeableness: float = Field(..., ge=0.0, le=1.0)
    neuroticism: float = Field(..., ge=0.0, le=1.0)


class MemoryMetadata(BaseModel):
    memory_id: str
    created_at: datetime
    tags: List[str] = Field(default_factory=list)
    salience: float = Field(..., ge=0.0, le=1.0)


class MemoryRecord(BaseModel):
    metadata: MemoryMetadata
    content: str


class StateUpdateRequest(BaseModel):
    current_state: EmotionState
    traits: PersonalityTraits
    event: str


class StateUpdateResponse(BaseModel):
    new_state: EmotionState
    explanation: str


class MemoryCreateRequest(BaseModel):
    content: str
    tags: List[str] = Field(default_factory=list)
    salience: float = Field(..., ge=0.0, le=1.0)


class MemoryListResponse(BaseModel):
    memories: List[MemoryRecord]


MEMORY_STORE: Dict[str, MemoryRecord] = {}


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _deterministic_event_signal(event: str) -> float:
    if not event:
        return 0.0
    signal = sum(ord(ch) for ch in event) % 100
    return signal / 100.0


def update_emotion_state(
    current_state: EmotionState, traits: PersonalityTraits, event: str
) -> EmotionState:
    signal = _deterministic_event_signal(event)
    valence_shift = (traits.extraversion - traits.neuroticism) * 0.15
    arousal_shift = (signal - 0.5) * 0.2
    dominance_shift = (traits.conscientiousness - 0.5) * 0.1

    new_valence = _clamp(current_state.valence + valence_shift + (signal - 0.5) * 0.05)
    new_arousal = _clamp(current_state.arousal + arousal_shift)
    new_dominance = _clamp(current_state.dominance + dominance_shift)

    return EmotionState(
        valence=new_valence,
        arousal=new_arousal,
        dominance=new_dominance,
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/state/update", response_model=StateUpdateResponse)
def state_update(payload: StateUpdateRequest) -> StateUpdateResponse:
    new_state = update_emotion_state(payload.current_state, payload.traits, payload.event)
    return StateUpdateResponse(
        new_state=new_state,
        explanation="Deterministic placeholder update based on traits and event signal.",
    )


@app.post("/memory/add", response_model=MemoryRecord)
def memory_add(payload: MemoryCreateRequest) -> MemoryRecord:
    memory_id = f"mem-{len(MEMORY_STORE) + 1:04d}"
    record = MemoryRecord(
        metadata=MemoryMetadata(
            memory_id=memory_id,
            created_at=datetime.now(timezone.utc),
            tags=payload.tags,
            salience=payload.salience,
        ),
        content=payload.content,
    )
    MEMORY_STORE[memory_id] = record
    return record


@app.get("/memory/list", response_model=MemoryListResponse)
def memory_list() -> MemoryListResponse:
    return MemoryListResponse(memories=list(MEMORY_STORE.values()))


@app.get("/memory/{memory_id}", response_model=MemoryRecord)
def memory_get(memory_id: str) -> MemoryRecord:
    record = MEMORY_STORE.get(memory_id)
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    return record
