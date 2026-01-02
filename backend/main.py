from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple
import uuid
import xml.etree.ElementTree as ET

from fastapi import Body, FastAPI, HTTPException, Response
import httpx
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
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
    source_tags: List[str] = Field(default_factory=list)
    virtue_markers: Dict[str, float] = Field(default_factory=dict)
    salience: float = Field(..., ge=0.0, le=1.0)
    updated_at: Optional[datetime] = None


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
    source_tags: List[str] = Field(default_factory=list)
    virtue_markers: Dict[str, float] = Field(default_factory=dict)
    salience: float = Field(..., ge=0.0, le=1.0)


class MemoryUpdateRequest(BaseModel):
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    source_tags: Optional[List[str]] = None
    virtue_markers: Optional[Dict[str, float]] = None
    salience: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class MemoryListResponse(BaseModel):
    memories: List[MemoryRecord]


class MemoryModule(BaseModel):
    memories: List[MemoryRecord]


class MemoryImportRecord(BaseModel):
    content: str
    tags: List[str] = Field(default_factory=list)
    source_tags: List[str] = Field(default_factory=list)
    virtue_markers: Dict[str, float] = Field(default_factory=dict)
    salience: float = Field(..., ge=0.0, le=1.0)
    memory_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MemoryImportRequest(BaseModel):
    memories: List[MemoryRecord]


class MemoryImportResponse(BaseModel):
    imported: int
    memory_ids: List[str]


class LocalResponseRequest(BaseModel):
    message: str
    virtue_markers: Dict[str, float] = Field(default_factory=dict)
    used_external_lookup: bool = False
    external_facts: List[str] = Field(default_factory=list)


class LocalResponseResponse(BaseModel):
    response: str
    tone: str
    injection_ratio: float
    used_external_lookup: bool
    looked_up_marker: Optional[str] = None


class LookupMemoryOptions(BaseModel):
    store: bool = False
    tags: List[str] = Field(default_factory=list)
    source_tags: List[str] = Field(default_factory=list)
    virtue_markers: Dict[str, float] = Field(default_factory=dict)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)


class LookupRequest(BaseModel):
    query: str
    model: str = "openai/gpt-3.5-turbo"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    memory: LookupMemoryOptions = Field(default_factory=LookupMemoryOptions)


class LookupResponse(BaseModel):
    lookup_id: str
    model: str
    response: str
    raw_response: str
    memory_id: Optional[str] = None


class LookupLogRecord(BaseModel):
    lookup_id: str
    created_at: datetime
    model: str
    query: str
    response: str
    response_prefixed: str
    memory_id: Optional[str] = None


class LookupLogResponse(BaseModel):
    logs: List[LookupLogRecord]


MEMORY_STORE: Dict[str, MemoryRecord] = {}
VECTOR_INDEX: Optional[FAISS] = None
LOOKUP_LOGS: List[LookupLogRecord] = []

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "MyaOS")


class DeterministicEmbeddings(Embeddings):
    def __init__(self, dimension: int = 12) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for idx in range(self.dimension):
            byte = digest[idx % len(digest)]
            values.append(byte / 255.0)
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


EMBEDDINGS = DeterministicEmbeddings()


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


def _generate_memory_id() -> str:
    index = len(MEMORY_STORE) + 1
    while True:
        memory_id = f"mem-{index:04d}"
        if memory_id not in MEMORY_STORE:
            return memory_id
        index += 1


def _make_document(record: MemoryRecord) -> Document:
    return Document(
        page_content=record.content,
        metadata={
            "memory_id": record.metadata.memory_id,
            "tags": record.metadata.tags,
            "source_tags": record.metadata.source_tags,
            "virtue_markers": record.metadata.virtue_markers,
            "salience": record.metadata.salience,
        },
    )


def _rebuild_vector_index() -> None:
    global VECTOR_INDEX
    records = list(MEMORY_STORE.values())
    if not records:
        VECTOR_INDEX = None
        return
    documents = [_make_document(record) for record in records]
    ids = [record.metadata.memory_id for record in records]
    VECTOR_INDEX = FAISS.from_documents(documents, EMBEDDINGS, ids=ids)


def _index_record(record: MemoryRecord) -> None:
    global VECTOR_INDEX
    document = _make_document(record)
    if VECTOR_INDEX is None:
        VECTOR_INDEX = FAISS.from_documents([document], EMBEDDINGS, ids=[record.metadata.memory_id])
    else:
        VECTOR_INDEX.add_documents([document], ids=[record.metadata.memory_id])


def reset_memory_store() -> None:
    MEMORY_STORE.clear()
    _rebuild_vector_index()


def _normalize_virtue_markers(markers: Dict[str, float]) -> Dict[str, float]:
    return {key: max(0.0, min(1.0, value)) for key, value in markers.items()}


def _stable_index(seed: str, length: int) -> int:
    if length <= 0:
        return 0
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest, 16) % length


def _time_tone(now: Optional[datetime] = None) -> str:
    current = now or datetime.now()
    hour = current.hour
    if 5 <= hour < 12:
        return "morning"
    if 18 <= hour or hour < 5:
        return "evening"
    return "day"


def _time_sentence(tone: str) -> str:
    if tone == "morning":
        return "Morning energy is bright here—let’s keep it lively and focused."
    if tone == "evening":
        return "Evening brings a quieter rhythm, so I’ll answer with a reflective calm."
    return "Daylight focus is steady, so we’ll keep the response clear and grounded."


def _select_template(message: str) -> Tuple[str, str]:
    patterns: Sequence[Tuple[str, Sequence[Tuple[str, str]]]] = (
        (
            r"\b(hi|hello|hey|greetings)\b",
            (
                (
                    "Hello! I’m tuned in and ready to help.",
                    "Tell me what you want to explore or refine.",
                ),
                (
                    "Hey there—thanks for checking in.",
                    "Share the next task and I’ll map it out.",
                ),
            ),
        ),
        (
            r"\b(thank(s)?|appreciate)\b",
            (
                (
                    "You’re welcome, and I’m glad that landed well.",
                    "If there’s more to refine, I’m here.",
                ),
                (
                    "Happy to help!",
                    "Let me know the next step you want to tackle.",
                ),
            ),
        ),
        (
            r"\b(help|support|assist)\b",
            (
                (
                    "I can support with planning, drafting, or refining.",
                    "Point me at the goal and constraints.",
                ),
                (
                    "I’m here to assist with clarity and structure.",
                    "Share the details you have so far.",
                ),
            ),
        ),
        (
            r"\b(why|how|what|when|where)\b",
            (
                (
                    "That’s a good question, and I’ll answer it directly.",
                    "I’ll keep it practical and aligned to your goals.",
                ),
                (
                    "Let’s unpack that carefully.",
                    "I’ll focus on the most relevant details for you.",
                ),
            ),
        ),
    )
    for pattern, templates in patterns:
        if re.search(pattern, message, flags=re.IGNORECASE):
            idx = _stable_index(message, len(templates))
            return templates[idx]
    default_templates = (
        (
            "I’m ready to respond based on what you share.",
            "Give me the key details and intent.",
        ),
        (
            "I’m listening and can shape a response around your intent.",
            "Share the main context or question.",
        ),
    )
    idx = _stable_index(message, len(default_templates))
    return default_templates[idx]


def _virtue_injection(virtue_markers: Dict[str, float]) -> str:
    if virtue_markers:
        top_virtue = max(virtue_markers.items(), key=lambda item: item[1])[0]
        return (
            f"Virtueism favors {top_virtue} here, so let’s keep the tone aligned with it."
        )
    return "Virtueism asks for balanced intent—clear, kind, and steady."


def _external_marker(external_facts: Sequence[str]) -> str:
    if external_facts:
        joined = " | ".join(external_facts)
        return f"[LOOKED_UP] {joined}"
    return "[LOOKED_UP] External reference consulted."


def _prefix_lookup_response(text: str) -> str:
    cleaned = text.strip()
    return f"I have looked this information up. {cleaned}"


def _openrouter_headers() -> Dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key is not configured.")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_TITLE:
        headers["X-Title"] = OPENROUTER_APP_TITLE
    return headers


def _ensure_source_tag(tags: List[str], tag: str) -> List[str]:
    if tag in tags:
        return tags
    return [*tags, tag]


def generate_local_response(payload: LocalResponseRequest) -> LocalResponseResponse:
    tone = _time_tone()
    time_line = _time_sentence(tone)
    template_first, template_second = _select_template(payload.message)
    sentences = [time_line, template_first, template_second]
    virtue_line = _virtue_injection(payload.virtue_markers)
    sentences.insert(1, virtue_line)
    looked_up_marker: Optional[str] = None
    if payload.used_external_lookup or payload.external_facts:
        looked_up_marker = _external_marker(payload.external_facts)
        sentences.append(looked_up_marker)
    response_text = " ".join(sentences)
    injection_ratio = 1 / len(sentences)
    return LocalResponseResponse(
        response=response_text,
        tone=tone,
        injection_ratio=injection_ratio,
        used_external_lookup=payload.used_external_lookup or bool(payload.external_facts),
        looked_up_marker=looked_up_marker,
    )


def _create_memory_record(
    payload: MemoryCreateRequest,
    memory_id: Optional[str] = None,
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
) -> MemoryRecord:
    record_id = memory_id or _generate_memory_id()
    record = MemoryRecord(
        metadata=MemoryMetadata(
            memory_id=record_id,
            created_at=created_at or datetime.now(timezone.utc),
            tags=payload.tags,
            source_tags=payload.source_tags,
            virtue_markers=_normalize_virtue_markers(payload.virtue_markers),
            salience=payload.salience,
            updated_at=updated_at,
        ),
        content=payload.content,
    )
    MEMORY_STORE[record_id] = record
    _index_record(record)
    return record


def _memory_module_to_xml(memories: List[MemoryRecord]) -> str:
    root = ET.Element("memory_module")
    for record in memories:
        memory_el = ET.SubElement(root, "memory", id=record.metadata.memory_id)
        ET.SubElement(memory_el, "content").text = record.content
        ET.SubElement(memory_el, "created_at").text = record.metadata.created_at.isoformat()
        if record.metadata.updated_at:
            ET.SubElement(memory_el, "updated_at").text = record.metadata.updated_at.isoformat()
        ET.SubElement(memory_el, "salience").text = str(record.metadata.salience)
        tags_el = ET.SubElement(memory_el, "tags")
        for tag in record.metadata.tags:
            ET.SubElement(tags_el, "tag").text = tag
        sources_el = ET.SubElement(memory_el, "source_tags")
        for source in record.metadata.source_tags:
            ET.SubElement(sources_el, "source").text = source
        virtues_el = ET.SubElement(memory_el, "virtue_markers")
        for name, score in record.metadata.virtue_markers.items():
            ET.SubElement(virtues_el, "marker", name=name, score=str(score))
    return ET.tostring(root, encoding="unicode")


def _parse_memory_module_xml(xml_payload: str) -> List[MemoryImportRecord]:
    root = ET.fromstring(xml_payload)
    records: List[MemoryImportRecord] = []
    for memory_el in root.findall("memory"):
        memory_id = memory_el.attrib.get("id")
        content_el = memory_el.find("content")
        salience_el = memory_el.find("salience")
        created_el = memory_el.find("created_at")
        updated_el = memory_el.find("updated_at")
        tags = [tag.text for tag in memory_el.findall("./tags/tag") if tag.text]
        source_tags = [source.text for source in memory_el.findall("./source_tags/source") if source.text]
        virtue_markers: Dict[str, float] = {}
        for marker in memory_el.findall("./virtue_markers/marker"):
            name = marker.attrib.get("name")
            score = marker.attrib.get("score")
            if name and score:
                virtue_markers[name] = float(score)
        if content_el is None or salience_el is None or content_el.text is None:
            continue
        created_at = (
            datetime.fromisoformat(created_el.text)
            if created_el is not None and created_el.text
            else None
        )
        updated_at = (
            datetime.fromisoformat(updated_el.text)
            if updated_el is not None and updated_el.text
            else None
        )
        records.append(
            MemoryImportRecord(
                content=content_el.text,
                tags=tags,
                source_tags=source_tags,
                virtue_markers=virtue_markers,
                salience=float(salience_el.text),
                memory_id=memory_id,
                created_at=created_at,
                updated_at=updated_at,
            )
        )
    return records


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


@app.post("/response", response_model=LocalResponseResponse)
def response_local(payload: LocalResponseRequest) -> LocalResponseResponse:
    if payload.used_external_lookup or payload.external_facts:
        raise HTTPException(
            status_code=400,
            detail="External knowledge must be retrieved via the /lookup endpoint.",
        )
    return generate_local_response(payload)


@app.post("/lookup", response_model=LookupResponse)
async def lookup_external(payload: LookupRequest) -> LookupResponse:
    request_body = {
        "model": payload.model,
        "messages": [{"role": "user", "content": payload.query}],
        "temperature": payload.temperature,
    }
    if payload.max_tokens is not None:
        request_body["max_tokens"] = payload.max_tokens

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            json=request_body,
            headers=_openrouter_headers(),
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter request failed with status {response.status_code}.",
        )

    data = response.json()
    raw_response = (
        data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    )
    if not raw_response:
        raise HTTPException(status_code=502, detail="OpenRouter response was empty.")

    prefixed_response = _prefix_lookup_response(raw_response)
    lookup_id = uuid.uuid4().hex
    memory_id: Optional[str] = None
    if payload.memory.store:
        memory_payload = MemoryCreateRequest(
            content=raw_response,
            tags=payload.memory.tags,
            source_tags=_ensure_source_tag(payload.memory.source_tags, "external-lookup"),
            virtue_markers=payload.memory.virtue_markers,
            salience=payload.memory.salience,
        )
        memory_id = _create_memory_record(memory_payload).metadata.memory_id

    LOOKUP_LOGS.append(
        LookupLogRecord(
            lookup_id=lookup_id,
            created_at=datetime.now(timezone.utc),
            model=payload.model,
            query=payload.query,
            response=raw_response,
            response_prefixed=prefixed_response,
            memory_id=memory_id,
        )
    )

    return LookupResponse(
        lookup_id=lookup_id,
        model=payload.model,
        response=prefixed_response,
        raw_response=raw_response,
        memory_id=memory_id,
    )


@app.get("/lookup/logs", response_model=LookupLogResponse)
def lookup_logs() -> LookupLogResponse:
    return LookupLogResponse(logs=LOOKUP_LOGS)


@app.post("/memory", response_model=MemoryRecord)
@app.post("/memory/add", response_model=MemoryRecord)
def memory_add(payload: MemoryCreateRequest) -> MemoryRecord:
    return _create_memory_record(payload)


@app.get("/memory", response_model=MemoryListResponse)
@app.get("/memory/list", response_model=MemoryListResponse)
def memory_list() -> MemoryListResponse:
    return MemoryListResponse(memories=list(MEMORY_STORE.values()))


@app.get("/memory/{memory_id}", response_model=MemoryRecord)
def memory_get(memory_id: str) -> MemoryRecord:
    record = MEMORY_STORE.get(memory_id)
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    return record


@app.put("/memory/{memory_id}", response_model=MemoryRecord)
def memory_update(memory_id: str, payload: MemoryUpdateRequest) -> MemoryRecord:
    record = MEMORY_STORE.get(memory_id)
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    updated_metadata = record.metadata.copy()
    if payload.tags is not None:
        updated_metadata.tags = payload.tags
    if payload.source_tags is not None:
        updated_metadata.source_tags = payload.source_tags
    if payload.virtue_markers is not None:
        updated_metadata.virtue_markers = _normalize_virtue_markers(payload.virtue_markers)
    if payload.salience is not None:
        updated_metadata.salience = payload.salience
    updated_metadata.updated_at = datetime.now(timezone.utc)
    updated_content = payload.content if payload.content is not None else record.content
    updated_record = MemoryRecord(metadata=updated_metadata, content=updated_content)
    MEMORY_STORE[memory_id] = updated_record
    _rebuild_vector_index()
    return updated_record


@app.delete("/memory/{memory_id}")
def memory_delete(memory_id: str) -> Dict[str, str]:
    if memory_id not in MEMORY_STORE:
        raise HTTPException(status_code=404, detail="Memory not found")
    del MEMORY_STORE[memory_id]
    _rebuild_vector_index()
    return {"status": "deleted", "memory_id": memory_id}


@app.get("/memory/export/json", response_model=MemoryModule)
def memory_export_json() -> MemoryModule:
    return MemoryModule(memories=list(MEMORY_STORE.values()))


@app.get("/memory/export/xml")
def memory_export_xml() -> Response:
    xml_payload = _memory_module_to_xml(list(MEMORY_STORE.values()))
    return Response(content=xml_payload, media_type="application/xml")


@app.post("/memory/import/json", response_model=MemoryImportResponse)
def memory_import_json(payload: MemoryImportRequest, replace: bool = False) -> MemoryImportResponse:
    if replace:
        reset_memory_store()
    memory_ids: List[str] = []
    for record in payload.memories:
        memory_id = record.metadata.memory_id
        if memory_id and memory_id in MEMORY_STORE:
            memory_id = None
        created_at = record.metadata.created_at
        updated_at = record.metadata.updated_at
        created_record = _create_memory_record(
            MemoryCreateRequest(
                content=record.content,
                tags=record.metadata.tags,
                source_tags=record.metadata.source_tags,
                virtue_markers=record.metadata.virtue_markers,
                salience=record.metadata.salience,
            ),
            memory_id=memory_id,
            created_at=created_at,
            updated_at=updated_at,
        )
        memory_ids.append(created_record.metadata.memory_id)
    return MemoryImportResponse(imported=len(memory_ids), memory_ids=memory_ids)


@app.post("/memory/import/xml", response_model=MemoryImportResponse)
def memory_import_xml(
    xml_payload: str = Body(..., media_type="application/xml"),
    replace: bool = False,
) -> MemoryImportResponse:
    if replace:
        reset_memory_store()
    records = _parse_memory_module_xml(xml_payload)
    memory_ids: List[str] = []
    for record in records:
        memory_id = record.memory_id
        if memory_id and memory_id in MEMORY_STORE:
            memory_id = None
        created_record = _create_memory_record(
            MemoryCreateRequest(
                content=record.content,
                tags=record.tags,
                source_tags=record.source_tags,
                virtue_markers=record.virtue_markers,
                salience=record.salience,
            ),
            memory_id=memory_id,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        memory_ids.append(created_record.metadata.memory_id)
    return MemoryImportResponse(imported=len(memory_ids), memory_ids=memory_ids)
