from __future__ import annotations

from datetime import datetime, timezone
import ast
import hashlib
import json
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
    user_approved: bool = False
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


class AppRegistryItem(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    endpoint: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    timestamp: datetime


class CalculatorRequest(BaseModel):
    expression: str


class CalculatorResponse(BaseModel):
    expression: str
    result: float


class ImageEditorRequest(BaseModel):
    asset: str
    action: str


class ImageEditorResponse(BaseModel):
    status: str
    pipeline: List[str]


class CalendarEvent(BaseModel):
    id: str
    title: str
    time: str
    location: str


class CalendarResponse(BaseModel):
    events: List[CalendarEvent]


class SSHRequest(BaseModel):
    command: str


class SSHResponse(BaseModel):
    output: str


MEMORY_STORE: Dict[str, MemoryRecord] = {}
VECTOR_INDEX: Optional[FAISS] = None
LOOKUP_LOGS: List[LookupLogRecord] = []

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "MyaOS")
LOOKUP_AUDIT_LOG_PATH = os.getenv("LOOKUP_AUDIT_LOG_PATH", "lookup_audit.log")
LOOKED_UP_MARKER = "[LOOKED_UP]"
EXTERNAL_LOOKUP_SOURCE_TAG = "external-lookup"

APP_REGISTRY = [
    {
        "id": "chat",
        "name": "Chat",
        "description": "Conversational co-pilot for MyaOS sessions.",
        "icon": "💬",
        "endpoint": "/apps/chat",
    },
    {
        "id": "calculator",
        "name": "Calculator",
        "description": "Scientific expressions & quick math checks.",
        "icon": "🧮",
        "endpoint": "/apps/calculator",
    },
    {
        "id": "image-editor",
        "name": "Image Editor",
        "description": "Queue edits for creative assets.",
        "icon": "🖼️",
        "endpoint": "/apps/image-editor",
    },
    {
        "id": "calendar",
        "name": "Calendar",
        "description": "Upcoming schedule and reminders.",
        "icon": "📅",
        "endpoint": "/apps/calendar",
    },
    {
        "id": "ssh",
        "name": "SSH Console",
        "description": "Issue remote commands via secure shell.",
        "icon": "🖥️",
        "endpoint": "/apps/ssh",
    },
]


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


_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
)


def _safe_calculate(expression: str) -> float:
    if not expression.strip():
        raise ValueError("Expression cannot be empty.")
    parsed = ast.parse(expression, mode="eval")
    for node in ast.walk(parsed):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError("Unsupported expression.")
    return float(eval(compile(parsed, "<calculator>", "eval"), {"__builtins__": {}}))


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


def _virtue_focus(virtue_markers: Dict[str, float]) -> str:
    if virtue_markers:
        return max(virtue_markers.items(), key=lambda item: item[1])[0]
    return "balance"


class LocalResponseEngine:
    def __init__(self) -> None:
        self.patterns: Sequence[Tuple[str, Sequence[Tuple[str, str]]]] = (
            (
                r"\b(hi|hello|hey|greetings)\b",
                (
                    (
                        "Hello! I’m tuned in and ready to help with {virtue} in mind.",
                        "Tell me what you want to explore or refine.",
                    ),
                    (
                        "Hey there—thanks for checking in.",
                        "Share the next task and I’ll map it out with {virtue}.",
                    ),
                ),
            ),
            (
                r"\b(thank(s)?|appreciate)\b",
                (
                    (
                        "You’re welcome, and I’m glad that landed well.",
                        "If there’s more to refine, we’ll keep it {virtue}-aligned.",
                    ),
                    (
                        "Happy to help!",
                        "Let me know the next step you want to tackle with {virtue}.",
                    ),
                ),
            ),
            (
                r"\b(help|support|assist)\b",
                (
                    (
                        "I can support with planning, drafting, or refining.",
                        "Point me at the goal and constraints so we keep {virtue} in view.",
                    ),
                    (
                        "I’m here to assist with clarity and structure.",
                        "Share the details you have so far and we’ll lead with {virtue}.",
                    ),
                ),
            ),
            (
                r"\b(why|how|what|when|where)\b",
                (
                    (
                        "That’s a good question, and I’ll answer it directly.",
                        "I’ll keep it practical and {virtue}-aligned for your goals.",
                    ),
                    (
                        "Let’s unpack that carefully.",
                        "I’ll focus on the most relevant details through {virtue}.",
                    ),
                ),
            ),
        )
        self.default_templates: Sequence[Tuple[str, str]] = (
            (
                "I’m ready to respond based on what you share.",
                "Give me the key details and intent, and we’ll keep {virtue} steady.",
            ),
            (
                "I’m listening and can shape a response around your intent.",
                "Share the main context or question, and we’ll ground it in {virtue}.",
            ),
        )

    def select_template(self, message: str, virtue_markers: Dict[str, float]) -> Tuple[str, str]:
        virtue = _virtue_focus(virtue_markers)
        for pattern, templates in self.patterns:
            if re.search(pattern, message, flags=re.IGNORECASE):
                idx = _stable_index(message, len(templates))
                return tuple(line.format(virtue=virtue) for line in templates[idx])
        idx = _stable_index(message, len(self.default_templates))
        return tuple(
            line.format(virtue=virtue) for line in self.default_templates[idx]
        )


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
        return f"{LOOKED_UP_MARKER} {joined}"
    return f"{LOOKED_UP_MARKER} External reference consulted."


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


class OpenRouterClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def chat_completion(self, request_body: Dict[str, object]) -> httpx.Response:
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.post(
                f"{self.base_url}/chat/completions",
                json=request_body,
                headers=_openrouter_headers(),
            )


def _ensure_source_tag(tags: List[str], tag: str) -> List[str]:
    if tag in tags:
        return tags
    return [*tags, tag]


def _guard_external_source_tags(tags: Sequence[str]) -> None:
    if EXTERNAL_LOOKUP_SOURCE_TAG in tags:
        raise HTTPException(
            status_code=400,
            detail="External knowledge must be stored via the /lookup endpoint.",
        )


def _write_lookup_audit_log(record: LookupLogRecord) -> None:
    payload = {
        "lookup_id": record.lookup_id,
        "created_at": record.created_at.isoformat(),
        "model": record.model,
        "query": record.query,
        "response": record.response,
        "response_prefixed": record.response_prefixed,
        "memory_id": record.memory_id,
    }
    with open(LOOKUP_AUDIT_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload) + "\n")


LOCAL_RESPONSE_ENGINE = LocalResponseEngine()
OPENROUTER_CLIENT = OpenRouterClient(OPENROUTER_BASE_URL)


def generate_local_response(payload: LocalResponseRequest) -> LocalResponseResponse:
    tone = _time_tone()
    time_line = _time_sentence(tone)
    virtue_line = _virtue_injection(payload.virtue_markers)
    template_first, template_second = LOCAL_RESPONSE_ENGINE.select_template(
        payload.message, payload.virtue_markers
    )
    sentences = [time_line, virtue_line, template_first, template_second]
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
    if payload.memory.store and not payload.memory.user_approved:
        raise HTTPException(
            status_code=400,
            detail="User approval is required before storing lookup responses in memory.",
        )
    request_body = {
        "model": payload.model,
        "messages": [{"role": "user", "content": payload.query}],
        "temperature": payload.temperature,
    }
    if payload.max_tokens is not None:
        request_body["max_tokens"] = payload.max_tokens

    response = await OPENROUTER_CLIENT.chat_completion(request_body)

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
            source_tags=_ensure_source_tag(payload.memory.source_tags, EXTERNAL_LOOKUP_SOURCE_TAG),
            virtue_markers=payload.memory.virtue_markers,
            salience=payload.memory.salience,
        )
        memory_id = _create_memory_record(memory_payload).metadata.memory_id
    log_record = LookupLogRecord(
        lookup_id=lookup_id,
        created_at=datetime.now(timezone.utc),
        model=payload.model,
        query=payload.query,
        response=raw_response,
        response_prefixed=prefixed_response,
        memory_id=memory_id,
    )
    LOOKUP_LOGS.append(log_record)
    _write_lookup_audit_log(log_record)

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
    _guard_external_source_tags(payload.source_tags)
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
        _guard_external_source_tags(payload.source_tags)
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
        _guard_external_source_tags(record.metadata.source_tags)
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
        _guard_external_source_tags(record.source_tags)
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


@app.get("/apps/registry", response_model=List[AppRegistryItem])
def app_registry() -> List[AppRegistryItem]:
    return [AppRegistryItem(**item) for item in APP_REGISTRY]


@app.post("/apps/chat", response_model=ChatResponse)
def chat_app(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    reply = (
        "MyaOS chat service received your message: "
        f"{message}. How else can I assist?"
    )
    return ChatResponse(reply=reply, timestamp=datetime.now(timezone.utc))


@app.post("/apps/calculator", response_model=CalculatorResponse)
def calculator_app(payload: CalculatorRequest) -> CalculatorResponse:
    try:
        result = _safe_calculate(payload.expression)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CalculatorResponse(expression=payload.expression, result=result)


@app.post("/apps/image-editor", response_model=ImageEditorResponse)
def image_editor_app(payload: ImageEditorRequest) -> ImageEditorResponse:
    pipeline = [
        f"Load asset '{payload.asset}'",
        f"Apply action '{payload.action}'",
        "Render preview",
        "Queue export",
    ]
    return ImageEditorResponse(status="Queued", pipeline=pipeline)


@app.get("/apps/calendar", response_model=CalendarResponse)
def calendar_app() -> CalendarResponse:
    events = [
        CalendarEvent(
            id="evt-101",
            title="Virtueism core sync",
            time="Today · 3:00 PM",
            location="Ops Bridge",
        ),
        CalendarEvent(
            id="evt-102",
            title="Memory lattice review",
            time="Tomorrow · 9:30 AM",
            location="Lab 7B",
        ),
        CalendarEvent(
            id="evt-103",
            title="Emotion engine retro",
            time="Friday · 1:00 PM",
            location="Studio C",
        ),
    ]
    return CalendarResponse(events=events)


@app.post("/apps/ssh", response_model=SSHResponse)
def ssh_app(payload: SSHRequest) -> SSHResponse:
    command = payload.command.strip()
    if not command:
        raise HTTPException(status_code=400, detail="Command cannot be empty.")
    output = (
        "myaos@remote:~$ "
        f"{command}\n"
        "total 8\n"
        "drwxr-xr-x  4 myaos staff  128 Sep 21 09:00 .\n"
        "drwxr-xr-x  8 myaos staff  256 Sep 21 08:55 ..\n"
        "-rw-r--r--  1 myaos staff 2048 Sep 21 08:59 system.log"
    )
    return SSHResponse(output=output)
