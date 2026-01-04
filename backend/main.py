from __future__ import annotations

import ast
import base64
from contextlib import contextmanager
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import uuid
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
import xml.etree.ElementTree as ET

from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pymysql
from pydantic import BaseModel, Field

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "MyaOS")
DEFAULT_OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_DEFAULT_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free"
)
OPENROUTER_CONTEXT_WINDOW_TOKENS = int(
    os.getenv("OPENROUTER_CONTEXT_WINDOW_TOKENS", "128000")
)
OPENROUTER_RESERVED_CONTEXT_TOKENS = int(
    os.getenv("OPENROUTER_RESERVED_CONTEXT_TOKENS", "32000")
)
OPENROUTER_MAX_COMPLETION_TOKENS = max(
    0, OPENROUTER_CONTEXT_WINDOW_TOKENS - OPENROUTER_RESERVED_CONTEXT_TOKENS
)
LOOKUP_AUDIT_LOG_PATH = os.getenv("LOOKUP_AUDIT_LOG_PATH", "lookup_audit.log")
LOOKED_UP_MARKER = "[LOOKED_UP]"
EXTERNAL_LOOKUP_SOURCE_TAG = "external-lookup"
AUTH_SECRET = os.getenv("AUTH_SECRET", "dev-secret-change-me")
AUTH_SESSION_TTL_MINUTES = int(os.getenv("AUTH_SESSION_TTL_MINUTES", "120"))
AUTH_STATE_TTL_SECONDS = int(os.getenv("AUTH_STATE_TTL_SECONDS", "600"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
FRONTEND_OAUTH_REDIRECT = os.getenv(
    "FRONTEND_OAUTH_REDIRECT", "http://localhost:3000/auth/callback"
)
OAUTH_REDIRECT_URI = os.getenv(
    "OAUTH_REDIRECT_URI", "http://localhost:8000/auth/oauth/callback"
)
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "myaos")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "myaos_password")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "myaos")
TRAIT_ANALYSIS_INTERVAL_SECONDS = int(
    os.getenv("TRAIT_ANALYSIS_INTERVAL_SECONDS", "300")
)
TRAIT_HISTORY_LIMIT = int(os.getenv("TRAIT_HISTORY_LIMIT", "120"))
TRAIT_ANALYSIS_ENABLED = os.getenv("TRAIT_ANALYSIS_ENABLED", "true").lower() == "true"
REFLECTION_SALIENCE_THRESHOLD = float(
    os.getenv("REFLECTION_SALIENCE_THRESHOLD", "0.8")
)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("myaos")

app = FastAPI(title="MyaOS Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmotionState(BaseModel):
    valence: float = Field(..., ge=0.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)
    dominance: float = Field(..., ge=0.0, le=1.0)
    label: str = "neutral"
    intensity: float = Field(0.0, ge=0.0, le=1.0)
    secondary_emotions: Dict[str, float] = Field(default_factory=dict)
    updated_at: Optional[datetime] = None


class EmotionTransition(BaseModel):
    timestamp: datetime
    event: str
    previous_state: EmotionState
    new_state: EmotionState


class PersonalityTraits(BaseModel):
    openness: float = Field(..., ge=0.0, le=1.0)
    conscientiousness: float = Field(..., ge=0.0, le=1.0)
    extraversion: float = Field(..., ge=0.0, le=1.0)
    agreeableness: float = Field(..., ge=0.0, le=1.0)
    neuroticism: float = Field(..., ge=0.0, le=1.0)


class PersonalityTraitSnapshot(BaseModel):
    snapshot_at: datetime
    traits: PersonalityTraits
    signals: Dict[str, float] = Field(default_factory=dict)


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


class EmotionSnapshot(BaseModel):
    state: EmotionState
    transitions: List[EmotionTransition]
    routine_phase: str = "unspecified"


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
    emotion_state: EmotionState
    emotion_baseline: EmotionState
    personality_traits: PersonalityTraits
    emotion_transitions: List[EmotionTransition]
    reflections: List["ReflectionRecord"] = Field(default_factory=list)
    routine_phase: str = "unspecified"
    trait_history: List[PersonalityTraitSnapshot] = Field(default_factory=list)


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
    emotion_state: Optional[EmotionState] = None
    emotion_baseline: Optional[EmotionState] = None
    personality_traits: Optional[PersonalityTraits] = None
    emotion_transitions: List[EmotionTransition] = Field(default_factory=list)
    routine_phase: Optional[str] = None
    trait_history: List[PersonalityTraitSnapshot] = Field(default_factory=list)


class MemoryModuleImportData(BaseModel):
    memories: List[MemoryImportRecord]
    emotion_state: Optional[EmotionState] = None
    emotion_baseline: Optional[EmotionState] = None
    personality_traits: Optional[PersonalityTraits] = None
    emotion_transitions: List[EmotionTransition] = Field(default_factory=list)
    routine_phase: Optional[str] = None


class ReflectionRecord(BaseModel):
    reflection_id: str
    created_at: datetime
    event_type: str
    content: str
    context: Dict[str, object] = Field(default_factory=dict)


class MemoryImportResponse(BaseModel):
    imported: int
    memory_ids: List[str]


class UserProfile(BaseModel):
    user_id: str
    email: str
    name: Optional[str] = None
    created_at: datetime


class AuthRegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class AuthLoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: UserProfile
    expires_at: datetime


class AuthMeResponse(BaseModel):
    user: UserProfile


class OAuthStartRequest(BaseModel):
    provider: str


class OAuthStartResponse(BaseModel):
    provider: str
    auth_url: str
    state: str


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
    model: str = DEFAULT_OPENROUTER_MODEL
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


class PasswordRecord(BaseModel):
    salt: str
    digest: str


class SessionRecord(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime


USERS_BY_EMAIL: Dict[str, UserProfile] = {}
USERS_BY_ID: Dict[str, UserProfile] = {}
USER_PASSWORDS: Dict[str, PasswordRecord] = {}
SESSIONS: Dict[str, SessionRecord] = {}
OAUTH_STATES: Dict[str, Dict[str, str]] = {}


OAUTH_PROVIDERS: Dict[str, Dict[str, Optional[str]]] = {
    "google": {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
        "scope": "openid email profile",
    },
    "github": {
        "client_id": os.getenv("GITHUB_CLIENT_ID"),
        "client_secret": os.getenv("GITHUB_CLIENT_SECRET"),
        "auth_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
        "emails_url": "https://api.github.com/user/emails",
        "scope": "read:user user:email",
    },
}


class RateLimiter:
    def __init__(self) -> None:
        self.hits: Dict[str, List[float]] = {}

    def check(self, key: str, limit: int, window_seconds: int) -> None:
        now = time.time()
        window_start = now - window_seconds
        timestamps = [ts for ts in self.hits.get(key, []) if ts > window_start]
        if len(timestamps) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please slow down and try again.",
            )
        timestamps.append(now)
        self.hits[key] = timestamps


RATE_LIMITER = RateLimiter()
USER_LOOKUP_LOGS: Dict[str, List[LookupLogRecord]] = {}
VECTOR_INDEX: Dict[str, FAISS] = {}
VECTOR_INDEX_KEY = "global"


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_password(password: str, salt: Optional[str] = None) -> PasswordRecord:
    resolved_salt = salt or secrets.token_hex(8)
    digest = hashlib.sha256(f"{resolved_salt}{password}".encode("utf-8")).hexdigest()
    return PasswordRecord(salt=resolved_salt, digest=digest)


def _verify_password(password: str, record: PasswordRecord) -> bool:
    digest = hashlib.sha256(f"{record.salt}{password}".encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, record.digest)


def _create_user(email: str, name: Optional[str], password: Optional[str]) -> UserProfile:
    normalized = _normalize_email(email)
    if normalized in USERS_BY_EMAIL:
        raise HTTPException(status_code=400, detail="Email is already registered.")
    user = UserProfile(
        user_id=uuid.uuid4().hex,
        email=normalized,
        name=name,
        created_at=datetime.now(timezone.utc),
    )
    USERS_BY_EMAIL[normalized] = user
    USERS_BY_ID[user.user_id] = user
    if password:
        USER_PASSWORDS[user.user_id] = _hash_password(password)
    return user


def _find_or_create_oauth_user(email: str, name: Optional[str]) -> UserProfile:
    normalized = _normalize_email(email)
    existing = USERS_BY_EMAIL.get(normalized)
    if existing:
        if name and not existing.name:
            existing.name = name
        return existing
    return _create_user(normalized, name=name, password=None)


def _create_session(user: UserProfile) -> SessionRecord:
    now = datetime.now(timezone.utc)
    session = SessionRecord(
        session_id=uuid.uuid4().hex,
        user_id=user.user_id,
        created_at=now,
        expires_at=now + timedelta(minutes=AUTH_SESSION_TTL_MINUTES),
    )
    SESSIONS[session.session_id] = session
    return session


def _encode_token(session: SessionRecord) -> str:
    payload = {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "exp": int(session.expires_at.timestamp()),
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    b64_payload = base64.urlsafe_b64encode(payload_bytes).decode("utf-8").rstrip("=")
    signature = hmac.new(
        AUTH_SECRET.encode("utf-8"),
        b64_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{b64_payload}.{signature}"


def _decode_token(token: str) -> Dict[str, object]:
    try:
        b64_payload, signature = token.split(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid auth token.") from exc
    expected = hmac.new(
        AUTH_SECRET.encode("utf-8"),
        b64_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid auth token.")
    padded = b64_payload + "=" * (-len(b64_payload) % 4)
    data = json.loads(base64.urlsafe_b64decode(padded))
    if not isinstance(data, dict):
        raise HTTPException(status_code=401, detail="Invalid auth token payload.")
    return data


def _current_user(request: Request) -> UserProfile:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token.")
    token = auth_header.split(" ", 1)[1].strip()
    payload = _decode_token(token)
    session_id = payload.get("session_id")
    if not isinstance(session_id, str):
        raise HTTPException(status_code=401, detail="Invalid auth token payload.")
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid auth session.")
    if session.expires_at < datetime.now(timezone.utc):
        SESSIONS.pop(session_id, None)
        raise HTTPException(status_code=401, detail="Auth session expired.")
    user = USERS_BY_ID.get(session.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid auth user.")
    return user


def _rate_limit(scope: str, limit: int, window_seconds: int):
    def _dependency(request: Request) -> None:
        host = request.client.host if request.client else "unknown"
        key = f"{scope}:{host}"
        RATE_LIMITER.check(key, limit=limit, window_seconds=window_seconds)

    return _dependency


@contextmanager
def _db_connection():
    connection = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )
    try:
        yield connection
    finally:
        connection.close()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _fetch_tag_map(
    cursor: pymysql.cursors.DictCursor,
    table: str,
    column: str,
    memory_ids: Sequence[int],
) -> Dict[int, List[str]]:
    if not memory_ids:
        return {}
    placeholders = ",".join(["%s"] * len(memory_ids))
    cursor.execute(
        f"SELECT memory_id, {column} FROM {table} WHERE memory_id IN ({placeholders})",
        list(memory_ids),
    )
    results: Dict[int, List[str]] = {}
    for row in cursor.fetchall():
        results.setdefault(row["memory_id"], []).append(row[column])
    return results


def _fetch_virtue_map(
    cursor: pymysql.cursors.DictCursor,
    memory_ids: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    if not memory_ids:
        return {}
    placeholders = ",".join(["%s"] * len(memory_ids))
    cursor.execute(
        f"""
        SELECT memory_id, virtue_name, virtue_score
        FROM memory_virtue_markers
        WHERE memory_id IN ({placeholders})
        """,
        list(memory_ids),
    )
    results: Dict[int, Dict[str, float]] = {}
    for row in cursor.fetchall():
        results.setdefault(row["memory_id"], {})[row["virtue_name"]] = float(
            row["virtue_score"]
        )
    return results


def _rows_to_memory_records(rows: Sequence[Dict[str, object]]) -> List[MemoryRecord]:
    if not rows:
        return []
    memory_ids = [row["id"] for row in rows]
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            tags_by_id = _fetch_tag_map(cursor, "memory_tags", "tag", memory_ids)
            sources_by_id = _fetch_tag_map(
                cursor, "memory_sources", "source_tag", memory_ids
            )
            virtues_by_id = _fetch_virtue_map(cursor, memory_ids)

    records: List[MemoryRecord] = []
    for row in rows:
        memory_id = row["memory_id"]
        record_id = row["id"]
        created_at = _as_utc(row["created_at"])
        updated_at = row.get("updated_at")
        updated_at_value = _as_utc(updated_at) if updated_at else None
        records.append(
            MemoryRecord(
                metadata=MemoryMetadata(
                    memory_id=memory_id,
                    created_at=created_at,
                    updated_at=updated_at_value,
                    tags=tags_by_id.get(record_id, []),
                    source_tags=sources_by_id.get(record_id, []),
                    virtue_markers=virtues_by_id.get(record_id, {}),
                    salience=float(row["salience"]),
                ),
                content=row["content"],
            )
        )
    return records


def _fetch_memory_rows(memory_id: Optional[str] = None) -> List[Dict[str, object]]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            if memory_id:
                cursor.execute(
                    """
                    SELECT id, memory_id, content, salience, created_at, updated_at
                    FROM memories
                    WHERE memory_id = %s
                    """,
                    (memory_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, memory_id, content, salience, created_at, updated_at
                    FROM memories
                    ORDER BY created_at ASC, id ASC
                    """
                )
            return cursor.fetchall()


def _fetch_memory_record(memory_id: str) -> Optional[MemoryRecord]:
    rows = _fetch_memory_rows(memory_id=memory_id)
    records = _rows_to_memory_records(rows)
    return records[0] if records else None


def _fetch_all_memories() -> List[MemoryRecord]:
    return _rows_to_memory_records(_fetch_memory_rows())


def _get_user_store(user_id: str) -> Dict[str, MemoryRecord]:
    return {record.metadata.memory_id: record for record in _fetch_all_memories()}


def _get_user_logs(user_id: str) -> List[LookupLogRecord]:
    return USER_LOOKUP_LOGS.setdefault(user_id, [])


def _safe_calculate(expression: str) -> float:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            if isinstance(node.op, ast.Mod):
                return left % right
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
        raise ValueError("Unsupported expression.")

    parsed = ast.parse(expression, mode="eval")
    if not all(isinstance(node, allowed_nodes) for node in ast.walk(parsed)):
        raise ValueError("Unsupported expression.")
    return _eval(parsed)


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _baseline_traits() -> PersonalityTraits:
    return PersonalityTraits(
        openness=0.5,
        conscientiousness=0.5,
        extraversion=0.5,
        agreeableness=0.5,
        neuroticism=0.5,
    )


def _fetch_latest_trait_snapshot() -> Optional[PersonalityTraitSnapshot]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT snapshot_at, openness, conscientiousness, extraversion,
                       agreeableness, neuroticism, signal_summary
                FROM personality_trait_history
                ORDER BY snapshot_at DESC, id DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
    if not row:
        return None
    signals = {}
    if row.get("signal_summary"):
        try:
            signals = json.loads(row["signal_summary"])
        except json.JSONDecodeError:
            signals = {}
    return PersonalityTraitSnapshot(
        snapshot_at=_as_utc(row["snapshot_at"]),
        traits=PersonalityTraits(
            openness=float(row["openness"]),
            conscientiousness=float(row["conscientiousness"]),
            extraversion=float(row["extraversion"]),
            agreeableness=float(row["agreeableness"]),
            neuroticism=float(row["neuroticism"]),
        ),
        signals=signals,
    )


def _fetch_trait_history(limit: int = TRAIT_HISTORY_LIMIT) -> List[PersonalityTraitSnapshot]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT snapshot_at, openness, conscientiousness, extraversion,
                       agreeableness, neuroticism, signal_summary
                FROM personality_trait_history
                ORDER BY snapshot_at DESC, id DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cursor.fetchall()
    snapshots: List[PersonalityTraitSnapshot] = []
    for row in reversed(rows):
        signals = {}
        if row.get("signal_summary"):
            try:
                signals = json.loads(row["signal_summary"])
            except json.JSONDecodeError:
                signals = {}
        snapshots.append(
            PersonalityTraitSnapshot(
                snapshot_at=_as_utc(row["snapshot_at"]),
                traits=PersonalityTraits(
                    openness=float(row["openness"]),
                    conscientiousness=float(row["conscientiousness"]),
                    extraversion=float(row["extraversion"]),
                    agreeableness=float(row["agreeableness"]),
                    neuroticism=float(row["neuroticism"]),
                ),
                signals=signals,
            )
        )
    return snapshots


def _current_personality_traits() -> PersonalityTraits:
    latest_snapshot = _fetch_latest_trait_snapshot()
    return latest_snapshot.traits if latest_snapshot else _baseline_traits()


def _store_trait_snapshot(
    traits: PersonalityTraits,
    snapshot_at: datetime,
    signals: Dict[str, float],
) -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO personality_trait_history (
                    snapshot_at, openness, conscientiousness, extraversion,
                    agreeableness, neuroticism, signal_summary
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    snapshot_at,
                    traits.openness,
                    traits.conscientiousness,
                    traits.extraversion,
                    traits.agreeableness,
                    traits.neuroticism,
                    json.dumps(signals),
                ),
            )
        connection.commit()


def _keyword_hits(text: str, keywords: Iterable[str]) -> int:
    if not text:
        return 0
    hits = 0
    for keyword in keywords:
        pattern = rf"\\b{re.escape(keyword)}\\b"
        hits += len(re.findall(pattern, text))
    return hits


def _memory_signal_weight(record: MemoryRecord) -> float:
    return max(0.3, record.metadata.salience)


def _collect_trait_signals(
    records: Iterable[MemoryRecord],
    since: Optional[datetime],
) -> Dict[str, float]:
    supportive_keywords = (
        "support",
        "help",
        "care",
        "kind",
        "empathy",
        "encourage",
        "listen",
        "gratitude",
        "thank",
    )
    conflict_keywords = (
        "conflict",
        "argue",
        "fight",
        "angry",
        "upset",
        "hostile",
        "criticize",
        "insult",
    )
    curiosity_keywords = (
        "curious",
        "explore",
        "discover",
        "learn",
        "new",
        "idea",
    )
    discipline_keywords = (
        "plan",
        "schedule",
        "routine",
        "discipline",
        "goal",
        "organize",
    )
    social_keywords = (
        "chat",
        "talk",
        "meet",
        "group",
        "share",
        "community",
        "sangat",
    )
    anxiety_keywords = (
        "anxious",
        "worry",
        "stress",
        "fear",
        "panic",
        "overwhelmed",
    )
    seva_keywords = (
        "seva",
        "langar",
        "kirat",
        "gurbani",
        "simran",
        "chardi",
        "hukam",
    )
    compassion_keywords = (
        "compassion",
        "mercy",
        "kindness",
        "forgive",
    )

    signals: Dict[str, float] = {
        "supportive": 0.0,
        "conflict": 0.0,
        "curiosity": 0.0,
        "discipline": 0.0,
        "social": 0.0,
        "anxiety": 0.0,
        "seva": 0.0,
        "compassion": 0.0,
    }

    for record in records:
        created_at = record.metadata.created_at
        if since and created_at <= since:
            continue
        tokens = [
            record.content,
            " ".join(record.metadata.tags),
            " ".join(record.metadata.source_tags),
            " ".join(record.metadata.virtue_markers.keys()),
        ]
        text = " ".join(tokens).lower()
        weight = _memory_signal_weight(record)
        signals["supportive"] += weight * _keyword_hits(text, supportive_keywords)
        signals["conflict"] += weight * _keyword_hits(text, conflict_keywords)
        signals["curiosity"] += weight * _keyword_hits(text, curiosity_keywords)
        signals["discipline"] += weight * _keyword_hits(text, discipline_keywords)
        signals["social"] += weight * _keyword_hits(text, social_keywords)
        signals["anxiety"] += weight * _keyword_hits(text, anxiety_keywords)
        signals["seva"] += weight * _keyword_hits(text, seva_keywords)
        signals["compassion"] += weight * _keyword_hits(text, compassion_keywords)
        for virtue_name, virtue_score in record.metadata.virtue_markers.items():
            virtue = virtue_name.lower()
            if "seva" in virtue or "service" in virtue:
                signals["seva"] += weight * float(virtue_score)
            if "compassion" in virtue or "daya" in virtue:
                signals["compassion"] += weight * float(virtue_score)

    return signals


def _evolve_traits(
    base: PersonalityTraits,
    signals: Dict[str, float],
) -> PersonalityTraits:
    agreeableness = base.agreeableness
    neuroticism = base.neuroticism
    openness = base.openness
    conscientiousness = base.conscientiousness
    extraversion = base.extraversion

    agreeableness += 0.012 * signals.get("supportive", 0.0)
    agreeableness += 0.02 * signals.get("seva", 0.0)
    agreeableness += 0.015 * signals.get("compassion", 0.0)
    agreeableness -= 0.01 * signals.get("conflict", 0.0)

    neuroticism += 0.015 * signals.get("conflict", 0.0)
    neuroticism += 0.01 * signals.get("anxiety", 0.0)
    neuroticism -= 0.006 * signals.get("supportive", 0.0)

    openness += 0.01 * signals.get("curiosity", 0.0)
    conscientiousness += 0.01 * signals.get("discipline", 0.0)
    conscientiousness += 0.008 * signals.get("seva", 0.0)
    extraversion += 0.01 * signals.get("social", 0.0)

    return PersonalityTraits(
        openness=_clamp(openness),
        conscientiousness=_clamp(conscientiousness),
        extraversion=_clamp(extraversion),
        agreeableness=_clamp(agreeableness),
        neuroticism=_clamp(neuroticism),
    )


def _run_trait_evolution_job() -> None:
    records = _fetch_all_memories()
    latest_snapshot = _fetch_latest_trait_snapshot()
    since = latest_snapshot.snapshot_at if latest_snapshot else None
    signals = _collect_trait_signals(records, since)
    if all(value == 0 for value in signals.values()):
        return
    base_traits = latest_snapshot.traits if latest_snapshot else _baseline_traits()
    updated_traits = _evolve_traits(base_traits, signals)
    snapshot_at = datetime.now(timezone.utc)
    _store_trait_snapshot(updated_traits, snapshot_at, signals)
    LOGGER.info(
        "trait_evolution snapshot=%s traits=(%.3f,%.3f,%.3f,%.3f,%.3f) signals=%s",
        snapshot_at.isoformat(),
        updated_traits.openness,
        updated_traits.conscientiousness,
        updated_traits.extraversion,
        updated_traits.agreeableness,
        updated_traits.neuroticism,
        signals,
    )


def _trait_evolution_loop() -> None:
    LOGGER.info("Starting trait evolution loop interval=%ss", TRAIT_ANALYSIS_INTERVAL_SECONDS)
    while True:
        try:
            _run_trait_evolution_job()
        except Exception as exc:
            LOGGER.warning("Trait evolution job failed: %s", exc)
        time.sleep(TRAIT_ANALYSIS_INTERVAL_SECONDS)


def _memory_stats() -> Tuple[int, int]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) AS record_count,
                       COALESCE(SUM(CHAR_LENGTH(content)), 0) AS content_bytes
                FROM memories
                """
            )
            row = cursor.fetchone()
    record_count = int(row["record_count"])
    content_bytes = int(row["content_bytes"] or 0)
    return record_count, content_bytes


def _log_memory_event(
    action: str,
    user_id: str,
    memory_id: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    record_count, content_bytes = _memory_stats()
    LOGGER.info(
        "memory_event action=%s user_id=%s memory_id=%s record_count=%s content_bytes=%s detail=%s",
        action,
        user_id,
        memory_id or "-",
        record_count,
        content_bytes,
        detail or "-",
    )


def _log_state_transition(
    user_id: str,
    event: str,
    current_state: EmotionState,
    new_state: EmotionState,
) -> None:
    LOGGER.info(
        "state_transition user_id=%s event=%s current=(%.3f,%.3f,%.3f,%s,%.3f) new=(%.3f,%.3f,%.3f,%s,%.3f)",
        user_id,
        event,
        current_state.valence,
        current_state.arousal,
        current_state.dominance,
        current_state.label,
        current_state.intensity,
        new_state.valence,
        new_state.arousal,
        new_state.dominance,
        new_state.label,
        new_state.intensity,
    )


def _format_reflection_content(detail: str) -> str:
    return (
        "Virtueism reflection: "
        f"{detail} "
        "In Sikh context, we keep seva, hukam, and chardi kala in view."
    )


def _emotion_state_context(state: EmotionState) -> Dict[str, object]:
    return {
        "valence": state.valence,
        "arousal": state.arousal,
        "dominance": state.dominance,
        "label": state.label,
        "intensity": state.intensity,
        "secondary_emotions": state.secondary_emotions,
        "updated_at": state.updated_at.isoformat() if state.updated_at else None,
    }


def _record_state_transition_reflection(
    user_id: str,
    event: str,
    current_state: EmotionState,
    new_state: EmotionState,
) -> None:
    detail = (
        "State transition noted after event "
        f"'{event}': {current_state.label} ➜ {new_state.label}."
    )
    _store_reflection_record(
        user_id,
        "state_transition",
        _format_reflection_content(detail),
        {
            "event": event,
            "previous_state": _emotion_state_context(current_state),
            "new_state": _emotion_state_context(new_state),
        },
    )


def _should_record_salience_reflection(
    previous_salience: Optional[float],
    new_salience: float,
) -> bool:
    if new_salience < REFLECTION_SALIENCE_THRESHOLD:
        return False
    if previous_salience is None:
        return True
    return previous_salience < REFLECTION_SALIENCE_THRESHOLD


def _record_salience_reflection(
    user_id: str,
    memory_id: str,
    salience: float,
    tags: Sequence[str],
) -> None:
    detail = (
        f"Memory {memory_id} reached salience {salience:.2f}, "
        "marking it as especially meaningful."
    )
    _store_reflection_record(
        user_id,
        "memory_salience_threshold",
        _format_reflection_content(detail),
        {
            "memory_id": memory_id,
            "salience": salience,
            "threshold": REFLECTION_SALIENCE_THRESHOLD,
            "tags": list(tags),
        },
    )


def _store_reflection_record(
    user_id: str,
    event_type: str,
    content: str,
    context: Optional[Dict[str, object]] = None,
) -> ReflectionRecord:
    reflection_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc)
    context_payload = context or {}
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO reflection_logs (
                    reflection_id, user_id, event_type, content, context, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    reflection_id,
                    user_id,
                    event_type,
                    content,
                    json.dumps(context_payload),
                    created_at,
                ),
            )
        connection.commit()
    return ReflectionRecord(
        reflection_id=reflection_id,
        created_at=created_at,
        event_type=event_type,
        content=content,
        context=context_payload,
    )


def _fetch_reflections(user_id: str) -> List[ReflectionRecord]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT reflection_id, created_at, event_type, content, context
                FROM reflection_logs
                WHERE user_id = %s
                ORDER BY created_at ASC, id ASC
                """,
                (user_id,),
            )
            rows = cursor.fetchall()
    reflections: List[ReflectionRecord] = []
    for row in rows:
        context_value: Dict[str, object] = {}
        if row.get("context"):
            try:
                context_value = json.loads(row["context"])
            except json.JSONDecodeError:
                context_value = {}
        reflections.append(
            ReflectionRecord(
                reflection_id=row["reflection_id"],
                created_at=_as_utc(row["created_at"]),
                event_type=row["event_type"],
                content=row["content"],
                context=context_value,
            )
        )
    return reflections


def _apply_imported_brain_state(
    emotion_state: Optional[EmotionState],
    emotion_baseline: Optional[EmotionState],
    emotion_transitions: Sequence[EmotionTransition],
    routine_phase: Optional[str],
    personality_traits: Optional[PersonalityTraits],
    trait_history: Sequence[PersonalityTraitSnapshot],
) -> None:
    global EMOTION_BASELINE
    global EMOTION_STATE
    global EMOTION_TRANSITIONS
    global CURRENT_ROUTINE_PHASE

    if emotion_baseline:
        EMOTION_BASELINE = emotion_baseline
    if emotion_state:
        EMOTION_STATE = emotion_state
    if emotion_transitions:
        EMOTION_TRANSITIONS = list(emotion_transitions)[-MAX_EMOTION_TRANSITIONS:]
    if routine_phase:
        CURRENT_ROUTINE_PHASE = routine_phase

    traits_snapshot = None
    if trait_history:
        traits_snapshot = trait_history[-1]
    traits = personality_traits or (traits_snapshot.traits if traits_snapshot else None)
    if traits:
        snapshot_at = (
            traits_snapshot.snapshot_at if traits_snapshot else datetime.now(timezone.utc)
        )
        signals = traits_snapshot.signals if traits_snapshot else {}
        _store_trait_snapshot(traits, snapshot_at, signals)


EMOTION_BASELINE = EmotionState(
    valence=0.5,
    arousal=0.5,
    dominance=0.5,
    label="neutral",
    intensity=0.0,
    secondary_emotions={"neutral": 0.0},
)
EMOTION_DECAY_HALF_LIFE_SECONDS = 300.0
EMOTION_BLEND_WEIGHT = 0.35
EMOTION_ROUTINE_BLEND_WEIGHT = 0.2
MAX_EMOTION_TRANSITIONS = 25
EMOTION_STATE = EmotionState(
    **EMOTION_BASELINE.dict(),
    updated_at=datetime.now(timezone.utc),
)
EMOTION_TRANSITIONS: List[EmotionTransition] = []
CURRENT_ROUTINE_PHASE = "unspecified"

EMOTION_EVENT_RULES = [
    {
        "keywords": ("gratitude", "thank", "appreciate"),
        "label": "proud",
        "valence": 0.25,
        "arousal": 0.1,
        "dominance": 0.15,
        "intensity": 0.6,
    },
    {
        "keywords": ("threat", "danger", "attack", "risk"),
        "label": "panicked",
        "valence": -0.35,
        "arousal": 0.45,
        "dominance": -0.25,
        "intensity": 0.85,
    },
    {
        "keywords": ("praise", "celebrate", "success", "win"),
        "label": "excited",
        "valence": 0.3,
        "arousal": 0.3,
        "dominance": 0.2,
        "intensity": 0.7,
    },
    {
        "keywords": ("loss", "failure", "sad", "grief"),
        "label": "downcast",
        "valence": -0.3,
        "arousal": -0.1,
        "dominance": -0.2,
        "intensity": 0.6,
    },
    {
        "keywords": ("conflict", "argument", "anger", "frustration"),
        "label": "frustrated",
        "valence": -0.2,
        "arousal": 0.25,
        "dominance": 0.05,
        "intensity": 0.55,
    },
    {
        "keywords": ("help", "support", "comfort"),
        "label": "relieved",
        "valence": 0.2,
        "arousal": -0.1,
        "dominance": 0.1,
        "intensity": 0.45,
    },
]


def _event_to_emotion_rule(event: str) -> Dict[str, object]:
    if not event:
        return {
            "label": "neutral",
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
            "intensity": 0.0,
        }
    lowered = event.lower()
    for rule in EMOTION_EVENT_RULES:
        if any(keyword in lowered for keyword in rule["keywords"]):
            return rule
    return {
        "label": "neutral",
        "valence": 0.0,
        "arousal": 0.0,
        "dominance": 0.0,
        "intensity": 0.1,
    }


def _apply_emotion_decay(state: EmotionState, now: datetime) -> EmotionState:
    if state.updated_at is None:
        return state
    elapsed = max(0.0, (now - state.updated_at).total_seconds())
    if elapsed <= 0.0:
        return state
    decay_factor = 0.5 ** (elapsed / EMOTION_DECAY_HALF_LIFE_SECONDS)
    decayed_valence = _clamp(
        EMOTION_BASELINE.valence + (state.valence - EMOTION_BASELINE.valence) * decay_factor
    )
    decayed_arousal = _clamp(
        EMOTION_BASELINE.arousal + (state.arousal - EMOTION_BASELINE.arousal) * decay_factor
    )
    decayed_dominance = _clamp(
        EMOTION_BASELINE.dominance
        + (state.dominance - EMOTION_BASELINE.dominance) * decay_factor
    )
    decayed_intensity = _clamp(state.intensity * decay_factor)
    decayed_label = state.label if decayed_intensity >= 0.15 else "neutral"
    decayed_secondary = {
        key: _clamp(value * decay_factor) for key, value in state.secondary_emotions.items()
    }
    if not decayed_secondary:
        decayed_secondary = {"neutral": decayed_intensity}
    return EmotionState(
        valence=decayed_valence,
        arousal=decayed_arousal,
        dominance=decayed_dominance,
        label=decayed_label,
        intensity=decayed_intensity,
        secondary_emotions=decayed_secondary,
        updated_at=state.updated_at,
    )


def _blend_emotion_states(
    base_state: EmotionState, target_state: EmotionState, blend_weight: float
) -> EmotionState:
    blend_weight = _clamp(blend_weight)
    valence = _clamp(
        (1.0 - blend_weight) * base_state.valence + blend_weight * target_state.valence
    )
    arousal = _clamp(
        (1.0 - blend_weight) * base_state.arousal + blend_weight * target_state.arousal
    )
    dominance = _clamp(
        (1.0 - blend_weight) * base_state.dominance + blend_weight * target_state.dominance
    )
    intensity = _clamp(
        (1.0 - blend_weight) * base_state.intensity
        + blend_weight * target_state.intensity
    )
    label = target_state.label if target_state.intensity >= 0.2 else base_state.label
    secondary = dict(base_state.secondary_emotions)
    for name, value in target_state.secondary_emotions.items():
        secondary[name] = _clamp(
            (1.0 - blend_weight) * secondary.get(name, 0.0) + blend_weight * value
        )
    return EmotionState(
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        label=label,
        intensity=intensity,
        secondary_emotions=secondary,
        updated_at=target_state.updated_at,
    )


def _routine_phase_and_baseline(now: datetime) -> Tuple[str, EmotionState]:
    local_time = now.astimezone()
    hour = local_time.hour
    if 5 <= hour < 12:
        phase = "morning"
        baseline = EmotionState(
            valence=0.6,
            arousal=0.65,
            dominance=0.55,
            label="lively",
            intensity=0.35,
            secondary_emotions={"lively": 0.35},
            updated_at=now,
        )
    elif 12 <= hour < 17:
        phase = "afternoon"
        baseline = EmotionState(
            valence=0.55,
            arousal=0.55,
            dominance=0.55,
            label="focused",
            intensity=0.25,
            secondary_emotions={"focused": 0.25},
            updated_at=now,
        )
    elif 17 <= hour < 21:
        phase = "evening"
        baseline = EmotionState(
            valence=0.52,
            arousal=0.45,
            dominance=0.5,
            label="mellow",
            intensity=0.22,
            secondary_emotions={"mellow": 0.22},
            updated_at=now,
        )
    else:
        phase = "night"
        baseline = EmotionState(
            valence=0.48,
            arousal=0.35,
            dominance=0.45,
            label="reflective",
            intensity=0.3,
            secondary_emotions={"reflective": 0.3},
            updated_at=now,
        )
    return phase, baseline



def _deterministic_event_signal(event: str) -> float:
    if not event:
        return 0.0
    signal = sum(ord(ch) for ch in event) % 100
    return signal / 100.0


def update_emotion_state(
    current_state: EmotionState, traits: PersonalityTraits, event: str
) -> EmotionState:
    global CURRENT_ROUTINE_PHASE
    now = datetime.now(timezone.utc)
    decayed_state = _apply_emotion_decay(current_state, now)
    routine_phase, routine_baseline = _routine_phase_and_baseline(now)
    CURRENT_ROUTINE_PHASE = routine_phase
    scheduled_state = _blend_emotion_states(
        decayed_state, routine_baseline, EMOTION_ROUTINE_BLEND_WEIGHT
    )
    signal = _deterministic_event_signal(event)
    rule = _event_to_emotion_rule(event)

    valence_shift = (
        rule["valence"]
        + (traits.extraversion - traits.neuroticism) * 0.15
        + (signal - 0.5) * 0.05
    )
    arousal_shift = rule["arousal"] + (signal - 0.5) * 0.2
    dominance_shift = rule["dominance"] + (traits.conscientiousness - 0.5) * 0.1

    target_state = EmotionState(
        valence=_clamp(scheduled_state.valence + valence_shift),
        arousal=_clamp(scheduled_state.arousal + arousal_shift),
        dominance=_clamp(scheduled_state.dominance + dominance_shift),
        label=str(rule["label"]),
        intensity=_clamp(float(rule["intensity"]) + abs(signal - 0.5) * 0.6),
        secondary_emotions={str(rule["label"]): _clamp(float(rule["intensity"]))},
        updated_at=now,
    )

    blended = _blend_emotion_states(scheduled_state, target_state, EMOTION_BLEND_WEIGHT)
    blended.updated_at = now
    return blended


def _generate_memory_id(connection: pymysql.connections.Connection) -> str:
    with connection.cursor() as cursor:
        cursor.execute("SELECT memory_id FROM memories")
        existing = {row["memory_id"] for row in cursor.fetchall()}
    index = len(existing) + 1
    while True:
        memory_id = f"mem-{index:04d}"
        if memory_id not in existing:
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
    records = _fetch_all_memories()
    if not records:
        VECTOR_INDEX.pop(VECTOR_INDEX_KEY, None)
        return
    documents = [_make_document(record) for record in records]
    ids = [record.metadata.memory_id for record in records]
    VECTOR_INDEX[VECTOR_INDEX_KEY] = FAISS.from_documents(documents, EMBEDDINGS, ids=ids)


def _index_record(record: MemoryRecord) -> None:
    vector_index = VECTOR_INDEX.get(VECTOR_INDEX_KEY)
    if vector_index is None:
        _rebuild_vector_index()
    else:
        document = _make_document(record)
        vector_index.add_documents([document], ids=[record.metadata.memory_id])


def reset_memory_store(user_id: str) -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM memories")
        connection.commit()
    _rebuild_vector_index()
    _log_memory_event("reset", user_id)


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
    return (
        f"{LOOKED_UP_MARKER} [{EXTERNAL_LOOKUP_SOURCE_TAG}] "
        f"I have looked this information up. {cleaned}"
    )


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


def _write_lookup_policy_audit(
    *,
    event: str,
    user_id: str,
    detail: str,
    source_tags: Optional[Sequence[str]] = None,
    content_preview: Optional[str] = None,
) -> None:
    payload = {
        "event": event,
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": detail,
        "source_tags": list(source_tags or []),
        "content_preview": content_preview,
    }
    with open(LOOKUP_AUDIT_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload) + "\n")


def _guard_external_source_tags(tags: Sequence[str], *, user_id: str, action: str) -> None:
    if EXTERNAL_LOOKUP_SOURCE_TAG in tags:
        _write_lookup_policy_audit(
            event="external_source_tag_blocked",
            user_id=user_id,
            detail=f"Blocked {action} with external source tag.",
            source_tags=tags,
        )
        raise HTTPException(
            status_code=400,
            detail="External knowledge must be stored via the /lookup endpoint.",
        )


def _guard_external_content(content: str, *, user_id: str, action: str) -> None:
    if LOOKED_UP_MARKER in content:
        _write_lookup_policy_audit(
            event="external_marker_blocked",
            user_id=user_id,
            detail=f"Blocked {action} with looked-up marker in content.",
            content_preview=content[:200],
        )
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
        "source_tag": EXTERNAL_LOOKUP_SOURCE_TAG,
        "looked_up_marker": LOOKED_UP_MARKER,
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
    user_id: str,
    payload: MemoryCreateRequest,
    memory_id: Optional[str] = None,
    created_at: Optional[datetime] = None,
    updated_at: Optional[datetime] = None,
) -> MemoryRecord:
    created_at_value = created_at or datetime.now(timezone.utc)
    updated_at_value = updated_at
    virtue_markers = _normalize_virtue_markers(payload.virtue_markers)
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            record_id = memory_id or _generate_memory_id(connection)
            cursor.execute(
                """
                INSERT INTO memories (memory_id, content, salience, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    record_id,
                    payload.content,
                    payload.salience,
                    created_at_value,
                    updated_at_value,
                ),
            )
            memory_row_id = cursor.lastrowid
            if payload.tags:
                cursor.executemany(
                    "INSERT INTO memory_tags (memory_id, tag) VALUES (%s, %s)",
                    [(memory_row_id, tag) for tag in payload.tags],
                )
            if payload.source_tags:
                cursor.executemany(
                    "INSERT INTO memory_sources (memory_id, source_tag) VALUES (%s, %s)",
                    [(memory_row_id, tag) for tag in payload.source_tags],
                )
            if virtue_markers:
                cursor.executemany(
                    """
                    INSERT INTO memory_virtue_markers (memory_id, virtue_name, virtue_score)
                    VALUES (%s, %s, %s)
                    """,
                    [
                        (memory_row_id, name, score)
                        for name, score in virtue_markers.items()
                    ],
                )
        connection.commit()
    record = MemoryRecord(
        metadata=MemoryMetadata(
            memory_id=record_id,
            created_at=_as_utc(created_at_value),
            tags=payload.tags,
            source_tags=payload.source_tags,
            virtue_markers=virtue_markers,
            salience=payload.salience,
            updated_at=_as_utc(updated_at_value) if updated_at_value else None,
        ),
        content=payload.content,
    )
    _index_record(record)
    _log_memory_event("create", user_id, memory_id=record_id)
    return record


def _existing_memory_ids(memory_ids: Iterable[str]) -> set[str]:
    ids = [memory_id for memory_id in memory_ids if memory_id]
    if not ids:
        return set()
    placeholders = ",".join(["%s"] * len(ids))
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT memory_id FROM memories WHERE memory_id IN ({placeholders})",
                ids,
            )
            return {row["memory_id"] for row in cursor.fetchall()}


def _update_memory_record(
    memory_id: str,
    payload: MemoryUpdateRequest,
) -> Optional[MemoryRecord]:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, memory_id, content, salience, created_at, updated_at
                FROM memories
                WHERE memory_id = %s
                """,
                (memory_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            updated_content = payload.content if payload.content is not None else row["content"]
            updated_salience = (
                payload.salience if payload.salience is not None else float(row["salience"])
            )
            updated_at = datetime.now(timezone.utc)
            cursor.execute(
                """
                UPDATE memories
                SET content = %s, salience = %s, updated_at = %s
                WHERE id = %s
                """,
                (updated_content, updated_salience, updated_at, row["id"]),
            )
            if payload.tags is not None:
                cursor.execute("DELETE FROM memory_tags WHERE memory_id = %s", (row["id"],))
                if payload.tags:
                    cursor.executemany(
                        "INSERT INTO memory_tags (memory_id, tag) VALUES (%s, %s)",
                        [(row["id"], tag) for tag in payload.tags],
                    )
            if payload.source_tags is not None:
                cursor.execute(
                    "DELETE FROM memory_sources WHERE memory_id = %s", (row["id"],)
                )
                if payload.source_tags:
                    cursor.executemany(
                        "INSERT INTO memory_sources (memory_id, source_tag) VALUES (%s, %s)",
                        [(row["id"], tag) for tag in payload.source_tags],
                    )
            if payload.virtue_markers is not None:
                normalized = _normalize_virtue_markers(payload.virtue_markers)
                cursor.execute(
                    "DELETE FROM memory_virtue_markers WHERE memory_id = %s", (row["id"],)
                )
                if normalized:
                    cursor.executemany(
                        """
                        INSERT INTO memory_virtue_markers
                            (memory_id, virtue_name, virtue_score)
                        VALUES (%s, %s, %s)
                        """,
                        [
                            (row["id"], name, score)
                            for name, score in normalized.items()
                        ],
                    )
        connection.commit()
    return _fetch_memory_record(memory_id)


def _delete_memory_record(memory_id: str) -> bool:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM memories WHERE memory_id = %s", (memory_id,))
            deleted = cursor.rowcount > 0
        connection.commit()
    return deleted


def _emotion_state_to_xml(parent: ET.Element, tag: str, state: EmotionState) -> ET.Element:
    emotion_el = ET.SubElement(parent, tag)
    ET.SubElement(emotion_el, "valence").text = str(state.valence)
    ET.SubElement(emotion_el, "arousal").text = str(state.arousal)
    ET.SubElement(emotion_el, "dominance").text = str(state.dominance)
    ET.SubElement(emotion_el, "label").text = state.label
    ET.SubElement(emotion_el, "intensity").text = str(state.intensity)
    if state.updated_at:
        ET.SubElement(emotion_el, "updated_at").text = state.updated_at.isoformat()
    secondary_el = ET.SubElement(emotion_el, "secondary_emotions")
    for name, value in state.secondary_emotions.items():
        ET.SubElement(secondary_el, "emotion", name=name, score=str(value))
    return emotion_el


def _personality_traits_to_xml(parent: ET.Element, traits: PersonalityTraits) -> None:
    traits_el = ET.SubElement(parent, "personality_traits")
    ET.SubElement(traits_el, "openness").text = str(traits.openness)
    ET.SubElement(traits_el, "conscientiousness").text = str(traits.conscientiousness)
    ET.SubElement(traits_el, "extraversion").text = str(traits.extraversion)
    ET.SubElement(traits_el, "agreeableness").text = str(traits.agreeableness)
    ET.SubElement(traits_el, "neuroticism").text = str(traits.neuroticism)


def _memory_module_to_xml(
    memories: List[MemoryRecord],
    emotion_state: EmotionState,
    emotion_transitions: Sequence[EmotionTransition],
    routine_phase: str,
    personality_traits: PersonalityTraits,
    emotion_baseline: EmotionState,
    reflections: Sequence[ReflectionRecord],
) -> str:
    root = ET.Element("memory_module")
    ET.SubElement(root, "routine_phase").text = routine_phase
    _emotion_state_to_xml(root, "emotion_state", emotion_state)
    _emotion_state_to_xml(root, "emotion_baseline", emotion_baseline)
    _personality_traits_to_xml(root, personality_traits)
    transitions_el = ET.SubElement(root, "emotion_transitions")
    for transition in emotion_transitions:
        transition_el = ET.SubElement(transitions_el, "transition")
        ET.SubElement(transition_el, "timestamp").text = transition.timestamp.isoformat()
        ET.SubElement(transition_el, "event").text = transition.event
        _emotion_state_to_xml(transition_el, "previous_state", transition.previous_state)
        _emotion_state_to_xml(transition_el, "new_state", transition.new_state)
    reflections_el = ET.SubElement(root, "reflections")
    for reflection in reflections:
        reflection_el = ET.SubElement(
            reflections_el, "reflection", id=reflection.reflection_id
        )
        ET.SubElement(reflection_el, "created_at").text = reflection.created_at.isoformat()
        ET.SubElement(reflection_el, "event_type").text = reflection.event_type
        ET.SubElement(reflection_el, "content").text = reflection.content
        ET.SubElement(reflection_el, "context").text = json.dumps(reflection.context)
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


def _parse_emotion_state_element(element: Optional[ET.Element]) -> Optional[EmotionState]:
    if element is None:
        return None
    valence_el = element.find("valence")
    arousal_el = element.find("arousal")
    dominance_el = element.find("dominance")
    label_el = element.find("label")
    intensity_el = element.find("intensity")
    if (
        valence_el is None
        or arousal_el is None
        or dominance_el is None
        or label_el is None
        or intensity_el is None
        or valence_el.text is None
        or arousal_el.text is None
        or dominance_el.text is None
        or label_el.text is None
        or intensity_el.text is None
    ):
        return None
    updated_el = element.find("updated_at")
    secondary: Dict[str, float] = {}
    for emotion_el in element.findall("./secondary_emotions/emotion"):
        name = emotion_el.attrib.get("name")
        score = emotion_el.attrib.get("score")
        if name and score:
            secondary[name] = float(score)
    if not secondary:
        secondary = {"neutral": float(intensity_el.text)}
    return EmotionState(
        valence=float(valence_el.text),
        arousal=float(arousal_el.text),
        dominance=float(dominance_el.text),
        label=label_el.text,
        intensity=float(intensity_el.text),
        secondary_emotions=secondary,
        updated_at=(
            datetime.fromisoformat(updated_el.text)
            if updated_el is not None and updated_el.text
            else None
        ),
    )


def _parse_personality_traits_element(
    element: Optional[ET.Element],
) -> Optional[PersonalityTraits]:
    if element is None:
        return None
    openness_el = element.find("openness")
    conscientiousness_el = element.find("conscientiousness")
    extraversion_el = element.find("extraversion")
    agreeableness_el = element.find("agreeableness")
    neuroticism_el = element.find("neuroticism")
    if (
        openness_el is None
        or conscientiousness_el is None
        or extraversion_el is None
        or agreeableness_el is None
        or neuroticism_el is None
        or openness_el.text is None
        or conscientiousness_el.text is None
        or extraversion_el.text is None
        or agreeableness_el.text is None
        or neuroticism_el.text is None
    ):
        return None
    return PersonalityTraits(
        openness=float(openness_el.text),
        conscientiousness=float(conscientiousness_el.text),
        extraversion=float(extraversion_el.text),
        agreeableness=float(agreeableness_el.text),
        neuroticism=float(neuroticism_el.text),
    )


def _parse_memory_module_xml(xml_payload: str) -> MemoryModuleImportData:
    root = ET.fromstring(xml_payload)
    records: List[MemoryImportRecord] = []
    for memory_el in root.findall("memory"):
        memory_id = memory_el.attrib.get("id")
        content_el = memory_el.find("content")
        salience_el = memory_el.find("salience")
        created_el = memory_el.find("created_at")
        updated_el = memory_el.find("updated_at")
        tags = [tag.text for tag in memory_el.findall("./tags/tag") if tag.text]
        source_tags = [
            source.text for source in memory_el.findall("./source_tags/source") if source.text
        ]
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
    emotion_state = _parse_emotion_state_element(root.find("emotion_state"))
    emotion_baseline = _parse_emotion_state_element(root.find("emotion_baseline"))
    personality_traits = _parse_personality_traits_element(root.find("personality_traits"))
    routine_phase_el = root.find("routine_phase")
    routine_phase = routine_phase_el.text if routine_phase_el is not None else None
    transitions: List[EmotionTransition] = []
    for transition_el in root.findall("./emotion_transitions/transition"):
        timestamp_el = transition_el.find("timestamp")
        event_el = transition_el.find("event")
        previous_state = _parse_emotion_state_element(transition_el.find("previous_state"))
        new_state = _parse_emotion_state_element(transition_el.find("new_state"))
        if (
            timestamp_el is None
            or event_el is None
            or timestamp_el.text is None
            or event_el.text is None
            or previous_state is None
            or new_state is None
        ):
            continue
        transitions.append(
            EmotionTransition(
                timestamp=datetime.fromisoformat(timestamp_el.text),
                event=event_el.text,
                previous_state=previous_state,
                new_state=new_state,
            )
        )
    return MemoryModuleImportData(
        memories=records,
        emotion_state=emotion_state,
        emotion_baseline=emotion_baseline,
        personality_traits=personality_traits,
        emotion_transitions=transitions,
        routine_phase=routine_phase,
    )


@app.on_event("startup")
def _rehydrate_vector_index() -> None:
    try:
        _rebuild_vector_index()
    except Exception as exc:
        LOGGER.warning("Failed to rehydrate vector index: %s", exc)
    if TRAIT_ANALYSIS_ENABLED:
        thread = threading.Thread(target=_trait_evolution_loop, daemon=True)
        thread.start()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/register", response_model=AuthResponse)
def auth_register(
    payload: AuthRegisterRequest,
    _: None = Depends(_rate_limit("auth", limit=5, window_seconds=60)),
) -> AuthResponse:
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    user = _create_user(payload.email, name=payload.name, password=payload.password)
    session = _create_session(user)
    token = _encode_token(session)
    return AuthResponse(token=token, user=user, expires_at=session.expires_at)


@app.post("/auth/login", response_model=AuthResponse)
def auth_login(
    payload: AuthLoginRequest,
    _: None = Depends(_rate_limit("auth", limit=5, window_seconds=60)),
) -> AuthResponse:
    normalized = _normalize_email(payload.email)
    user = USERS_BY_EMAIL.get(normalized)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    record = USER_PASSWORDS.get(user.user_id)
    if not record or not _verify_password(payload.password, record):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    session = _create_session(user)
    token = _encode_token(session)
    return AuthResponse(token=token, user=user, expires_at=session.expires_at)


@app.post("/auth/logout")
def auth_logout(
    user: UserProfile = Depends(_current_user),
) -> Dict[str, str]:
    session_ids = [sid for sid, session in SESSIONS.items() if session.user_id == user.user_id]
    for session_id in session_ids:
        SESSIONS.pop(session_id, None)
    return {"status": "logged_out"}


@app.get("/auth/me", response_model=AuthMeResponse)
def auth_me(user: UserProfile = Depends(_current_user)) -> AuthMeResponse:
    return AuthMeResponse(user=user)


@app.post("/auth/oauth/start", response_model=OAuthStartResponse)
def auth_oauth_start(
    payload: OAuthStartRequest,
    _: None = Depends(_rate_limit("auth", limit=5, window_seconds=60)),
) -> OAuthStartResponse:
    provider = payload.provider.lower()
    config = OAUTH_PROVIDERS.get(provider)
    if not config:
        raise HTTPException(status_code=400, detail="Unsupported OAuth provider.")
    if not config.get("client_id") or not config.get("client_secret"):
        raise HTTPException(status_code=400, detail="OAuth provider is not configured.")
    state = secrets.token_urlsafe(16)
    OAUTH_STATES[state] = {
        "provider": provider,
        "created_at": str(int(time.time())),
    }
    query = {
        "response_type": "code",
        "client_id": config["client_id"],
        "redirect_uri": OAUTH_REDIRECT_URI,
        "scope": config["scope"],
        "state": state,
    }
    auth_url = f"{config['auth_url']}?{urlencode(query)}"
    return OAuthStartResponse(provider=provider, auth_url=auth_url, state=state)


@app.get("/auth/oauth/callback")
async def auth_oauth_callback(
    provider: str,
    code: str,
    state: str,
    _: None = Depends(_rate_limit("auth", limit=10, window_seconds=60)),
) -> RedirectResponse:
    config = OAUTH_PROVIDERS.get(provider.lower())
    if not config:
        raise HTTPException(status_code=400, detail="Unsupported OAuth provider.")
    state_payload = OAUTH_STATES.pop(state, None)
    if not state_payload or state_payload.get("provider") != provider.lower():
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")
    created_at = int(state_payload.get("created_at", "0"))
    if int(time.time()) - created_at > AUTH_STATE_TTL_SECONDS:
        raise HTTPException(status_code=400, detail="OAuth state expired.")

    token_payload = {
        "client_id": config["client_id"],
        "client_secret": config["client_secret"],
        "code": code,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        token_headers = {"Accept": "application/json"}
        token_response = await client.post(config["token_url"], data=token_payload, headers=token_headers)
        if token_response.status_code >= 400:
            raise HTTPException(status_code=502, detail="OAuth token exchange failed.")
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=502, detail="OAuth access token missing.")

        if provider.lower() == "github":
            user_response = await client.get(
                config["userinfo_url"],
                headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
            )
            if user_response.status_code >= 400:
                raise HTTPException(status_code=502, detail="GitHub user lookup failed.")
            user_data = user_response.json()
            email = user_data.get("email")
            if not email:
                emails_response = await client.get(
                    config["emails_url"],
                    headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
                )
                if emails_response.status_code >= 400:
                    raise HTTPException(status_code=502, detail="GitHub email lookup failed.")
                emails = emails_response.json()
                primary = next(
                    (item for item in emails if item.get("primary") and item.get("verified")), None
                )
                email = (primary or (emails[0] if emails else {})).get("email")
            name = user_data.get("name") or user_data.get("login")
        else:
            user_response = await client.get(
                config["userinfo_url"],
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if user_response.status_code >= 400:
                raise HTTPException(status_code=502, detail="OAuth user lookup failed.")
            user_data = user_response.json()
            email = user_data.get("email")
            name = user_data.get("name") or user_data.get("given_name")

    if not email:
        raise HTTPException(status_code=400, detail="OAuth provider did not return an email.")

    user = _find_or_create_oauth_user(email, name)
    session = _create_session(user)
    token = _encode_token(session)

    redirect_url = urlparse(FRONTEND_OAUTH_REDIRECT)
    query_params = dict(parse_qsl(redirect_url.query))
    query_params.update({"token": token})
    new_query = urlencode(query_params)
    final_url = urlunparse(redirect_url._replace(query=new_query))
    return RedirectResponse(url=final_url)


@app.post("/state/update", response_model=StateUpdateResponse)
def state_update(
    payload: StateUpdateRequest,
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=30, window_seconds=60)),
) -> StateUpdateResponse:
    global EMOTION_STATE
    current_state = payload.current_state
    if current_state.updated_at is None and EMOTION_STATE.updated_at is not None:
        current_state = EMOTION_STATE
    new_state = update_emotion_state(current_state, payload.traits, payload.event)
    _log_state_transition(
        user.user_id,
        payload.event,
        current_state,
        new_state,
    )
    _record_state_transition_reflection(
        user.user_id,
        payload.event,
        current_state,
        new_state,
    )
    transition = EmotionTransition(
        timestamp=new_state.updated_at or datetime.now(timezone.utc),
        event=payload.event,
        previous_state=current_state,
        new_state=new_state,
    )
    EMOTION_STATE = new_state
    EMOTION_TRANSITIONS.append(transition)
    if len(EMOTION_TRANSITIONS) > MAX_EMOTION_TRANSITIONS:
        EMOTION_TRANSITIONS.pop(0)
    return StateUpdateResponse(
        new_state=new_state,
        explanation=(
            "Blended deterministic update with decay, time-of-day routine baseline, "
            "event rules, and trait shaping."
        ),
    )


@app.get("/state/snapshot", response_model=EmotionSnapshot)
def state_snapshot(
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=30, window_seconds=60)),
) -> EmotionSnapshot:
    _ = user
    return EmotionSnapshot(
        state=EMOTION_STATE,
        transitions=list(EMOTION_TRANSITIONS),
        routine_phase=CURRENT_ROUTINE_PHASE,
    )


@app.post("/response", response_model=LocalResponseResponse)
def response_local(
    payload: LocalResponseRequest,
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=30, window_seconds=60)),
) -> LocalResponseResponse:
    if payload.used_external_lookup or payload.external_facts:
        raise HTTPException(
            status_code=400,
            detail="External knowledge must be retrieved via the /lookup endpoint.",
        )
    return generate_local_response(payload)


@app.post("/lookup", response_model=LookupResponse)
async def lookup_external(
    payload: LookupRequest,
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=20, window_seconds=60)),
) -> LookupResponse:
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
    elif payload.model == DEFAULT_OPENROUTER_MODEL:
        request_body["max_tokens"] = OPENROUTER_MAX_COMPLETION_TOKENS

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
        memory_id = _create_memory_record(user.user_id, memory_payload).metadata.memory_id
    log_record = LookupLogRecord(
        lookup_id=lookup_id,
        created_at=datetime.now(timezone.utc),
        model=payload.model,
        query=payload.query,
        response=raw_response,
        response_prefixed=prefixed_response,
        memory_id=memory_id,
    )
    _get_user_logs(user.user_id).append(log_record)
    _write_lookup_audit_log(log_record)

    return LookupResponse(
        lookup_id=lookup_id,
        model=payload.model,
        response=prefixed_response,
        raw_response=raw_response,
        memory_id=memory_id,
    )


@app.get("/lookup/logs", response_model=LookupLogResponse)
def lookup_logs(
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=20, window_seconds=60)),
) -> LookupLogResponse:
    return LookupLogResponse(logs=_get_user_logs(user.user_id))


@app.post("/memory", response_model=MemoryRecord)
@app.post("/memory/add", response_model=MemoryRecord)
def memory_add(
    payload: MemoryCreateRequest,
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=30, window_seconds=60)),
) -> MemoryRecord:
    _guard_external_source_tags(
        payload.source_tags, user_id=user.user_id, action="memory create"
    )
    _guard_external_content(payload.content, user_id=user.user_id, action="memory create")
    record = _create_memory_record(user.user_id, payload)
    if _should_record_salience_reflection(None, record.metadata.salience):
        _record_salience_reflection(
            user.user_id,
            record.metadata.memory_id,
            record.metadata.salience,
            record.metadata.tags,
        )
    return record


@app.get("/memory", response_model=MemoryListResponse)
@app.get("/memory/list", response_model=MemoryListResponse)
def memory_list(
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=30, window_seconds=60)),
) -> MemoryListResponse:
    return MemoryListResponse(memories=_fetch_all_memories())


@app.get("/memory/{memory_id}", response_model=MemoryRecord)
def memory_get(
    memory_id: str,
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=30, window_seconds=60)),
) -> MemoryRecord:
    record = _fetch_memory_record(memory_id)
    if not record:
        raise HTTPException(status_code=404, detail="Memory not found")
    return record


@app.put("/memory/{memory_id}", response_model=MemoryRecord)
def memory_update(
    memory_id: str,
    payload: MemoryUpdateRequest,
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=30, window_seconds=60)),
) -> MemoryRecord:
    previous_record = _fetch_memory_record(memory_id)
    if payload.source_tags is not None:
        _guard_external_source_tags(
            payload.source_tags, user_id=user.user_id, action="memory update"
        )
    if payload.content is not None:
        _guard_external_content(
            payload.content, user_id=user.user_id, action="memory update"
        )
    updated_record = _update_memory_record(memory_id, payload)
    if not updated_record:
        raise HTTPException(status_code=404, detail="Memory not found")
    if previous_record and _should_record_salience_reflection(
        previous_record.metadata.salience, updated_record.metadata.salience
    ):
        _record_salience_reflection(
            user.user_id,
            updated_record.metadata.memory_id,
            updated_record.metadata.salience,
            updated_record.metadata.tags,
        )
    _rebuild_vector_index()
    _log_memory_event("update", user.user_id, memory_id=memory_id)
    return updated_record


@app.delete("/memory/{memory_id}")
def memory_delete(
    memory_id: str,
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=30, window_seconds=60)),
) -> Dict[str, str]:
    deleted = _delete_memory_record(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    _rebuild_vector_index()
    _log_memory_event("delete", user.user_id, memory_id=memory_id)
    return {"status": "deleted", "memory_id": memory_id}


@app.get("/memory/export/json", response_model=MemoryModule)
def memory_export_json(
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=10, window_seconds=60)),
) -> MemoryModule:
    return MemoryModule(
        memories=_fetch_all_memories(),
        emotion_state=EMOTION_STATE,
        emotion_baseline=EMOTION_BASELINE,
        personality_traits=_current_personality_traits(),
        emotion_transitions=list(EMOTION_TRANSITIONS),
        reflections=_fetch_reflections(user.user_id),
        routine_phase=CURRENT_ROUTINE_PHASE,
        trait_history=_fetch_trait_history(),
    )


@app.get("/memory/export/xml")
def memory_export_xml(
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=10, window_seconds=60)),
) -> Response:
    xml_payload = _memory_module_to_xml(
        _fetch_all_memories(),
        EMOTION_STATE,
        EMOTION_TRANSITIONS,
        CURRENT_ROUTINE_PHASE,
        _current_personality_traits(),
        EMOTION_BASELINE,
        _fetch_reflections(user.user_id),
    )
    return Response(content=xml_payload, media_type="application/xml")


@app.post("/memory/import/json", response_model=MemoryImportResponse)
def memory_import_json(
    payload: MemoryImportRequest,
    replace: bool = False,
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=10, window_seconds=60)),
) -> MemoryImportResponse:
    if replace:
        reset_memory_store(user.user_id)
    memory_ids: List[str] = []
    existing_ids = _existing_memory_ids(
        record.metadata.memory_id for record in payload.memories
    )
    for record in payload.memories:
        _guard_external_source_tags(
            record.metadata.source_tags, user_id=user.user_id, action="memory import"
        )
        _guard_external_content(
            record.content, user_id=user.user_id, action="memory import"
        )
        memory_id = record.metadata.memory_id
        if memory_id and memory_id in existing_ids:
            memory_id = None
        created_at = record.metadata.created_at
        updated_at = record.metadata.updated_at
        created_record = _create_memory_record(
            user.user_id,
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
    _apply_imported_brain_state(
        payload.emotion_state,
        payload.emotion_baseline,
        payload.emotion_transitions,
        payload.routine_phase,
        payload.personality_traits,
        payload.trait_history,
    )
    _log_memory_event(
        "import_json",
        user.user_id,
        detail=f"imported={len(memory_ids)} replace={replace}",
    )
    return MemoryImportResponse(imported=len(memory_ids), memory_ids=memory_ids)


@app.post("/memory/import/xml", response_model=MemoryImportResponse)
def memory_import_xml(
    xml_payload: str = Body(..., media_type="application/xml"),
    replace: bool = False,
    user: UserProfile = Depends(_current_user),
    _: None = Depends(_rate_limit("sensitive", limit=10, window_seconds=60)),
) -> MemoryImportResponse:
    if replace:
        reset_memory_store(user.user_id)
    module_data = _parse_memory_module_xml(xml_payload)
    memory_ids: List[str] = []
    existing_ids = _existing_memory_ids(record.memory_id for record in module_data.memories)
    for record in module_data.memories:
        _guard_external_source_tags(
            record.source_tags, user_id=user.user_id, action="memory import"
        )
        _guard_external_content(
            record.content, user_id=user.user_id, action="memory import"
        )
        memory_id = record.memory_id
        if memory_id and memory_id in existing_ids:
            memory_id = None
        created_record = _create_memory_record(
            user.user_id,
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
    _apply_imported_brain_state(
        module_data.emotion_state,
        module_data.emotion_baseline,
        module_data.emotion_transitions,
        module_data.routine_phase,
        module_data.personality_traits,
        [],
    )
    _log_memory_event(
        "import_xml",
        user.user_id,
        detail=f"imported={len(memory_ids)} replace={replace}",
    )
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
