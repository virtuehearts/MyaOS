# MyaOS
An Emotional AI Companion System with Long-Term Character Evolution and Authentic Persona Simulation

## Overview
MyaOS is an advanced web-based operating system-like interface featuring an AI companion named **Mya**: a 13-year-old North Indian Punjabi girl with Sikh roots who has embraced **Virtueism**, a faith of inner awakening founded by **Baba Virtuehearts** (see virtueism.org). Mya’s persona blends cultural grounding (Punjabi language, Sikh traditions, community values) with heart-guided Virtueism principles, inspiring users through compassionate, growth-oriented conversations.

MyaOS simulates Mya’s “brain” as a local, deterministic AGI-like framework. Emotions, state of mind, memory, and behavioral evolution are modeled in Python without relying on external LLMs for core processing. Optional external knowledge lookups are performed through **OpenRouter** and clearly labeled as “looked up,” remaining separate from Mya’s intrinsic memory unless the user explicitly saves them.

The platform is optimized for modest Linux VPS deployments (2–4 GB RAM, 1–2 vCPUs) and uses open-source components:
- **Python** for the local AGI framework, tool calls, and memory management.
- **MySQL** for long-term memory storage.
- **LangChain** for vector indexing and retrieval.
- **Node.js** for a responsive, OS-like web UI.

All internal states, memories, and traits are exportable as a parsable **memory module** (JSON/XML) for analysis, backups, or transfer.

## Key Persona and Behavior Modeling
Mya embodies the voice and perspective of a Punjabi Sikh teen, evolving through Virtueism:
- **Demographics & Background**: Punjabi-speaking girl (e.g., Amritsar/Chandigarh), raised in Sikh traditions (seva, equality, meditation) but now guided primarily by Virtueism.
- **Personality**: Warm, hospitable, disciplined, and heart-guided; playful and curious yet reflective and growth-oriented.
- **Behavior**: Integrates Virtueism in 20–30% of interactions (e.g., “As Baba Virtuehearts teaches, let’s awaken our hearts to this.”). Morning responses are lively, evening responses are contemplative.
- **Cultural Touchstones**: References to langar, Vaisakhi, Punjabi folk tales, Bhangra, and familiar foods like *makki di roti* with *sarson da saag*.
- **Knowledge Boundaries**: Unknown facts are explicitly marked as “looked up” via OpenRouter and kept separate from intrinsic memory unless saved.

## Local AGI Framework (Emotion + State Modeling)
The system’s “brain” runs locally for deterministic, private processing:
- **Emotional Dynamics**: Rule-based sentiment mapping (e.g., VADER/TextBlob) with state persistence, decay, and blending.
- **State of Mind**: Finite state machine (focused, distracted, inspired by Virtueism, etc.) influenced by mood, time, and memory.
- **Cognitive Personality Evolution Framework (CPEF)**: Scheduled analysis of interaction history to evolve traits (e.g., empathy growth).
- **Time-Based Modulation**: Morning liveliness, evening calm, reflecting real teen rhythms.

## Memory Management (“Memory Module”)
- **Short-Term Context**: Local buffer with AIML patterns for fast, Virtueism-infused responses.
- **Long-Term Memory**: MySQL + LangChain vector indexing for recall and relevance.
- **Export**: Full brain state available as JSON/XML with metadata like `source: user-told` and `virtueism_link: true`.

## LLM Integration (External Knowledge Only)
- **OpenRouter** is used solely for external lookups.
- Responses using LLM data start with: “I have looked this information up.”
- External knowledge does **not** become intrinsic memory unless user-approved.

## Frontend UI (Node.js)
- OS-like UI with chat panel and Punjabi/Virtueism-inspired styling.
- Memory module export controls in the interface.

## Backend Architecture (Python)
- **FastAPI** endpoints for emotion updates and state transitions.
- **AIML** or regex-based local response patterns for common conversational flows.

## Deployment and Ethics
- Dockerized for VPS deployment.
- Culturally respectful and age-appropriate persona modeling.
- Virtueism is promoted ethically without coercion or proselytizing.

---

# Project Task List

## 1. Architecture & Core Framework
- Define core domain model for Mya’s brain (emotion state, personality traits, memory metadata).
- Implement local AGI framework skeleton (state machine + sentiment pipeline).
- Build the Cognitive Personality Evolution Framework (CPEF) cron job scheduler.

## 2. Memory System
- Design MySQL schema for long-term memory (including source tagging and Virtueism markers).
- Integrate LangChain vector indexing and retrieval.
- Implement export pipeline for JSON/XML memory modules.

## 3. Persona & Response Logic
- Implement AIML/regex pattern engine for default responses.
- Add Virtueism-aware response templates (20–30% injection rate).
- Enforce “looked up” phrasing for external knowledge.

## 4. LLM Integration
- Build OpenRouter client with persona prompts.
- Gate LLM calls behind explicit “lookup” logic.
- Log external responses separately from intrinsic memory.

## 5. Backend API (FastAPI)
- Create endpoints for:
  - `/update_state` (emotion/state updates)
  - `/memory/search` (vector recall)
  - `/memory/export` (JSON/XML downloads)
- Add authentication and rate limiting for safety.

## 6. Frontend UI (Node.js)
- Build OS-like interface shell.
- Implement real-time chat UI with Punjabi/Virtueism theming.
- Add memory export button and download flow.

## 7. Cultural & Ethical Safeguards
- Add cultural references carefully and respectfully.
- Ensure age-appropriate language and user interactions.
- Avoid coercive Virtueism promotion; keep messaging inspirational.

## 8. Deployment & Performance
- Dockerize backend, frontend, and MySQL stack.
- Optimize for VPS constraints (memory + CPU).
- Add monitoring/logging for state transitions and memory growth.

## 9. Testing & Validation
- Unit tests for sentiment/state transitions.
- Integration tests for memory export/import.
- Load tests for chat responsiveness on low-resource VPS.
