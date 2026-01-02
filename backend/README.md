# MyaOS Backend (FastAPI)

## Overview
This backend provides a deterministic, placeholder emotional state pipeline and simple memory APIs.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## Run
```bash
uvicorn backend.main:app --reload
```

## Endpoints
### Health
- `GET /health`

### State Updates
- `POST /state/update`

Request body:
```json
{
  "current_state": { "valence": 0.5, "arousal": 0.5, "dominance": 0.5 },
  "traits": {
    "openness": 0.6,
    "conscientiousness": 0.7,
    "extraversion": 0.5,
    "agreeableness": 0.4,
    "neuroticism": 0.2
  },
  "event": "User greeted the system."
}
```

Response:
```json
{
  "new_state": { "valence": 0.53, "arousal": 0.51, "dominance": 0.52 },
  "explanation": "Deterministic placeholder update based on traits and event signal."
}
```

### Memory Operations
- `POST /memory` or `POST /memory/add`
- `GET /memory` or `GET /memory/list`
- `GET /memory/{memory_id}`
- `PUT /memory/{memory_id}`
- `DELETE /memory/{memory_id}`
- `GET /memory/export/json`
- `GET /memory/export/xml`
- `POST /memory/import/json`
- `POST /memory/import/xml`

Request body:
```json
{
  "content": "User likes calming music.",
  "tags": ["preference", "music"],
  "source_tags": ["chat", "user-17"],
  "virtue_markers": { "temperance": 0.7 },
  "salience": 0.8
}
```

## Notes
- Memory storage is in-memory and resets on restart.
- The emotion update logic is deterministic and intended as a placeholder.
