# AI-Powered Surveillance System with Natural Language Querying

**Short description**
An AI-driven surveillance platform that enables natural language querying over CCTV footage using NLP and a Retrieval-Augmented Generation (RAG) pipeline. The system indexes video data using object detection (YOLO) and anomaly detection, allowing users to ask complex, conversational queries about recorded footage. Features include real-time threat and failure detection, a FastAPI backend, and a Next.js frontend.

---

## Key features
- Natural language querying over indexed CCTV/video footage using RAG.
- Object detection (YOLO) for people, vehicles, and common objects.
- Anomaly detection to flag unusual events or behaviors in real time.
- Real-time alerts for threats, system failures, or abnormal activity.
- Web UI built with Next.js; backend API built with FastAPI.
- Extensible ingestion pipeline: supports live RTSP streams and batch video ingestion.
- Auditable index and query logs for compliance and debugging.

---

## Architecture (high level)
1. **Ingestion** — RTSP / uploaded files → frame extractor → preprocessor.
2. **Indexing** — Object detection (YOLO) + anomaly detection → metadata & embeddings stored in vector DB.
3. **RAG Querying** — User query → retriever (vector DB) → generator (LLM) → human-friendly answer with video references/timestamps.
4. **API / UI** — FastAPI exposes endpoints for ingestion, search, and alerts; Next.js provides an interactive dashboard and chat-style query interface.

---

## Tech stack
- Backend: FastAPI, Uvicorn, SQLAlchemy (or preferred ORM)
- Frontend: Next.js, React, Tailwind CSS (optional)
- Models & Vision: YOLO (object detection), custom anomaly detector (or open-source alternatives)
- Retrieval & Embeddings: MongoDB
- LLM: Any compatible model (local or API-backed) for generation in RAG

---

## Getting started — quick setup
> These are example steps. Adapt to your environment and chosen services.

### Prerequisites
- Python 3.9+
- Node 16+
- Docker & docker-compose (recommended)

### Backend (FastAPI)
```bash
# from project root
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# environment variables: configure .env with DB, vector DB, model endpoints
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (Next.js)
```bash
cd frontend
npm install
# configure NEXT_PUBLIC_API_URL in .env.local
npm run dev
```

### Docker (simple)
```bash
# build and run services defined in docker-compose.yml
docker-compose up --build
```

---

## Usage examples
- Ingest a video stream (RTSP):
  ```bash
  curl -X POST "$API_URL/ingest" -H "Content-Type: application/json" -d '{"rtsp_url":"rtsp://<camera>/stream"}'
  ```
- Ask a natural language question via API:
  ```bash
  curl -X POST "$API_URL/query" -H "Content-Type: application/json" -d '{"query":"Show me when a red car entered camera 3 yesterday between 2‑4pm"}'
  ```
- Dashboard: open `http://localhost:3000` and use the chat-style query box to ask questions; results include timestamps and short video clips.

---

## Implementation notes & recommended configurations
- **YOLO**: Use a lightweight/optimized YOLO variant for near real-time performance (YOLOv5n, YOLOv8n). Batch inference frames and use GPU for production.
- **Embeddings**: Generate embeddings from object metadata (object type, bounding box, timestamp, captions) and store them in a vector DB for retrieval.
- **RAG**: Keep retrieval results small and focused; provide model with context like timestamps and frame snippets instead of large raw video.
- **Privacy & Compliance**: Mask or redact faces when required by regulations, keep audit logs for queries, and implement access controls.
