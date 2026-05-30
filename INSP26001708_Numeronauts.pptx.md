# AI-Powered Surveillance & Crowd Management with Natural Language Querying

### 9 AI Models. 1 Unified Pipeline. From Question to Video Evidence in 107ms.

## Team: **Numeronauts**

| Member | Role |
|---|---|
| **Sujal M H** | Project Leader & AI Engineer |
| **Sujnan Kumar** | Backend Engineer |
| **Yashas Shetty** | Research Lead & Alerts API Developer |
| **Suhan D Shet** | Systems Integration Engineer |
| **Diya Shetty** | UX Developer |
| **Ms. Amrutha** | Project Guide |

**sparkle.kpit.com**

---

# SLIDE 1 - THE PROBLEM

## Surveillance Today Is Broken

Modern facilities deploy **hundreds of cameras** generating **500+ hours of footage per day** - yet less than **5%** is ever reviewed. When an incident occurs, operators face three critical failures:

### 1. Manual Monitoring Doesn't Scale
A single operator watching 16 camera feeds loses effective attention after **20 minutes** (Security Industry Association, 2024). Critical events are missed in real-time.

### 2. Search Requires Knowing *When* - But You're Searching Because You Don't Know
Traditional CCTV offers only **timestamp-based scrubbing**. Finding *"the person in a red jacket near Gate 3"* means manually reviewing hours of footage across multiple cameras.

### 3. Zero Semantic Understanding
Existing systems store raw video with no comprehension of **what** happened, **who** was involved, or **why** it matters. There is no way to ask a question and get an answer.

> ### The Core Question:
> *What if security teams could simply ask - in plain English - "Show me everyone who entered the restricted zone with a bag after 10 PM" and get the exact video clips in under a second?*

---

# SLIDE 2 - THE INNOVATION

## A Retrieval-Augmented Generation (RAG) Pipeline for Video Surveillance

Our system transforms passive CCTV streams into a **searchable, queryable, proactive intelligence asset** by orchestrating **9 deep learning models** in a unified real-time pipeline across three stages:

---

### 1. SEE - Real-Time Multimodal Perception

Every frame is analyzed by multiple models simultaneously:

| Capability | Model / Technique | Output |
|---|---|---|
| Object Detection + Segmentation | **YOLOv11m-seg** (instance segmentation) | Bounding boxes, class labels, pixel masks |
| Multi-Object Tracking | **BoT-SORT + OSNet** (appearance-based) | Persistent track IDs across frames |
| Person Attribute Recognition | **OpenVINO** (crossroad-0238 model) | 7 attributes: gender, bag, hat, hair length, pants, sleeves, jacket |
| Perceptual Color Extraction | **CIEDE2000** in LAB color space | 25 named clothing colors (perceptually accurate, not naive RGB) |

> *Result: Rich structured metadata per frame - who is there, what they look like, what they're wearing, and where they are.*

---

### 2. UNDERSTAND - Semantic Comprehension

Raw detections are transformed into **meaning** through vision-language models and embedding fusion:

| Capability | Model | Dimension |
|---|---|---|
| Scene Captioning | **Qwen2-VL-2B-Instruct** (with ViT-GPT2 + CLIP zero-shot fallback) | Natural language scene description |
| Scene Embeddings | **OpenCLIP ViT-B-32** (laion2b_s34b_b79k) | 512-dim vector per frame |
| Person Visual Encoding | **SigLIP** (google/siglip-base-patch16-224) | 768-dim vector per person crop |
| Person Text Encoding | **all-MiniLM-L12-v2** (Sentence Transformers) | 384-dim vector from attribute text |
| **Multimodal Person Fusion** | **0.6 x SigLIP + 0.4 x MiniLM -> L2-normalized** | **1152-dim fused person vector** |

> *Result: Every person and every scene is represented as a high-dimensional vector that captures both visual appearance and semantic meaning.*

---

### 3. INTERACT - Natural Language Intelligence

Users interact through conversation, not timestamps:

* **Natural Language Parsing**: LLM-powered (GPT-4o-mini / Ollama) intent classification with 16-field structured output + 250-line regex fallback parser
* **Hybrid Search**: Simultaneous MongoDB structured query + dual FAISS vector search (scene-level + person-level)
* **Adaptive Retrieval**: 8 tunable accuracy features - query expansion, recency boost, MMR diversity re-ranking, adaptive confidence thresholds
* **LLM Answer Generation**: Context-aware natural language response with ranked video clips
* **Conversational Alert Creation**: Users can say *"Alert me when more than 5 people gather in Zone A after 10 PM"* -> system auto-creates a fully structured monitoring rule

> *Result: From natural language question to video evidence in **107ms average retrieval latency**.*

---

# SLIDE 3 - SYSTEM ARCHITECTURE

<!-- Replace with architectural flow diagram -->
> **Figure 1: Architectural Flow Diagram**

### Five-Layer Architecture

| Layer | Components |
|---|---|
| **Input** | Live RTSP/MJPEG streams, Video file upload, Threaded frame capture (full pipeline parity: uploads processed identically to live) |
| **Vision (9 Models)** | YOLOv11m-seg -> BoT-SORT+OSNet -> OpenVINO Batch Attributes -> CIEDE2000 Color -> Qwen2-VL Captions -> OpenCLIP + SigLIP+MiniLM |
| **Structured Storage** | MongoDB (13 collections, compound-indexed, time-series optimized) |
| **Vector Storage** | FAISS: 512-dim scene index + 1152-dim person index |
| **Intelligence** | NL Parser (LLM) -> Hybrid Search -> Result Merger -> Answer Gen, Alert Engine -> Anomaly Detector (Z-score) -> SSE Streaming |
| **Interface** | Next.js 14 Dashboard, Conversation UI, Alerts, Analytics - Live MJPEG streams, Dark/Light mode, SSE real-time updates |

### Complete Model Inventory (9 AI/ML Models)

| # | Model | Source | Purpose |
|---|---|---|---|
| 1 | YOLOv11m-seg | Ultralytics | Detection + instance segmentation |
| 2 | BoT-SORT | boxmot | Multi-object tracking |
| 3 | OSNet (x1_0, MSMT17) | boxmot | Appearance-based re-identification |
| 4 | person-attributes-crossroad-0238 | Intel OpenVINO Model Zoo | 7-attribute person recognition |
| 5 | OpenCLIP ViT-B-32 | LAION (laion2b_s34b_b79k) | Scene-level 512-dim embeddings |
| 6 | SigLIP-base-patch16-224 | Google | Person visual 768-dim embeddings |
| 7 | all-MiniLM-L12-v2 | Sentence Transformers | Person text 384-dim embeddings |
| 8 | Qwen2-VL-2B-Instruct | Alibaba | Vision-language frame captioning |
| 9 | GPT-4o-mini / Ollama | OpenAI / Local | NL parsing + answer generation |

### Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React, TypeScript, Tailwind CSS, Recharts |
| Backend | Python 3.10+, FastAPI, WebSockets, SSE |
| Computer Vision | PyTorch, OpenCV, Ultralytics, OpenVINO, ONNX Runtime |
| Vision-Language | Qwen2-VL, OpenCLIP, SigLIP, Sentence Transformers |
| Database | MongoDB (13 collections, compound indexes, time-series optimized) |
| Vector Search | FAISS (dual index: 512-dim scenes + 1152-dim persons) |
| Video | FFmpeg (H.264, CRF 23, faststart), 5-min chunked continuous recording |
| Deployment | Docker, NVIDIA CUDA, GPU acceleration |

---

# SLIDE 4 - KEY CAPABILITIES

<!-- Replace with Dashboard + Conversation UI screenshots -->
> **Figure 2: Dashboard** - Real-time KPIs, live camera grid with zone overlays, recent alerts
>
> **Figure 3: Conversation Page** - Natural language querying with progressive processing steps, video clip results

---

### Capability 1: Natural Language Video Retrieval

A user types:
> *"Show me people with backpacks near Exit B in the last 30 minutes"*

**What happens under the hood:**
1. LLM parses intent -> extracts: object=person, attribute=backpack, location=Exit B, time=last 30min
2. MongoDB query filters by camera zone, time range, and object class
3. FAISS searches both scene (512-dim) and person (1152-dim) indices by text embedding similarity
4. Results merged with adaptive gap computation, recency boost, and MMR diversity re-ranking
5. Video clips stitched from snapshots or recording slices
6. LLM generates a natural language summary with the ranked clips

**Response time: ~107ms** for 500,000 indexed frames.

---

### Capability 2: Conversational Alert Creation

A user says:
> *"Alert me when someone enters the restricted zone with a bag after 10 PM"*

**The system automatically creates:**
- Event type: zone_occupancy
- Object filter: person with attribute has_bag = true
- Zone target: restricted zone (by name match)
- Time-of-day window: 22:00-06:00 (supports cross-midnight)
- Severity: high
- Cooldown: 60s (configurable)
- Actions: store_clip, push_ws (real-time WebSocket notification)

No manual rule configuration - just describe what you want in plain English.

---

### Capability 3: Proactive Alerting Engine

Real-time, per-frame evaluation with **6 alert types**:

| Alert Type | How It Works |
|---|---|
| **Crowd Density** | Person count >= configurable threshold per camera/zone |
| **Loitering Detection** | Track history duration >= threshold (default 60s) |
| **Fight Detection** | Motion energy heuristic: >=2 persons within 24px proximity + avg motion energy >= 0.03 over 10-frame window |
| **Zone Occupancy** | Per-zone person counting via normalized bbox centroids against ROI polygons |
| **Object Count Rules** | Configurable comparisons: ==, >=, <=, >, < for any object class |
| **Time-of-Day Windows** | Alerts active only during specified hours (supports cross-midnight, e.g., 22:00-06:00) |

Infrastructure: Rule cache (5s TTL), per-rule cooldown tracking, **SSE streaming** for instant push to frontend.

---

### Capability 4: Statistical Anomaly Detection

**No predefined rules needed** - the system discovers anomalies automatically:

- **Method**: Z-score analysis against a **7-day rolling hourly baseline** (mean + standard deviation)
- **Three anomaly types**:
  - crowd_spike - Person count significantly above the hourly historical average (Z >= 3.0)
  - off_hours - Detections during historically zero-activity hours (Z >= 2.0)
  - unusual_object - Object types rarely seen on a specific camera (Z >= 1.5)
- Results cached in MongoDB (10-min TTL) and exposed on the dashboard

---

# SLIDE 5 - PERFORMANCE & METRICS

### Measured Performance

| Metric | Value |
|---|---|
| Detection Precision (Person class) | **96.5%** |
| Gender Recognition Accuracy | **91%** |
| Retrieval Latency (500K indexed frames) | **107ms** |
| NL Parsing Success Rate | **94%** |
| AI/ML Models Orchestrated | **9** |
| MongoDB Collections (indexed) | **13** |
| API Endpoints | **30+** |
| Person Attributes (real-time) | **7 per person** |
| Named Colors Recognized | **25** (CIEDE2000 perceptual) |
| Recording Format | **H.264, 5-min chunks, continuous** |
| Alert Streaming | **SSE, sub-second delivery** |

### Impact Comparison

| Scenario | Traditional CCTV | Our System |
|---|---|---|
| Finding a specific person | **20-60 min** manual scrubbing | **107ms** - ask in English, get clips |
| Detecting a fight | Missed unless operator is watching | **Real-time** automated detection + alert |
| Creating a monitoring rule | IT team configures software | **Say it in chat** - auto-created |
| Discovering unusual patterns | Requires human analysis | **Automatic** Z-score anomaly detection |
| Color-based person search | Not possible | **25 colors**, CIEDE2000 perceptual matching |

---

# SLIDE 6 - COMPETITIVE DIFFERENTIATION

### Why This System Doesn't Exist Yet

| Capability | Traditional CCTV | Commercial AI NVRs (Verkada, BriefCam) | **Our System** |
|---|---|---|---|
| Object Detection | No | Yes (basic) | **YOLOv11m-seg** (instance segmentation) |
| Person Attributes | No | Limited | **7 attributes** + 25-color CIEDE2000 |
| Natural Language Querying | No | No | **Full conversational RAG pipeline** |
| Semantic Video Search | No | No | **Dual FAISS index** (scene + person) |
| Vision-Language Captioning | No | No | **Qwen2-VL** with tiered fallback |
| Chat-Based Alert Creation | No | No | **NL -> structured rule, automatic** |
| Statistical Anomaly Detection | No | Basic | **Z-score, 7-day rolling baseline** |
| Multimodal Person Re-ID | No | Visual only | **1152-dim SigLIP x MiniLM fusion** |
| Open Architecture | No (proprietary) | No (proprietary, expensive) | **Open stack, self-hosted** |

### Unique Technical Innovations

1. **Multimodal 1152-dim Person Re-ID** - Novel weighted fusion of SigLIP visual embeddings (768-dim) + MiniLM textual attribute embeddings (384-dim), L2-normalized. Captures both *how a person looks* and *what they're wearing* in a single searchable vector.

2. **CIEDE2000 Perceptual Color Matching** - Industry-standard Delta E 2000 color distance in LAB color space. Identifies 25 named clothing colors with human-like perceptual accuracy, not naive RGB comparison.

3. **Tiered VLM Fallback Architecture** - Qwen2-VL-2B (primary) -> ViT-GPT2 (fallback) -> CLIP zero-shot classification against 21 surveillance-oriented labels (final fallback). Production-grade graceful degradation.

4. **Adaptive Retrieval Engine** - 8 configurable accuracy features working in concert: query expansion with synonym dictionaries, CLIP embedding fusion (0.6/0.4 weighted), adaptive confidence thresholds (25th percentile floor), recency boost (24h half-life decay), MMR diversity re-ranking (lambda=0.3), adaptive gap computation, temporal overlap detection, and clip expansion.

---

# SLIDE 7 - FRONTEND EXPERIENCE

### 5 Purpose-Built Pages

| Page | Key Features |
|---|---|
| **Dashboard** (/) | 4 real-time KPI summary cards (cameras, alerts, anomalies, detections), live MJPEG camera grid with zone overlays, recent alerts feed, auto-refresh every 10s |
| **Conversation** (/conversation) | Two modes: *retrieve* (find footage) and *alert* (create rules). LLM intent classification, progressive processing steps UI showing each pipeline stage, session management, inline video clip playback |
| **Alerts** (/alerts) | SSE real-time toast notifications, sortable/filterable table (TanStack Table) with full CRUD, rule creation dialog with dropdowns for objects, severity, conditions, zones, and time windows |
| **Analytics** (/analytics) | Recharts-powered: activity over time (line), color distribution (pie), camera uptime (bar), media statistics |
| **Settings** (/settings) | Theme toggle (light/dark/system), LLM provider configuration (OpenAI / OpenRouter / Ollama), indexing mode selection, system reset |

**Design**: DM Sans + DM Mono typography, island-based layout, CSS custom properties, fully responsive, dark mode via next-themes.

> **Figure 4: Alerts Management Page** - Rule creation + real-time SSE alert stream
>
> **Figure 5: Analytics Charts** - Activity trends, color distribution, camera health

---

# SLIDE 8 - FUTURE ROADMAP

### Near-Term (3-6 Months)

| Initiative | Target |
|---|---|
| **Cross-Camera Person Re-ID** | Unified global identity graph across N cameras - persistent tracking as a person moves between feeds. Target: **95% re-ID accuracy** |
| **Edge Deployment** | Full pipeline optimization via ONNX + OpenVINO quantization for Jetson Nano / Raspberry Pi. Target: **40% latency reduction** |
| **Automated Incident Reports** | LLM-powered PDF report generation from alert logs - timeline, clips, attribute summaries, exported for legal/compliance use |

### Medium-Term (6-12 Months)

| Initiative | Target |
|---|---|
| **Crowd Flow Prediction** | Temporal density modeling with heatmaps and congestion forecasting for proactive crowd management |
| **Multi-Language NL Support** | Extend NL parser to Kannada, Hindi, and other regional languages for broader accessibility |
| **Federated Learning** | On-premise model fine-tuning without sending video data to the cloud - privacy-first architecture |

---

# Thank You

### Team Numeronauts - KPIT Sparkle 2026

> *"9 AI models, 1 pipeline - from question to evidence in 107ms."*

**sparkle.kpit.com**

---
