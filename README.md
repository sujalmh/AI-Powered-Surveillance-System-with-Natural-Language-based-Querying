# AI-Powered Surveillance System with Natural Language-based Querying

![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge&logo=activity)
![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge&logo=git)
![Platform](https://img.shields.io/badge/Platform-Linux%20|%20macOS-lightgrey?style=for-the-badge&logo=linux)

## Overview

**AI-Powered-Surveillance-System-with-Natural-Language-based-Querying** is a state-of-the-art video analytics and monitoring platform designed to provide real-time insights and automated surveillance capabilities. Leveraging advanced computer vision models and Large Language Models (LLMs), this system transforms traditional video feeds into actionable intelligence. It offers object detection, multi-object tracking, semantic search, and an interactive conversational interface for querying video data naturally.

---

## Key Features

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14.0-000000?style=flat&logo=next.js&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=flat&logo=opencv&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-6.0+-47A248?style=flat&logo=mongodb&logoColor=white)

### Core Capabilities

*   **Real-time Monitoring Dashboard**
    Varied grid layouts for live camera feeds with low-latency streaming and health status indicators.

*   **Advanced Object Detection**
    Integration with Ultralytics YOLO and OpenVINO for high-performance inference, capable of detecting a wide range of objects in real-time.

*   **Semantic Video Search**
    Utilizes CLIP models and Faiss vector databases to enable natural language searching of video archives (e.g., "Show me when a red truck entered the premise").

*   **Interactive AI Assistant**
    A conversational interface powered by LangChain and OpenAI, allowing users to query system status, retrieve logs, and analyze historical data through chat.

*   **Automated Alerting & Tracking**
    Robust multi-object tracking (BoT-SORT) combined with a configurable alert system for anomaly detection and perimeter breaches.

*   **Analytics & Reporting**
    Comprehensive visualization of detection statistics, traffic patterns, and system metrics.

---

## Technical Architecture

The system is built on a decoupled, scalable architecture.

### Backend (`/backend`)

The backend is engineered with **FastAPI** for high performance and asynchronous processing.
*   **Vision Pipeline**: Handles video ingestion, frame pre-processing, and inference pipelines.
*   **Intelligence Layer**: Manages LLM interactions, structured output parsing, and semantic indexing.
*   **Data Layer**: MongoDB for metadata storage and Faiss for high-dimensional vector similarity search.

### Frontend (`/frontend`)

The frontend is a modern **Next.js** application designed for responsiveness and interactivity.
*   **UI Framework**: Built with TailwindCSS and Radix UI components for a professional, accessible design system.
*   **Visualization**: Dynamic charts powered by Recharts for data analytics.
*   **Real-time Updates**: Websocket integration for live alert propagation and dashboard synchronization.

---

## Getting Started

Follow these instructions to set up the project locally for development.

### Prerequisites

*   Python 3.10+
*   Node.js 18+
*   MongoDB Instance
*   CUDA-compatible GPU (Recommended for inference acceleration)

### Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment:**
    Create a `.env` file in the `backend` directory. Here is a template with standard defaults:

    ```ini
    # App Settings
    APP_NAME="AI-Powered-Surveillance-System-with-Natural-Language-based-Querying"
    DEBUG=true
    ALLOWED_ORIGINS="http://localhost:3000"

    # Database
    MONGODB_URI="mongodb://localhost:27017"
    MONGO_DB_NAME="ai_surveillance"

    # AI & ML Models
    MODEL_PATH="yolo11m-seg.pt"
    # LLM Provider (openai or ollama)
    LLM_PROVIDER="openai"
    OPENAI_API_KEY="sk-..."
    # If using Ollama locally
    # OLLAMA_BASE_URL="http://localhost:11434"

    # Semantic Search & Vector Store
    ENABLE_SEMANTIC=true
    EMBED_DEVICE="cuda" # Use 'cpu' if no GPU available
    HF_TOKEN="hf_..."
    ```

5.  **Start the server:**
    ```bash
    uvicorn main:app --reload
    ```

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Start the development server:**
    ```bash
    npm run dev
    ```

4.  **Access the application:**
    Open your browser and navigate to `http://localhost:3000`.

---

## System Requirements

| Component | Minimum Specification | Recommended Specification |
|:---|:---|:---|
| **OS** | Ubuntu 20.04 LTS / macOS 12+ | Ubuntu 22.04 LTS |
| **CPU** | 4 Cores | 8+ Cores |
| **RAM** | 16 GB | 32 GB+ |
| **GPU** | NVIDIA GTX 1060 (6GB) | NVIDIA RTX 3060 (12GB) or higher |
| **Storage** | 50 GB SSD | 500 GB NVMe SSD |

