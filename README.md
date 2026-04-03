# AI Writing Assistant (DocuMesh)

Welcome to the **AI Writing Assistant** platform! This is a comprehensive, microservices-based application designed to elevate your writing experience. It leverages multiple specialized AI models for paraphrasing, grammar checking, summarization, tone adjustment, text simplification, and also features a Retrieval-Augmented Generation (RAG) system.

## 🌟 Features

- **Paraphrase**: Rephrase your textual content in different styles while preserving meaning.
- **Grammar Check**: Automatically detect and fix grammatical errors.
- **Simplify**: Make complex texts easier to read and understand.
- **Tone Adjustment**: Modify the tone of your text (e.g., formal, casual, persuasive).
- **Summarization**: Extract core insights from lengthy documents.
- **RAG (Retrieval-Augmented Generation)**: Intelligent querying against your uploaded knowledge base.

## 🏗 Architecture

The platform follows a modern microservices architecture containerized with Docker, comprising the following components:

- **Frontend**: A sleek, responsive user interface built with React & Vite.
- **API Gateway**: A central entry point for routing client requests to the appropriate AI services.
- **NLP Services**: A suite of specialized, decoupled services:
  - `paraphrase-service`
  - `grammar-service`
  - `simplify-service`
  - `tone-service`
  - `summarize-service`
- **RAG Service & Worker**: Handles embedding, document vectorization, and query resolution.
- **Data Stores**:
  - **PostgreSQL**: Relational database for structured application data.
  - **Redis**: High-performance caching and message broker for Celery queues.
  - **Qdrant**: Advanced vector database tailored for RAG embeddings.
- **Observability Stack**: Prometheus & Grafana to monitor system health and metrics.

## 🚀 Getting Started

### Prerequisites

- Docker
- Docker Compose

### Running the Application Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Emperor-4037/DocuMesh.git
   cd DocuMesh
   ```

2. **Start the environment with Docker Compose:**
   Run the following command to build and launch all microservices in disconnected mode:
   ```bash
   docker-compose up -d --build
   ```

3. **Access the application:**
   - **Frontend UI**: http://localhost:3000
   - **Grafana Dashboard**: http://localhost:3001 (Credentials defaults are via `.env` or `admin`/`admin`)

## 🛠 Tech Stack

- **Frontend**: React, Vite
- **Backend & Gateway**: Python, FastAPI / Flask (containerized NLP microservices)
- **Database**: PostgreSQL (Relational), Qdrant (Vector)
- **Queuing & Cache**: Redis, Celery
- **Monitoring**: Prometheus, Grafana
- **Infrastructure**: Docker, Docker Compose

## 🤝 Contributing

We welcome contributions! Please follow our established workflows:
1. Fork the repo and create your feature branch: `git checkout -b feature/my-cool-feature`
2. Commit your changes: `git commit -m 'Add some feature'`
3. Push to the branch: `git push origin feature/my-cool-feature`
4. Submit a pull request.
