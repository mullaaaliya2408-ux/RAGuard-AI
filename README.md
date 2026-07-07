# 🛡️ RAGuard AI
### Self-Correcting Retrieval-Augmented Generation (RAG) with Explainable AI

> **An intelligent Retrieval-Augmented Generation system that knows when it doesn't know.**
>
> RAGuard AI combines Hybrid Retrieval, Self-Correction, Evidence Verification, Reflection, and Explainable Confidence Scoring to produce trustworthy AI-generated answers while minimizing hallucinations.

---

## 🚀 Overview

Large Language Models often generate confident but incorrect responses when retrieved information is incomplete or contradictory.

**RAGuard AI** solves this problem through a multi-agent retrieval pipeline that continuously evaluates its own evidence before answering.

Instead of blindly generating responses, the system can:

- 🔍 Retrieve relevant evidence
- 🧠 Evaluate retrieval quality
- 🔄 Rewrite ambiguous queries
- 📚 Retrieve again if necessary
- ✅ Verify supporting evidence
- ⚠️ Detect contradictory documents
- 🤔 Reflect on answer quality
- 📊 Generate an explainable confidence score
- ❌ Refuse to answer when evidence is insufficient

---

# ✨ Features

## 📄 Intelligent Document Processing

- Digital PDF Processing
- OCR for Scanned PDFs
- Image Support
- Automatic Document Classification
- Metadata Extraction
- Smart Chunking
- Incremental Indexing

---

## 🔍 Hybrid Retrieval Engine

- FAISS Semantic Search
- BM25 Keyword Search
- Metadata Search
- Query Expansion
- Weighted Hybrid Ranking
- Top-K Retrieval

---

## 🧠 Self-Correction Loop

Implements an autonomous retrieval improvement cycle:

User Query

↓

Query Understanding

↓

Hybrid Retrieval

↓

Retrieval Quality Evaluation

↓

Evidence Verification

↓

Query Rewrite

↓

Retrieve Again

↓

Reflection Agent

↓

Final Answer

---

## 🤖 AI Agents

### Query Understanding Agent

- Intent Detection
- Entity Extraction
- Query Rewriting
- Ambiguity Detection

---

### Retrieval Quality Checker

Evaluates:

- Similarity Score
- Coverage Score
- Evidence Diversity
- Metadata Completeness

---

### Evidence Verification Agent

Checks whether retrieved chunks actually answer the user's question instead of relying solely on similarity scores.

---

### Contradiction Detection Agent

Detects conflicting information between retrieved documents and alerts users before generating an answer.

---

### Reflection Agent

The model critiques its own reasoning.

It can decide to:

- Retrieve Again
- Continue
- Return Low Confidence

---

## 📊 Explainable Confidence Engine

Confidence is computed using multiple interpretable factors.

Factors include:

- Semantic Similarity
- Verification Confidence
- Supporting Chunk Count
- Evidence Agreement
- OCR Confidence
- Chunk Metadata Quality
- Retrieval Success

Confidence Levels:

- 🟢 High
- 🟡 Medium
- 🔴 Low

---

# 🏗️ System Architecture

```
                 User Query
                     │
                     ▼
         Query Understanding Agent
                     │
                     ▼
          Hybrid Retrieval Engine
          (FAISS + BM25 + Metadata)
                     │
                     ▼
      Retrieval Quality Assessment
                     │
          ┌──────────┴──────────┐
          │                     │
      Quality Pass?         Rewrite Query
          │                     │
          ▼                     │
 Evidence Verification ◄────────┘
          │
          ▼
 Contradiction Detection
          │
          ▼
     Reflection Agent
          │
          ▼
 Confidence Engine
          │
          ▼
      Final Answer
```

---

# ⚙️ Tech Stack

## Frontend

- Next.js 16
- React
- TypeScript
- Tailwind CSS

---

## Backend

- FastAPI
- Python
- Pydantic
- Uvicorn

---

## AI / NLP

- Google Gemini
- Sentence Transformers
- FAISS
- BM25
- OCR

---

## Data Processing

- PDF Parsing
- OCR Processing
- Chunking
- Metadata Extraction

---

# 📂 Project Structure

```
RAGuard-AI
│
├── backend
│   ├── api
│   ├── services
│   ├── embeddings
│   ├── vectorstore
│   ├── models
│   ├── documents
│   └── scripts
│
├── frontend
│   ├── src
│   │   ├── app
│   │   ├── components
│   │   └── lib
│   └── public
│
└── README.md
```

---

# 🖥️ Installation

## Clone

```bash
git clone https://github.com/mullaaaliya2408-ux/RAGuard-AI.git

cd RAGuard-AI
```

---

## Backend

```bash
cd backend

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

uvicorn main:app --reload
```

Backend runs on:

```
http://127.0.0.1:8000
```

Swagger:

```
http://127.0.0.1:8000/docs
```

---

## Frontend

```bash
cd frontend

npm install

npm run dev
```

Frontend:

```
http://localhost:3000
```

---

# 📸 Screenshots

## Dashboard

(Add Screenshot)

---

## Upload Documents

(Add Screenshot)

---

## Ask Questions

(Add Screenshot)

---

## Evidence Viewer

(Add Screenshot)

---

## Confidence Report

(Add Screenshot)

---

## Evaluation Dashboard

(Add Screenshot)

---

# 📈 Benchmark Evaluation

The evaluation pipeline measures:

- Accuracy
- Precision
- Recall
- Hallucination Rate
- Average Confidence
- Retrieval Latency

---

# 🔒 Hallucination Prevention

Unlike traditional RAG systems, RAGuard AI can:

✔ Refuse unsupported questions

✔ Rewrite unclear queries

✔ Verify retrieved evidence

✔ Detect conflicting documents

✔ Explain confidence scores

✔ Reflect before answering

---

# 🎯 Future Improvements

- Multi-document summarization
- Knowledge Graph Retrieval
- Multimodal RAG
- Streaming Responses
- Agent Memory
- Multi-language Support
- Citation Ranking
- Vector Database Support
- Docker Deployment
- Kubernetes Deployment

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository

2. Create a new branch

3. Commit your changes

4. Submit a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👩‍💻 Author

**Aaliya Mulla**

Artificial Intelligence & Data Science Engineer

Passionate about Explainable AI, Retrieval-Augmented Generation, NLP, and Intelligent Systems.

GitHub:
https://github.com/mullaaaliya2408-ux

---

## ⭐ Support

If you found this project useful,

⭐ Star this repository!
