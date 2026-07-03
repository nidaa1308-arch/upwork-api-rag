# RAG-on-Docs

A Retr ieval-Augmented Generation (RAG) application that answers developer questions about the Upwork API by retrieving relevant information from the official documentation before generating a response. Instead of relying solely on the LLM's pre-trained knowledge, the system grounds every answer in retrieved documentation, improving factual accuracy and reducing hallucinations.

---

## Features

- Retrieval-Augmented Generation (RAG)
- Semantic Search using BAAI/bge-small-en-v1.5 embeddings
- ChromaDB Vector Store
- Cosine Similarity Retrieval
- Custom Rule-Based Reranking
- Groq-hosted Llama 3.1-8B-Instant
- Real-Time Streaming Responses
- Hallucination Guardrails
- Source Citations & Chunk Inspection
- Interactive Streamlit Interface

---

## System Architecture

### Offline Indexing Pipeline

```text
API Documentation (PDF)
        │
        ▼
Text Extraction & Chunking
        │
        ▼
Generate Embeddings (BAAI/bge-small-en-v1.5)
        │
        ▼
Store Embeddings in ChromaDB
(Cosine Similarity Index)
```

### Online Retrieval & Generation Pipeline

```text
User Query
     │
     ▼
Generate Query Embedding
     │
     ▼
Cosine Similarity Search (ChromaDB)
Retrieve Top-40 Candidate Chunks
     │
     ▼
Custom Rule-Based Reranker
     │
     ▼
Select Top-5 Relevant Chunks
     │
     ▼
Prompt Construction
     │
     ▼
Llama 3.1-8B-Instant (Groq)
     │
     ▼
Grounded Response + Source Chunks
```

---

## Retrieval Pipeline

For each user query, an embedding is generated using **BAAI/bge-small-en-v1.5**. ChromaDB performs cosine similarity search to retrieve the Top-40 candidate chunks from the vector store. A custom rule-based reranker then refines these candidates using semantic overlap, bigram matching, keyword scoring, and query-aware heuristics to improve retrieval precision. The Top-5 highest-scoring chunks are incorporated into the prompt and supplied to Llama 3.1, ensuring that responses are grounded in the retrieved documentation rather than the model's parametric knowledge.

---

## Design Decisions

### Why Retrieval-Augmented Generation?

Large Language Models may generate incorrect or outdated information when answering documentation-specific questions. RAG improves reliability by retrieving relevant documentation first and using it as context for generation.

### Why BAAI/bge-small-en-v1.5?

- Lightweight and efficient
- Strong semantic retrieval performance
- Well suited for technical documentation search.

### Why Cosine Similarity?

Sentence embeddings primarily encode semantic direction rather than magnitude. Cosine similarity measures the angle between vectors, making it a natural choice for comparing semantic similarity.

### Why ChromaDB?

- Persistent local vector database.
- Efficient similarity search.
- Simple integration with LangChain.

### Why Retrieve Top-40 and Rerank Top-5?

Retrieving a larger candidate pool improves recall by reducing the chance of missing relevant documentation. A custom rule-based reranking layer then improves precision by selecting the most relevant chunks before they are passed to the LLM.

### Why Groq?

Groq provides low-latency inference and streaming responses, making the chatbot feel responsive while maintaining high-quality generation.

---

## Tech Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| User Interface | Streamlit |
| Framework | LangChain |
| LLM | Llama 3.1-8B-Instant |
| Inference | Groq API |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| Retrieval Method | Cosine Similarity Search |
| Reranking | Custom Rule-Based Reranker |
| Environment Management | python-dotenv |

---

## Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit interface and chat workflow |
| `ingest.py` | PDF ingestion, text chunking, embedding generation, and ChromaDB index creation |
| `rag.py` | Semantic retrieval, cosine similarity search, and custom reranking |
| `llm.py` | Groq Llama 3.1 integration and response streaming |
| `prompts.py` | System prompts and hallucination guardrails |
| `config.py` | Centralized project configuration |

---

## Example Questions

- What is the rate limit for the Upwork API?
- How long is an OAuth access token valid for?
- Who can create subscriptions on the Upwork API?
- Can I use a Client Credentials Grant to access a user's private contract details?

---

## Challenges

One of the biggest challenges was improving retrieval generalization. Although the initial system answered manually tested queries correctly, retrieval quality dropped for differently worded questions because relevant chunks were not always ranked near the top.

To improve retrieval performance, the candidate pool was expanded from Top-24 to Top-40, a generic rule-based reranker was introduced, and the prompt was strengthened with additional hallucination guardrails. These improvements resulted in more reliable, context-grounded responses across a wider range of user queries.

---

## Future Improvements

- Hybrid Retrieval (BM25 + Dense Embeddings)
- Cross-Encoder Reranker
- Retrieval Evaluation using Precision@K, Recall@K, and MRR
- Multiple Document Collections
- Conversation Memory
- Docker Containerization & Cloud Deployment

---

## License

This project is licensed under the MIT License.
