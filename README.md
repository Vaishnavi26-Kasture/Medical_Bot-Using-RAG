# RAG_Disease-Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers medical encyclopedia questions, grounded in *The Gale Encyclopedia of Medicine*. It retrieves relevant passages from a local FAISS vector store and uses an LLM to generate answers with source attribution — reducing hallucination compared to asking an LLM directly.

> **⚠️ Disclaimer:** This project is for educational purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

## How It Works

PDF (Gale Encyclopedia of Medicine)

│

▼

Load & chunk text  (create_memory_for_llm.py)

│

▼

Embed chunks (sentence-transformers/all-MiniLM-L6-v2)

│

▼

FAISS vector store  (vectorstore/db_faiss)

│

▼

Query → retrieve top-k chunks → LLM → grounded answer

(create_memory_with_llm.py)

1. **Indexing** — `create_memory_for_llm.py` loads PDFs from `data/`, splits them into chunks, embeds them, and saves a FAISS index to `vectorstore/db_faiss`.
2. **Querying** — `create_memory_with_llm.py` loads that FAISS index, retrieves the most relevant chunks for a user's question, and passes them as context to an LLM via a `RetrievalQA` chain, returning an answer plus the source document(s) used.

## Project Structure

RAG_Disease-Chatbot/

├── data/                       # Place source PDF(s) here (not committed)

├── vectorstore/db_faiss/       # Generated FAISS index (not committed)

├── create_memory_for_llm.py    # Step 1: build the vector store from PDFs

├── create_memory_with_llm.py   # Step 2: query the vector store via an LLM

├── test.py                     # Quick script to sanity-check HF endpoint connectivity

├── requirement.txt             # Python dependencies

├── .env                        # HF_TOKEN (not committed)

└── README.md


## Tech Stack

- **LangChain** — orchestration (`RetrievalQA` chain)
- **FAISS** — vector similarity search
- **HuggingFace Inference API** — LLM inference (`HuggingFaceEndpoint` + `ChatHuggingFace`)
- **sentence-transformers/all-MiniLM-L6-v2** — embedding model
- **python-dotenv** — environment variable management

## Output

![Scrrenshot(107).png](https://github.com/Vaishnavi26-Kasture/Medical_Bot-Using-RAG/blob/main/Screenshot%20(107).png)


