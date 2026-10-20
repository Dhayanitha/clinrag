# ClinRAG
### Evidence-Grounded Clinical Knowledge Retrieval System

ClinRAG is a Retrieval-Augmented Generation (RAG) application that answers clinical knowledge questions using evidence retrieved from the MedQuAD dataset. It combines dense retrieval, lexical search, rank fusion, and cross-encoder reranking, with Google Gemini for deployed answer generation and Ollama as a local alternative.

**Disclaimer:** This is a research prototype for clinical information retrieval. It is not intended for diagnosis or personalized medical advice.

## Features

- MedQuAD XML data ingestion and preprocessing
- Semantic retrieval using Sentence Transformers and FAISS
- Lexical retrieval using BM25
- Hybrid ranking using Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- LLM-based answer generation using Google Gemini or local Ollama
- Automatic retries for temporary Gemini server errors
- FastAPI endpoint for question answering
- Streamlit interface for interactive queries
- Display of retrieved evidence, relevance scores, and source links
- Retrieval evaluation using HitRate@5 and Mean Reciprocal Rank (MRR)

## Architecture

```text
User Question
      |
      v
Dense Retrieval (FAISS) + BM25 Retrieval
      |
      v
Reciprocal Rank Fusion (RRF)
      |
      v
Cross-Encoder Reranking
      |
      v
Top Retrieved Evidence
      |
      v
LLM Answer Generation
(Google Gemini in deployment / Ollama locally)
      |
      v
Generated Answer + Evidence + Sources
      |
      v
Streamlit UI / FastAPI
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Dataset | MedQuAD |
| Data processing | Pandas, NumPy |
| Embeddings | `all-MiniLM-L6-v2` |
| Vector retrieval | FAISS |
| Lexical retrieval | BM25 |
| Hybrid ranking | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-Encoder |
| Cloud language model | Google Gemini (`gemini-flash-latest`) |
| Local language model | Qwen2.5 1.5B via Ollama |
| API | FastAPI |
| User interface | Streamlit |

## Project Structure

```text
clinrag/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── src/
│   ├── api.py
│   ├── data_ingestion.py
│   ├── data_cleaning.py
│   ├── embeddings.py
│   ├── build_index.py
│   ├── index.py
│   ├── retrieval.py
│   ├── bm25_retrieval.py
│   ├── hybrid_retrieval.py
│   ├── reranker.py
│   ├── reranked_retrieval.py
│   ├── generation.py
│   ├── rag_pipeline.py
│   ├── create_evaluation_set.py
│   └── evaluate_retrieval.py
├── notebooks/
├── data/
│   ├── raw/
│   └── processed/
```

The `data/` directory contains the MedQuAD dataset and generated artifacts, which are excluded from Git using `.gitignore`.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Dhayanitha/clinrag.git
cd clinrag
```

### 2. Create and activate a Conda environment

```bash
conda create -n clinrag python=3.11
conda activate clinrag
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama from https://ollama.com and download the model:

```bash
ollama pull qwen2.5:1.5b
```

Make sure the Ollama service is running before submitting questions.

### 5. Prepare the dataset and build the retrieval index

Download the MedQuAD dataset and place its XML files under `data/raw/MedQuAD/`.

Run the following commands from the project root in the activated `clinrag` environment:

```bash
python src/data_ingestion.py
python src/data_cleaning.py
python src/embeddings.py
python src/build_index.py
```

These scripts process the XML files, clean the extracted clinical knowledge, generate text embeddings, and build the FAISS retrieval index.

Ensure the processed CSV files, embeddings, and index are generated successfully before running the application. The dataset and generated artifacts are excluded from Git and must be prepared locally.

## Run the Application

### Configure the language model

ClinRAG supports two language model providers:

- **Google Gemini:** The default provider, used by the deployed Streamlit application. Configure `GEMINI_API_KEY` through Streamlit secrets or an environment variable. Never commit API keys to Git.
- **Ollama:** An alternative for local development. Install Ollama and download the model:

```bash
ollama pull qwen2.5:1.5b
```

To select Ollama in Windows PowerShell for the current terminal session, run:

```powershell
$env:LLM_PROVIDER = "ollama"
```
To use Gemini explicitly in the same session, run:

```powershell
$env:LLM_PROVIDER = "gemini"
```

Start the application from that terminal after selecting the provider. Gemini requires a valid API key; Ollama requires its service and model to be available locally.

### Streamlit interface

The Streamlit interface uses Google Gemini for answer generation when `GEMINI_API_KEY` is configured.

For local development, configure the API key using Streamlit secrets or an environment variable. Never commit API keys to Git.

Start the interface from the project root:

```bash
streamlit run app.py
```

### FastAPI service

Start the API from the project root:

```bash
uvicorn src.api:app --reload
```

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

The API provides a health endpoint and a question-answering endpoint.

**Note:** The API and Streamlit interface are separate ways to interact with the project. Ensure the required dataset artifacts and model dependencies are available before starting either service.

## Retrieval Evaluation

The retrieval pipeline was evaluated using 35 queries with known relevant document IDs. Performance was measured using HitRate@5 and Mean Reciprocal Rank (MRR).

| Retrieval method | HitRate@5 | MRR |
|---|---:|---:|
| FAISS dense retrieval | 0.9143 | 0.7819 |
| BM25 | 0.8286 | 0.7248 |
| Hybrid RRF | 0.9429 | 0.7486 |
| Hybrid + Cross-Encoder | 0.9714 | 0.8176 |

The Hybrid + Cross-Encoder pipeline achieved a HitRate@5 of 97.14% and an MRR of 0.8176 on this evaluation set.

**Evaluation caveat:** These results are preliminary. The 35-query set is a development evaluation, not an independent held-out test set, and relevance labels are based on matching source document IDs. Performance may therefore be optimistic. Retrieval metrics measure document retrieval quality, not clinical accuracy or the correctness of generated answers.

## Limitations

- The evaluation set is small and needs expansion.
- Retrieved evidence may include passages that are only partially relevant.
- Generated responses can contain errors and should be checked against the cited source material.
- Source URLs should be verified for availability and accuracy.
- The application is a research prototype and is not a substitute for professional medical advice.

## Future Improvements

- Expand the retrieval evaluation dataset.
- Improve evidence relevance and source validation.
- Add automated tests for retrieval, generation, and API behavior.
- Improve citation grounding and answer quality evaluation.
- Explore deployment options for the API and user interface.

## Acknowledgements

This project uses the MedQuAD dataset for clinical question-answering research. Please consult the dataset's original repository and license terms before redistribution.

