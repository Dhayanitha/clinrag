# ClinRAG
### Evidence-Grounded Clinical Knowledge Retrieval System

ClinRAG is a Retrieval-Augmented Generation (RAG) application that answers clinical knowledge questions using evidence retrieved from the MedQuAD dataset. It combines dense retrieval, lexical search, rank fusion, reranking, and a locally running language model to generate evidence-grounded responses.

**Disclaimer:** This is a research prototype for clinical information retrieval. It is not intended for diagnosis or personalized medical advice.

## Features

- MedQuAD XML data ingestion and preprocessing
- Semantic retrieval using Sentence Transformers and FAISS
- Lexical retrieval using BM25
- Hybrid ranking using Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Local answer generation using Ollama and Qwen2.5 1.5B
- FastAPI endpoint for question answering
- Streamlit interface for interactive queries
- Display of retrieved evidence, relevance scores, and source links
- Initial retrieval evaluation using Recall@5 and Mean Reciprocal Rank (MRR)

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
Ollama (Qwen2.5 1.5B)
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
| Hybrid ranking | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder |
| Language model | Qwen2.5 1.5B via Ollama |
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
└── tests/
```

The `data/` directory contains the MedQuAD dataset and generated artifacts, which are excluded from Git using `.gitignore`.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
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

### 5. Prepare the dataset

Download the MedQuAD dataset and place its XML files under the expected `data/raw/MedQuAD/` directory.

Run the ingestion and cleaning scripts from the project root:

```bash
python src/data_ingestion.py
python src/data_cleaning.py
```

Run the embedding and FAISS index creation scripts as applicable to your implementation. The processed CSV files, embeddings, and index must exist before retrieval can run.

## Run the Application

### Streamlit interface

```bash
streamlit run app.py
```

### FastAPI service

```bash
uvicorn src.api:app --reload
```

Open the FastAPI interactive documentation at:

http://127.0.0.1:8000/docs

The API includes a health endpoint and a question-answering endpoint.

## Retrieval Evaluation

The initial evaluation used five example queries with known relevant document IDs.

| Retrieval method | Recall@5 | MRR |
|---|---:|---:|
| FAISS dense retrieval | 1.00 | 0.90 |
| BM25 | 0.60 | 0.40 |
| Hybrid RRF | 1.00 | 0.7333 |
| Hybrid + Cross-Encoder | 1.00 | 0.80 |

These results are preliminary and based on a very small evaluation set. They should not be interpreted as evidence of general retrieval performance or clinical accuracy.

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

