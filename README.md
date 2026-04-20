<div align="center">

# Information Retrieval System

**A full-stack search engine over ophthalmology documents scraped from [anpig.it](http://www.anpig.it)**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Italian_NLP-4CAF50)

*Final project — Information Retrieval · University of Trieste (UniTS)*

</div>

---

## Overview

This project implements a search engine from scratch over a corpus of ophthalmology documents crawled from **www.anpig.it**. Users can submit queries and iteratively refine results through a relevance feedback mechanism.

The system is built without external search libraries — the inverted index, TF-IDF weighting, BM25 scoring, and Rocchio feedback are all implemented from first principles.

---

## Features

| Component | Description |
|---|---|
| **Inverted index** | Built from a 372 KB JSON corpus at startup |
| **Italian NLP pipeline** | Tokenisation · Snowball stemming · stopword removal (NLTK) |
| **BM25 scoring** | Okapi BM25 for document ranking |
| **TF-IDF query vector** | Weighted query representation |
| **Rocchio feedback** | Query vector adjusted toward relevant documents (α=1.0 β=0.75 γ=0.25) |
| **Evaluation metrics** | Precision · Recall · F1 displayed after each feedback round |
| **Flask web UI** | Search page + results page with inline feedback form |

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/GigiMoschetta/Information-Retrieval.git
cd Information-Retrieval
docker compose up --build
```

Open **http://localhost:5002**

> On the first run, the inverted index is built automatically from the corpus (~1–2 s). Subsequent runs load from cache.

### Manual

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

---

## How It Works

```
Query → Italian NLP pipeline → TF-IDF query vector
      ↓
BM25 scoring over inverted index → ranked results
      ↓
User marks relevant / not-relevant documents
      ↓
Rocchio algorithm adjusts query vector → refined results + F1 score
```

---

## Project Structure

```
.
├── app.py                             # Flask app — indexing, search, feedback
├── data/
│   └── dysderadb.anpig_complete.json  # Scraped corpus (372 KB)
├── templates/
│   ├── search.html                    # Search page
│   └── results.html                  # Results + feedback form
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tech Stack

`Python 3.11` · `Flask` · `NLTK` · `dill` · `Bootstrap 5` · `Docker`
