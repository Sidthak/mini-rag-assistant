# Mini RAG Assistant

A lightweight Retrieval-Augmented Generation (RAG) prototype that answers natural language questions from uploaded documents — with source citations, confidence scores, and multi-turn conversation memory.

Built for the Firstsource STEM POC assessment.

---

## What It Does

- Upload any PDF or TXT file as your knowledge base
- Ask questions in plain English
- Get answers grounded strictly in your documents — no hallucination
- Every answer includes a confidence score and the exact source passages used
- Follow-up questions work naturally thanks to multi-turn memory

---

## Project Structure

```
mini_rag_assistant/
├── app.py              # Streamlit UI — file upload, chat interface, confidence display
├── rag_engine.py       # Core RAG logic — ingestion, retrieval, generation
├── config.py           # All tunable parameters in one place
├── requirements.txt    # Python dependencies
├── .env                # API key (not committed to git)
└── .gitignore
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.11 or higher
- An OpenAI API key (get one at https://platform.openai.com/api-keys)

### 2. Clone and install

```bash
git clone <your-repo-url>
cd mini_rag_assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

### 4. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | ≥1.45.0 | Web UI |
| langchain | ≥0.3.0 | RAG orchestration |
| langchain-community | ≥0.3.0 | Document loaders, ChromaDB integration |
| langchain-openai | ≥0.2.0 | OpenAI embeddings and chat model |
| chromadb | ≥0.5.23 | Local vector database |
| openai | ≥1.50.0 | API client |
| pypdf | ≥5.0.0 | PDF parsing |
| python-dotenv | ≥1.0.1 | Environment variable management |
| tiktoken | ≥0.8.0 | Token counting for chunking |

---

## Architecture

### Ingestion Pipeline (runs once per document set)

```
User uploads PDF or TXT
        ↓
Document loaded via PyPDFLoader or TextLoader
        ↓
Split into chunks (500 chars, 100 char overlap)
        ↓
Each chunk embedded using OpenAI text-embedding-ada-002
        ↓
Embeddings stored in local ChromaDB vector database
```

### Query Pipeline (runs on every question)

```
User types a question
        ↓
Question embedded using same ada-002 model
        ↓
Cosine similarity search in ChromaDB → top 4 most relevant chunks retrieved
        ↓
Chunks + full conversation history + question sent to GPT-3.5-turbo
        ↓
LLM generates answer strictly from retrieved context
        ↓
Answer returned with confidence score + source citations
```

### Multi-Turn Memory

Every query passes the full conversation history (as LangChain `HumanMessage` and `AIMessage` objects) to the LLM. This means follow-up questions like "explain that simpler" or "give me an example" work without repeating context.

### Key Design Decisions

| Decision | Reason |
|---|---|
| ChromaDB over FAISS | Built-in persistence, no manual serialization |
| chunk_size=500, overlap=100 | Balances retrieval precision vs context preservation |
| temperature=0 | Reduces creativity, maximises factual grounding |
| score_threshold=0.3 | Filters weakly related chunks before sending to LLM |
| Strict grounding prompt | Instructs LLM to never answer outside retrieved context |

---

## Confidence Scoring

The confidence score is calculated as the **average cosine similarity** of the top-k retrieved chunks against the query embedding, scaled to a percentage.

```python
results = vector_store.similarity_search_with_relevance_scores(question, k=4)
avg = sum(score for _, score in results) / len(results)
confidence = round(min(avg * 100, 100.0), 1)
```

### Interpretation

| Score | Colour | Meaning |
|---|---|---|
| ≥ 70% | 🟢 Green | High match — answer is well-supported by the document |
| 40–69% | 🟡 Yellow | Moderate match — answer may be partially supported |
| < 40% | 🔴 Red | Low match — treat with caution, verify manually |

**Important:** This is an approximation based on embedding similarity, not a true probability. A high score means the retrieved passages are relevant to the question — it does not guarantee the answer is factually correct.

---

## Example Inputs and Outputs

### Test Setup

- **Documents uploaded:** `main_notes.pdf` (3.1 MB, Stanford CS229 ML notes), `20200202-sitrep-13-ncov-v3.pdf` (0.8 MB, WHO COVID-19 report)
- **Total corpus:** 3.9 MB across 2 PDFs

---

### Example 1 — Technical Question

**Input:**
```
Why is logistic regression better than linear regression for binary classification?
```

**Output:**
```
Logistic regression is better than linear regression for binary classification 
because it models the probability of the output being in a certain class using 
the sigmoid function, which constrains the output between 0 and 1, making it 
suitable for binary classification tasks.
```

**Confidence:** 77.7% 🟢
**Sources:** main_notes.pdf — 4 passages used

---

### Example 2 — Multi-Turn Follow-Up

**Input (follow-up to Example 1):**
```
Can you explain that simpler?
```

**Output:**
```
Logistic regression is better for binary classification because it gives outputs 
between 0 and 1, representing probabilities, which is more suitable for 
classifying into two categories.
```

**Confidence:** 66.6% 🟡
**Note:** Confidence dropped on the follow-up because "explain simpler" is a vague query — the system retrieved slightly less relevant chunks. The answer is still grounded in the document.

---

### Example 3 — Cross-Document Query

**Input:**
```
What is machine learning?
```

**Output:**
```
Machine learning is a method of training models to analyze and understand data, 
typically by minimizing a loss function to fit the data.
```

**Confidence:** 73.4% 🟢
**Sources:** main_notes.pdf — 4 passages used

---

### Example 4 — Out-of-Scope Question (Grounding Test)

**Input:**
```
What is the capital of France?
```

**Output:**
```
I couldn't find that in the provided documents.
```

**Confidence:** 12.3% 🔴
**Note:** The system correctly refused to answer — this information is not in the uploaded documents. This demonstrates grounding working as intended.

---

### Sample Log Output

```json
{
  "question": "Why is logistic regression better than linear regression for binary classification?",
  "answer": "Logistic regression is better than linear regression for binary classification because it models the probability of the output being in a certain class using the sigmoid function...",
  "confidence": 77.7,
  "sources": [
    {
      "source": "main_notes.pdf",
      "page": 16,
      "excerpt": "...logistic regression models p(y=1|x) using the sigmoid function g(z) = 1/(1+e^-z)..."
    },
    {
      "source": "main_notes.pdf",
      "page": 17,
      "excerpt": "...unlike linear regression, the output is constrained between 0 and 1..."
    }
  ]
}
```

---

## Limitations

- Supports PDF and TXT only — no Word, Excel, or scanned image PDFs
- Confidence score is an approximation, not a true probability
- No persistent chat history across browser sessions
- Large documents (100+ pages) may take longer to index on first load
- Requires an OpenAI API key — small cost per query (typically under $0.01)

---

## Data Sources Used

- **Stanford CS229 Machine Learning Lecture Notes** — Andrew Ng, publicly available at cs229.stanford.edu
- **WHO Situation Report 13 — COVID-19** — publicly available at who.int

---
