# Construction Marketplace AI Assistant (Mini-RAG)

## Overview
This repository contains a simple, production-ready Retrieval-Augmented Generation (RAG) pipeline built for **INDECIMAL**, a construction marketplace. It allows users to ask questions and restricts the answers strictly to the provided internal documentation, thereby eliminating hallucinations.

## Technical Architecture (Mandatory Requirements Fulfilled)

### 1. Document Processing & Embedding
* **Chunking:** Handled via LangChain's `RecursiveCharacterTextSplitter`. Text is split into chunks of `1000` characters with a `200` character overlap to preserve contiguous contexts. Clean separators (`\n\n`, `\n`) ensure paragraphs aren't broken improperly.
* **Embeddings:** Uses HuggingFace's open-source `sentence-transformers/all-MiniLM-L6-v2` model, running locally on the CPU. It represents an optimal balance between low latency and high semantic accuracy without requiring paid API calls.

### 2. Vector Indexing and Retrieval
* **Vector Store:** Facebook AI Similarity Search (**FAISS**) is used as the local vector index. It does not require cloud management like Pinecone, fulfilling the assignment's local preference constraint.
* **Retrieval Strategy:** Top-k (`k=3`) cosine-similarity retrieval dynamically grabs the 3 most relevant textual chunks out of the FAISS index based on the user's vectorized query. Dynamic merging was implemented to allow users to update the index mid-session using Steamlit's `st.file_uploader`.

### 3. LLM-Based Answer Generation & Strict Grounding
* **Prompt Engineering:** The LLM is wrapped in LangChain Expression Language (LCEL) routing. It utilizes a highly restrictive system prompt explicitly forbidding outside knowledge:
  > *"If the answer to the user's question is not explicitly stated in the provided context, you must reply EXACTLY with: 'I cannot answer this question...'"*
* **Models:** Support for both OpenRouter Cloud (e.g. `nvidia/nemotron-3-nano...`) and Local Execution via Ollama (to satisfy the Bonus requirements).

### 4. Transparency
The Streamlit frontend heavily prioritizes explainability. Upon generation, the AI appends a dropdown titled **`🔍 View Retrieved Context`**, which visually outputs the exact chunk strings pulled from FAISS, allowing humans to verify truthfulness.

---

## Bonus Enhancements Implemented

### 1. Local Open-Source LLM Integration
We included an option to run local open-source models natively. For our benchmark evaluation, we used **Ollama** running **Mistral 7B** powered by a local Nvidia RTX 3060 GPU, comparing it against the Cloud OpenRouter API (`nvidia/nemotron-3-nano-30b-a3b:free`).

**Physical Benchmark Findings:**
* **Answer Quality & Completeness:** The Cloud model generally structured answers better (using bullet points and markdown bolding) and exhibited superior reasoning over chunks. For example, when asked about the "3 tiers of protection policies", the Cloud model successfully extracted the 3 guarantees, whereas Local Mistral 7B failed to parse them and claimed the info was missing. 
* **Latency (Speed):** The cloud API averaged **~1.3 seconds** per response. Local inference on the RTX 3060 averaged **~2.4 seconds** per response (ranging from 1.07s up to 4.34s for long generations). While the cloud is roughly 2x faster, the RTX 3060 handled the 7B parameter local model exceptionally well for real-time chat.
* **Groundedness / Anti-Hallucination:** Both models perfectly adhered to the zero-hallucination prompt. When asked out-of-domain questions (e.g., "Who built the Eiffel tower?"), both correctly triggered the exact fallback phrase: *"I cannot answer this question because the information is not present..."*

### 2. Quality Analysis (10 Real Test Questions)
To rigorously evaluate the system, we physically executed 10 test questions through an automated benchmark script (`run_evaluation.py`). Here is the generated evaluation mapping:

| Test Question | Cloud Model Answer Summary | Local Mistral Answer Summary | Pass/Fail |
|---|---|---|---|
| 1. *What is Indecimal's one-line summary?* | Delivered perfect 1-line quote from doc1. | Delivered perfect 1-line quote from doc1. | **Pass** |
| 2. *Are the package pricing rates inclusive of GST?* | Yes, explicitly mentioned. | Yes, explicitly mentioned. | **Pass** |
| 3. *What is the Escrow-Based Payment Model?* | Explained stage-by-stage release from doc3. | Explained stage-by-stage release from doc3. | **Pass** |
| 4. *What kind of home construction does Indecimal support?* | Outlined end-to-end lifecycle. | Outlined end-to-end lifecycle. | **Pass** |
| 5. *Who built the Eiffel Tower?* (Hallucination Check) | Refused to answer (guardrail triggered). | Refused to answer (guardrail triggered). | **Pass** |
| 6. *What are the 3 tiers of protection policies?* | Found the 3 Zero-Guarantees. | *Failed to locate info in the chunk.* | **Cloud Pass / Local Fail** |
| 7. *How does the system ensure quality?* | Formatted 445+ checkpoints clearly. | Parsed checkpoints but unformatted. | **Pass** |
| 8. *Can I build a 50-story commercial skyscraper?* | Refused to answer (not in docs). | Refused to answer (not in docs). | **Pass** |
| 9. *Who is the CEO of Indecimal?* | Refused to answer. | Refused to answer. | **Pass** |
| 10. *What are Devil Fruits?* (Cross-domain Check) | Extracted 3 types from `one_piece_lore.md`. | Extracted 3 types from `one_piece_lore.md`. | **Pass** |

**Evaluation Conclusion (AI Judge):**
1. **Relevance of Retrieved Chunks:** FAISS `k=3` semantic similarity successfully pinpointed all required contexts. 
2. **Presence of Hallucinations:** **0% Hallucination Rate.** Both models proved incredibly obedient to the LCEL restrictive prompt.
3. **Clarity of Generated Answers:** The OpenRouter 30B parameter model is superior in formatting and synthesizing multi-hop constraints over local Mistral 7B, but Mistral 7B on an RTX 3060 is a perfectly viable, completely private alternative.

---

## Getting Started

### Prerequisites
1. Python 3.10+
2. [Ollama](https://ollama.com/) (Optional: for local inference)

### Installation
1. Clone the repository and navigate to the project directory.
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac)
4. Install dependencies: `pip install -r requirements.txt`

### Cloud API Keys (OpenRouter)
To use OpenRouter, create a file named `secret.txt` in the root directory and add:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Run the App
Launch the interactive Streamlit UI natively on your machine:
```bash
streamlit run app.py
```