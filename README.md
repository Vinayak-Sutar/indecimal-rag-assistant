# Construction Marketplace AI Assistant (Mini-RAG)

This repository contains a Retrieval-Augmented Generation (RAG) pipeline built for **INDECIMAL**, a construction marketplace. It acts as an internal knowledge base assistant, allowing users to ask questions about company policies and specifications. To ensure reliability, the pipeline restricts answers exclusively to the provided internal documentation, strongly preventing AI hallucinations.

## Technical Architecture

### 1. Document Processing & Embedding
* **Chunking:** I used LangChain's `RecursiveCharacterTextSplitter`. The documents are split into chunks of `1000` characters with a `200` character overlap to make sure context isn't lost between paragraphs.
* **Embeddings:** I opted for HuggingFace's open-source `sentence-transformers/all-MiniLM-L6-v2` model. It runs locally on the CPU and offers a great balance between low latency and accurate semantic search without needing paid API calls.

### 2. Vector Indexing and Retrieval
* **Vector Store:** For the local vector database, I used **FAISS** (Facebook AI Similarity Search). This avoids the need for external cloud management tools like Pinecone.
* **Retrieval Strategy:** The system uses top-k (`k=3`) cosine-similarity retrieval. When a user asks a question, it finds the 3 most relevant textual chunks from the FAISS index. I also added dynamic merging, so users can upload new PDFs/MD files mid-session through the Streamlit interface and instantly query them.

### 3. LLM-Based Answer Generation & Grounding
* **Prompt Engineering:** The language model is orchestrated using LangChain Expression Language (LCEL). I wrote a highly restrictive system prompt to enforce grounding:
  > *"If the answer to the user's question is not explicitly stated in the provided context, you must reply EXACTLY with: 'I cannot answer this question...'"*
* **Model Selection:** The application supports two modes:
  1. A cloud-based model via OpenRouter API (`nvidia/nemotron-3-nano-30b`).
  2. Local execution via Ollama (e.g., `Mistral 7B` or `Gemma 2B`), allowing the entire pipeline to run offline.

### 4. UI & Transparency
The app runs on Streamlit. To ensure explainability, every time the assistant generates an answer, it populates a dropdown titled **`🔍 View Retrieved Context`**. This allows users to physically see the exact chunks pulled from FAISS and verify that the AI is telling the truth.

---

## Benchmark Evaluation (Local vs Cloud)

I ran a quick script (`run_evaluation.py`) to benchmark the system using 10 test questions based on the internal docs. I compared a local **Mistral 7B** (running on an RTX 3060) against the OpenRouter Cloud API.

**Key Findings:**
* **Answer Quality:** The Cloud model generally structured answers a bit better (using markdown and bullet points) and was slightly better at reasoning across multiple chunks. For instance, when asked to list "3 tiers of policies", the larger cloud model successfully extracted them while the local 7B model missed them.
* **Latency:** The cloud API averaged **~1.3 seconds** per response. Local inference on the RTX 3060 averaged **~2.4 seconds**. Both are easily fast enough for a real-time chat interface.
* **Groundedness:** Both models adhered perfectly to the zero-hallucination prompt. When asked out-of-domain questions like *"Who built the Eiffel tower?"*, both correctly triggered the fallback phrase and refused to answer.

### Summary of the 10 Test Questions:
1. *What is Indecimal's one-line summary?* — **Pass** 
2. *Are the package pricing rates inclusive of GST?* — **Pass**
3. *What is the Escrow-Based Payment Model?* — **Pass**
4. *What kind of home construction does Indecimal support?* — **Pass**
5. *Who built the Eiffel Tower?* (Hallucination Check) — **Pass** (Refused to answer)
6. *What are the 3 tiers of protection policies?* — **Cloud Pass / Local Fail**
7. *How does the system ensure quality?* — **Pass**
8. *Can I build a 50-story commercial skyscraper?* — **Pass** (Refused to answer)
9. *Who is the CEO of Indecimal?* — **Pass** (Refused to answer)
10. *What are Devil Fruits?* (Cross-domain Check) — **Pass** (Successfully extracted offline)

Across both systems, the hallucination rate was 0%.

---

## Local Setup Instructions

### Prerequisites
1. Python 3.10+
2. [Ollama](https://ollama.com/) (Optional, required only if you want to test local inference)

### Installation
1. Clone the repository and navigate to the project directory.
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/Mac)
4. Install dependencies: `pip install -r requirements.txt`

### Cloud API Keys (OpenRouter)
To use the OpenRouter fallback, add an API key to the top-level directory (or configure Streamlit Cloud secrets during deployment):
Create a `secret.txt` file and insert:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Run the App
```bash
streamlit run app.py
```
