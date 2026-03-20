# Construction Marketplace AI Assistant (Mini-RAG)

This repository contains a Retrieval-Augmented Generation (RAG) pipeline built for **INDECIMAL**, a construction marketplace. It acts as an internal knowledge base assistant, allowing users to ask questions about company policies and specifications. To ensure reliability, the pipeline restricts answers exclusively to the provided internal documentation, strongly preventing AI hallucinations.

## 1. Setup & Local Installation

### Prerequisites
* **Python 3.10+** (Tested on Python 3.10)
* **[Ollama](https://ollama.com/)** (Optional, only required if you want to run offline local inference)
* An **Nvidia GPU** is highly recommended for running local models, though it is not strictly required.

### Installation Instructions
1. Clone the repository and navigate to the project directory.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Optional: If you intend to run the local Offline LLM, install Ollama and pull the mistral model:
   ```bash
   ollama pull mistral
   ```

### Connecting to Cloud OpenRouter
To use the high-speed cloud fallback, create a `.env` file in the root directory and add your OpenRouter API key:
```env
OPENROUTER_API_KEY=sk-or-your-key-here
```

### Running the System
Launch the interactive Streamlit user interface natively by running:
```bash
streamlit run app.py
```
This will start a local server at `http://localhost:8501`.

---

## 2. Architecture & Design Decisions

### Document Chunking & Search
* **Chunking Strategy:** Used LangChain's `RecursiveCharacterTextSplitter`. The text is split into chunks of `1000` characters with a `200` character overlap to maintain context between paragraphs.
* **Embeddings Model:** HuggingFace's open-source `sentence-transformers/all-MiniLM-L6-v2`. It runs 100% locally on the CPU, offering a great balance between low latency and accurate vector representation without paid APIs.
* **Vector Store & Retrieval:** Selected **FAISS** (Facebook AI Similarity Search) as the localized, in-memory vector database instead of cloud alternatives like Pinecone. The retriever utilizes a dynamic `top_k` cosine-similarity search.

### LLM Integration & Strict Grounding
* **Prompt Engineering (Anti-Hallucination):** The core intelligence relies on LangChain Expression Language (LCEL). To rigidly bind the generation solely to the retrieved context, implemented an exclusionary system prompt: 
  > *"If the answer is not in the provided Context, say EXACTLY: 'I cannot answer this question because the information is not present in the provided documents.'"*
* **Transparency:** Within the Streamlit User Interface, I added an expandable "Retrieved Documents" section. This acts as an explainability anchor—every time the AI replies, it prints out the precise raw chunks it parsed from FAISS.
