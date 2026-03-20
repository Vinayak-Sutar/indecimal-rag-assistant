import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------------------
# 1. LLM and Generation Logic
# -------------------------------------------------------------------


def instantiate_llm(model_choice, local_model_name="mistral"):
    """
    Initializes and returns the language model based on the user's selection.
    """
    if model_choice == "Local (Ollama)":
        return Ollama(model=local_model_name, temperature=0.3)

    # OpenRouter Logic
    openrouter_api_key = None
    try:
        if os.path.exists("./secret.txt"):
            with open("./secret.txt", "r") as f:
                content = f.read().strip()
                if "=" in content:
                    openrouter_api_key = content.split("=")[1].strip()
        # Fallback to Streamlit SECRETS for cloud deployment
        elif hasattr(st, "secrets") and "OPENROUTER_API_KEY" in st.secrets:
            openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass

    if openrouter_api_key:
        os.environ["OPENAI_API_KEY"] = openrouter_api_key
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0.3,
            default_headers={
                "HTTP-Referer": "https://github.com/vinayak/indecimal",
                "X-Title": "Indecimal Assistant",
            }
        )
    return None


# Prompt template enforcing Anti-Hallucination but allowing Conversation History
PROMPT_TEMPLATE = """You are a strictly objective and highly restricted AI assistant.

CRITICAL INSTRUCTIONS:
1. You are strictly forbidden from using any outside knowledge, pre-training data, or general knowledge.
2. Your ONLY sources of facts are the provided 'Context' and 'Previous Chat History'.
3. If the user asks a factual question, you must verify that the answer is explicitly contained in the provided Context.
4. If the exact answer to the user's question cannot be found in the Context below, you MUST reply with EXACTLY this string: "I cannot answer this question because the information is not present in the provided documents." Do not output anything else.
5. Do not guess, do not infer, do not extrapolate, and do not invent information if the direct answer is missing.

Previous Chat History:
{chat_history}

Context: 
{context}

Question: {question}
Answer:"""


def build_rag_chain(vectorstore, llm, chat_history_str, top_k=3):
    """
    Constructs the LangChain Expression Language (LCEL) chain.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    qa_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: chat_history_str
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


def generate_answer(prompt, vectorstore, llm, st_messages, top_k=3):
    """
    Parses chat history and triggers the RAG generation.
    Returns the final answer string and the source chunks used.
    """
    chat_history_str = ""
    # Exclude the current prompt that was just appended to the state
    for msg in st_messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        chat_history_str += f"{role}: {msg['content']}\n"

    rag_chain, retriever = build_rag_chain(
        vectorstore, llm, chat_history_str, top_k=top_k)

    # Retrieve docs for UI transparency, and run generation
    source_documents = retriever.invoke(prompt)
    try:
        answer = rag_chain.invoke(prompt)
    except Exception as e:
        source_documents = []  # Hide retrieved documents on error
        error_msg = str(e).lower()
        if "429" in error_msg or "rate limit" in error_msg:
            answer = "Looks like the free tier rate limit has been reached. Please try again later."
        else:
            answer = f"An unexpected error occurred: {str(e)}"

    return answer, source_documents

# -------------------------------------------------------------------
# 2. Vector DB and Embedding Logic
# -------------------------------------------------------------------


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )


def get_embeddings_model():
    # Free, local CPU embeddings model
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_and_embed_defaults(docs_dir='./docs/'):
    """
    Loads all markdown files from the defaults docs folder.
    """
    loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    if not documents:
        return None, [], 0

    chunks = get_text_splitter().split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, get_embeddings_model())
    file_list = list(
        set([doc.metadata.get('source', 'Unknown') for doc in documents]))

    return vectorstore, file_list, len(chunks)


def process_uploaded_documents(uploaded_files, temp_dir):
    """
    Parses dynamic uploads (PDF/TXT/MD), creates a temporary FAISS index.
    """
    new_documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            new_documents.extend(loader.load())
        else:
            loader = TextLoader(file_path)
            new_documents.extend(loader.load())

    if new_documents:
        chunks = get_text_splitter().split_documents(new_documents)
        temp_vectorstore = FAISS.from_documents(chunks, get_embeddings_model())
        added_files = [f.name for f in uploaded_files]
        return temp_vectorstore, added_files, len(chunks)

    return None, [], 0
