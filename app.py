import streamlit as st
import os

# Import modularized functions from our engine backend
from rag_engine import (
    instantiate_llm,
    load_and_embed_defaults,
    process_uploaded_documents,
    generate_answer
)

# --- App UI Configuration ---
st.set_page_config(
    page_title="Construction Marketplace Assistant",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Construction Marketplace AI Assistant")
st.markdown("Ask questions based on internal documents. The answers are generated strictly using the requested documents.")

# --- Session State Initialization ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "loaded_files" not in st.session_state:
    # Keeps track of which files are currently indexed in FAISS
    st.session_state.loaded_files = []

# Ensure workspace temp directory exists
temp_dir = "./temp_docs"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# --- Sidebar Widget Setup ---
with st.sidebar:
    st.header("⚙️ Settings")

    # 1. Model Configuration
    st.markdown("**Model Configuration**")
    st.info("🌐 Running Cloud Distribution (OpenRouter via Streamlit Cloud)")
    model_choice = "Cloud (OpenRouter)"

    # Initialize connection to Language Model via the engine wrapper
    llm = instantiate_llm(model_choice)

    if llm:
        st.success("API Key loaded from deployment secrets successfully!")
    else:
        st.error("No API key found in Streamlit secrets.")

    st.divider()

    # 2. Knowledge Base Status & Controls
    st.header("📄 Knowledge Base")
    st.markdown("Manage your document context here.")

    # Base Documents loader
    if st.button("Load Default Documents (`docs/`)"):
        with st.spinner("Loading and Embedding Default Documents..."):
            try:
                vectorstore, new_files, num_chunks = load_and_embed_defaults()
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.loaded_files = new_files
                    st.success(
                        f"Successfully loaded and embedded {num_chunks} chunks!")
                else:
                    st.warning("No documents found in `docs/` folder.")
            except Exception as e:
                st.error(f"Error loading documents: {e}")

    # Custom Document Uploader
    uploaded_files = st.file_uploader(
        "Upload Custom Documents (PDF/TXT/MD)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process & Add Uploaded Files"):
        with st.spinner("Processing uploaded files..."):
            temp_vectorstore, added_files, num_chunks = process_uploaded_documents(
                uploaded_files, temp_dir
            )

            if temp_vectorstore:
                # Merge into existing index, or instantiate if empty
                if st.session_state.vectorstore is not None:
                    st.session_state.vectorstore.merge_from(temp_vectorstore)
                else:
                    st.session_state.vectorstore = temp_vectorstore

                st.session_state.loaded_files.extend(added_files)
                # Deduplicate records
                st.session_state.loaded_files = list(
                    set(st.session_state.loaded_files))

                st.success(
                    f"Added {num_chunks} chunks from {len(uploaded_files)} files!")

    # 3. Dynamic Active File Preview List
    st.divider()
    st.subheader("📚 Currently Loaded Files")

    if not st.session_state.loaded_files:
        st.info("No files currently loaded in the Knowledge Base.")
    else:
        for file in st.session_state.loaded_files:
            # Render a neat expander block for transparency
            with st.expander(f"📄 {os.path.basename(file)}"):
                try:
                    # Resolve path whether it's local docs/ or dynamic temp_docs/
                    target_path = file if os.path.exists(
                        file) else os.path.join(temp_dir, file)

                    if os.path.exists(target_path):
                        if target_path.endswith('.pdf'):
                            st.caption(
                                "PDF preview not supported in this view. It is correctly embedded, however.")
                        else:
                            with open(target_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                st.text(
                                    content[:2000] + ("..." if len(content) > 2000 else ""))
                except Exception as e:
                    st.error(f"Cannot preview file: {str(e)}")

# --- Main Chat Input & Output Interface ---

# Remount previous messages on app reload
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Optionally display chunk history for visibility
        if "context" in message and message["context"]:
            with st.expander("🔍 View Retrieved Context"):
                for idx, doc in enumerate(message["context"]):
                    st.markdown(
                        f"**Chunk {idx + 1} (Source: {doc.metadata.get('source', 'Unknown')})**")
                    st.info(doc.page_content)

# Accept User Input
if prompt := st.chat_input("Ask a question (e.g., 'What factors affect construction project delays?'):"):
    # Append the question to memory
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and Stream generation
    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            st.warning("Please load documents from the sidebar first.")
        elif llm is None:
            st.error(
                "LLM is not properly configured. Check your API keys if using OpenRouter.")
        else:
            with st.spinner("Thinking..."):
                # Pass directly to our abstracted engine logic
                answer, source_documents = generate_answer(
                    prompt=prompt,
                    vectorstore=st.session_state.vectorstore,
                    llm=llm,
                    st_messages=st.session_state.messages
                )

                st.markdown(answer)

                # Show contextual chunk grounding mapping
                with st.expander("🔍 View Retrieved Context"):
                    for idx, doc in enumerate(source_documents):
                        st.markdown(
                            f"**Chunk {idx + 1} (Source: {doc.metadata.get('source', 'Unknown')})**")
                        st.info(doc.page_content)

                # Store AI response and reference context
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "context": source_documents
                })
