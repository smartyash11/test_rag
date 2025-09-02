# ==============================================================================
# 1. IMPORTS AND SETUP
# ==============================================================================
import streamlit as st
import os
import time
from dotenv import load_dotenv

# Pinecone + Cohere
from pinecone import Pinecone
import cohere

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# ==============================================================================
# 2. API KEY AND ENVIRONMENT CONFIGURATION
# ==============================================================================

load_dotenv()

st.sidebar.title("🛠️ Configuration")
st.sidebar.markdown("Enter your API keys and Pinecone Host URL below.")

def get_config_value(service_name, env_var):
    """Get config from Streamlit sidebar or environment variables."""
    value = st.sidebar.text_input(
        f"{service_name}",
        type="password" if "KEY" in env_var else "default",
        help=f"Get this from your {service_name.split(' ')[0]} dashboard."
    )
    if value:
        return value
    return os.getenv(env_var)

OPENAI_API_KEY = get_config_value("OpenAI API Key", "OPENAI_API_KEY")
PINECONE_API_KEY = get_config_value("Pinecone API Key", "PINECONE_API_KEY")
COHERE_API_KEY = get_config_value("Cohere API Key", "COHERE_API_KEY")
PINECONE_HOST = get_config_value("Pinecone Host URL", "PINECONE_HOST")

INDEX_NAME = "minirag1"

# ==============================================================================
# 3. GLOBAL INITIALIZATIONS
# ==============================================================================

embeddings = None
llm = None
co = None
index = None
keys_provided = OPENAI_API_KEY and PINECONE_API_KEY and COHERE_API_KEY and PINECONE_HOST

if keys_provided:
    try:
        # a. Embedding Model
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )

        # b. Chat Model
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY
        )

        # c. Cohere Reranker
        co = cohere.Client(COHERE_API_KEY)

        # d. Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME, host=PINECONE_HOST)

        st.sidebar.success("✅ Successfully connected to all services!")

    except Exception as e:
        st.error(f"❌ Initialization error: {e}")
        keys_provided = False
else:
    st.warning("⚠️ Please provide all required API keys and the Pinecone Host URL in the sidebar.")

# ==============================================================================
# 4. CORE RAG PIPELINE FUNCTIONS
# ==============================================================================

def process_and_embed(text, source_title):
    """Splits text, embeds chunks, and upserts into Pinecone index."""
    if not keys_provided:
        st.error("Cannot process text. API keys are missing or invalid.")
        return

    with st.spinner("Chunking text, creating embeddings, and storing in vector DB..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        documents = []
        for i, chunk in enumerate(chunks):
            embedding_vector = embeddings.embed_documents([chunk])[0]
            documents.append({
                "id": f"{source_title}-{i}",
                "values": embedding_vector,
                "metadata": {
                    "source": source_title,
                    "text": chunk,
                    "section": "main",
                    "position": i
                }
            })
        # Upsert in batches
        for i in range(0, len(documents), 100):
            batch = documents[i:i+100]
            index.upsert(vectors=batch)

def retrieve_and_rerank(query, top_k=10):
    """Retrieve documents from Pinecone and rerank them with Cohere."""
    if not keys_provided:
        return []

    query_embedding = embeddings.embed_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    retrieved_docs = results.get("matches", [])
    retrieved_texts = [doc["metadata"]["text"] for doc in retrieved_docs]

    if not retrieved_texts:
        return []

    reranked_results = co.rerank(
        query=query,
        documents=retrieved_texts,
        top_n=3,
        model="rerank-english-v2.0"
    )

    final_docs = []
    for result in reranked_results.results:
        original_doc = retrieved_docs[result.index]
        final_docs.append({
            "text": result.document.text,
            "score": result.relevance_score,
            "source": original_doc["metadata"]["source"],
            "position": original_doc["metadata"]["position"]
        })
    return final_docs

def generate_answer(query, reranked_docs):
    """Generate a grounded answer using LangChain + OpenAI LLM."""
    if not keys_provided:
        return "", []

    context = ""
    sources_for_display = []
    for i, doc in enumerate(reranked_docs):
        context += f"Source [{i+1}]: {doc['text']}\n\n"
        sources_for_display.append({
            "citation": f"[{i+1}]",
            "source": doc["source"],
            "text": doc["text"],
            "score": doc["score"]
        })

    prompt_template = ChatPromptTemplate.from_template("""
    You are an AI assistant. Answer the user's question based ONLY on the provided context.
    - Your answer must be grounded in the information from the sources.
    - For each sentence, cite the source using the format [index].
    - If the context doesn't contain the answer, state: "I cannot answer this question based on the provided context."

    Context:
    {context}

    Question: {question}
    Answer:
    """)

    chain = prompt_template | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    return answer, sources_for_display

# ==============================================================================
# 5. STREAMLIT FRONTEND INTERFACE
# ==============================================================================

st.title("📄 Mini RAG: AI Engineer Assessment")
st.markdown("Upload a text file, embed it into Pinecone, and then ask grounded questions using RAG.")

if "processed" not in st.session_state:
    st.session_state.processed = False
if "file_name" not in st.session_state:
    st.session_state.file_name = ""

st.subheader("1. Provide Text Content")
input_method = st.radio("Choose input method:", ("Upload a .txt file", "Paste text"))
text_input, source_title = "", "pasted_text"

if input_method == "Upload a .txt file":
    uploaded_file = st.file_uploader("Choose a file", type="txt")
    if uploaded_file:
        text_input = uploaded_file.read().decode("utf-8")
        source_title = uploaded_file.name
else:
    text_input = st.text_area("Paste your text here", height=200)

if st.button("Process and Embed"):
    if not keys_provided:
        st.error("Please provide all API keys and the Host URL in the sidebar first!")
    elif text_input:
        process_and_embed(text_input, source_title)
        st.session_state.processed = True
        st.session_state.file_name = source_title
        st.success(f"✅ Successfully processed and embedded content from '{source_title}'. You can now ask questions.")
    else:
        st.warning("⚠️ Please upload a file or paste text before processing.")

if st.session_state.processed:
    st.subheader(f"2. Ask a Question About '{st.session_state.file_name}'")
    user_query = st.text_input("Enter your question:", key="query_box")

    if st.button("Get Answer"):
        if not user_query:
            st.warning("⚠️ Please enter a question.")
        elif not keys_provided:
            st.error("❌ API keys are missing. Please enter them in the sidebar.")
        else:
            start_time = time.time()
            with st.spinner("Retrieving, reranking, and generating answer..."):
                reranked_docs = retrieve_and_rerank(user_query)
                if reranked_docs:
                    answer, sources = generate_answer(user_query, reranked_docs)
                    end_time = time.time()

                    st.write("### 🧠 Answer")
                    st.markdown(answer)

                    st.write("### 📚 Sources & Citations")
                    for src in sources:
                        with st.expander(f"**{src['citation']} Source:** `{src['source']}` (Relevance Score: {src['score']:.2f})"):
                            st.write(src["text"])

                    st.sidebar.markdown("---")
                    st.sidebar.subheader("⚡ Performance Metrics")
                    st.sidebar.info(f"**Request Timing:** {end_time - start_time:.2f} seconds")
                else:
                    st.error("❌ No relevant information found. Try rephrasing your question.")
