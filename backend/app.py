import os
import tempfile

import streamlit as st
import os

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.tools import tool

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from langchain_tavily import TavilySearch


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG PDF Research Agent",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 RAG PDF Research Agent")

st.write(
    "Upload a PDF and ask questions about it. "
    "The AI can also search the web for current information."
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "agent" not in st.session_state:
    st.session_state.agent = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# API KEYS
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

if not TAVILY_API_KEY:
    try:
        TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    except Exception:
        pass

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is missing.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY is missing.")
    st.stop()


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


embeddings = get_embeddings()


# ============================================================
# GEMINI MODEL
# ============================================================

@st.cache_resource
def get_model():

    return init_chat_model(
        "google_genai:gemini-3.1-flash-lite",
        api_key=GEMINI_API_KEY,
        temperature=0
    )


model = get_model()


# ============================================================
# TAVILY WEB SEARCH
# ============================================================

@st.cache_resource
def get_web_search():

    return TavilySearch(
        max_results=3,
        search_depth="advanced",
        tavily_api_key=TAVILY_API_KEY
    )


web_search_tool = get_web_search()


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
    help="Upload a PDF document to create a searchable knowledge base."
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    # Process only when a new PDF is uploaded
    if st.session_state.pdf_name != uploaded_file.name:

        with st.spinner("📄 Processing PDF..."):

            try:

                # ------------------------------------------------
                # Save PDF temporarily
                # ------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:

                    tmp_file.write(
                        uploaded_file.getvalue()
                    )

                    pdf_path = tmp_file.name


                # ------------------------------------------------
                # Read PDF
                # ------------------------------------------------

                reader = PdfReader(pdf_path)

                documents = []

                for page_number, page in enumerate(reader.pages):

                    text = page.extract_text()

                    if text and text.strip():

                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": uploaded_file.name,
                                    "page": page_number + 1
                                }
                            )
                        )


                if not documents:

                    st.error(
                        "Could not extract text from this PDF."
                    )

                    os.remove(pdf_path)

                    st.stop()


                # ------------------------------------------------
                # Split PDF into chunks
                # ------------------------------------------------

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                all_splits = text_splitter.split_documents(
                    documents
                )


                # ------------------------------------------------
                # Create unique Chroma collection
                # ------------------------------------------------

                collection_name = (
                    "pdf_"
                    + uploaded_file.name
                    .replace(" ", "_")
                    .replace(".", "_")
                )


                # ------------------------------------------------
                # Create vector store
                # ------------------------------------------------

                vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings
                )


                # ------------------------------------------------
                # Add chunks
                # ------------------------------------------------

                vector_store.add_documents(
                    documents=all_splits
                )


                # ------------------------------------------------
                # Save vector store in session
                # ------------------------------------------------

                st.session_state.vector_store = vector_store

                st.session_state.pdf_name = uploaded_file.name

                st.session_state.num_chunks = len(all_splits)

                st.session_state.messages = []


                # ====================================================
                # PDF RETRIEVAL TOOL
                # ====================================================

                def create_pdf_retriever(vector_store):

                    @tool
                    def retrieve_from_pdf(query: str) -> str:
                        """
                        Search the uploaded PDF for information
                        relevant to the user's question.
                        """

                        retrieved_docs = (
                            vector_store.similarity_search(
                                query,
                                k=4
                            )
                        )

                        if not retrieved_docs:

                            return (
                                "No relevant information was found "
                                "in the uploaded PDF."
                            )


                        results = []

                        for doc in retrieved_docs:

                            source = doc.metadata.get(
                                "source",
                                "Uploaded PDF"
                            )

                            page = doc.metadata.get(
                                "page",
                                "Unknown"
                            )

                            results.append(
                                f"""
Source: {source}
Page: {page}

Content:
{doc.page_content}
"""
                            )


                        return "\n\n".join(results)


                    return retrieve_from_pdf


                retrieve_from_pdf = create_pdf_retriever(
                    vector_store
                )


                # ====================================================
                # SYSTEM PROMPT
                # ====================================================

                system_prompt = """
You are a professional research assistant.

You have access to TWO tools.

============================================================
TOOL 1: retrieve_from_pdf
============================================================

This tool searches the PDF uploaded by the user.

Use it when the user asks about:

- concepts explained in the PDF
- methods from the PDF
- equations from the PDF
- experiments from the PDF
- authors' claims
- results mentioned in the PDF
- any information that can be answered from the PDF

============================================================
TOOL 2: tavily_search
============================================================

This tool searches the current web.

Use it when the user asks about:

- recent information
- latest research
- current technologies
- recent developments
- current versions
- current events
- information that is not available in the PDF

============================================================
IMPORTANT TOOL SELECTION
============================================================

CASE 1:
If the user asks only about the uploaded PDF:

Use:
retrieve_from_pdf

Do NOT use web search unnecessarily.

------------------------------------------------------------

CASE 2:
If the user asks only about current/recent information:

Use:
tavily_search

------------------------------------------------------------

CASE 3:
If the user asks for a comparison between the PDF
and recent/current technology:

Use BOTH:

retrieve_from_pdf
+
tavily_search

------------------------------------------------------------

CASE 4:
If the user asks something unrelated to both
the PDF and current information:

Answer normally if possible.

============================================================
ANSWER RULES
============================================================

1. Never invent information.

2. If information comes from the PDF, mention the
   PDF page number when available.

3. If web search was used, explicitly say:

   "I also checked current web information."

4. If both tools were used, clearly separate:

   "From the PDF"
   and
   "From current web research"

5. Do not display raw tool output.

6. Do not display Python dictionaries.

7. Do not display internal tool metadata.

8. Do not display search-result signatures,
   IDs, or other internal data.

9. Give the user a clean, readable answer.

10. Use Markdown when useful.

11. For comparisons, prefer tables.

12. If the PDF does not contain the answer,
    clearly say so instead of guessing.
"""


                # ====================================================
                # CREATE AGENT
                # ====================================================

                agent = create_agent(
                    model=model,
                    tools=[
                        retrieve_from_pdf,
                        web_search_tool
                    ],
                    system_prompt=system_prompt
                )


                # Save agent

                st.session_state.agent = agent


                # ------------------------------------------------
                # Delete temporary PDF
                # ------------------------------------------------

                try:
                    os.remove(pdf_path)
                except Exception:
                    pass


                st.success(
                    f"✅ PDF processed successfully! "
                    f"{len(all_splits)} chunks created."
                )


            except Exception as e:

                st.error(
                    f"Error while processing PDF:\n\n{str(e)}"
                )

                st.stop()


# ============================================================
# PDF STATUS
# ============================================================

if st.session_state.vector_store is not None:

    st.info(
        f"""
📄 **Current PDF:** {st.session_state.pdf_name}

🧩 **Chunks:** {st.session_state.num_chunks}
"""
    )


# ============================================================
# CHAT SECTION
# ============================================================

st.divider()

st.subheader("💬 Ask questions about your PDF")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CLEAN CONTENT FUNCTION
# ============================================================

def extract_clean_text(content):
    """
    Gemini/LangChain can sometimes return content as:

    "normal string"

    OR

    [
        {"type": "text", "text": "answer"}
    ]

    This function converts both into clean text.
    """

    # -----------------------------------------
    # Case 1: Normal string
    # -----------------------------------------

    if isinstance(content, str):

        return content


    # -----------------------------------------
    # Case 2: List of content blocks
    # -----------------------------------------

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, str):

                text_parts.append(item)

            elif isinstance(item, dict):

                # Usually:
                # {"type": "text", "text": "..."}

                if item.get("type") == "text":

                    text = item.get("text")

                    if text:
                        text_parts.append(text)

                elif "text" in item:

                    text = item.get("text")

                    if text:
                        text_parts.append(text)


        return "\n".join(text_parts)


    # -----------------------------------------
    # Case 3: Dictionary
    # -----------------------------------------

    if isinstance(content, dict):

        if "text" in content:

            return str(content["text"])

        if "content" in content:

            return extract_clean_text(
                content["content"]
            )


    # -----------------------------------------
    # Fallback
    # -----------------------------------------

    return str(content)


# ============================================================
# USER QUESTION
# ============================================================

user_query = st.chat_input(
    "Ask a question about your PDF..."
)


if user_query:

    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if st.session_state.agent is None:

        st.warning(
            "⚠️ Please upload a PDF before asking questions."
        )

        st.stop()


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )


    with st.chat_message("user"):

        st.markdown(user_query)


    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🔎 Researching..."):

            try:

                response = st.session_state.agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_query
                            }
                        ]
                    }
                )


                # ====================================================
                # GET FINAL MESSAGE
                # ====================================================

                final_message = response["messages"][-1]


                # ====================================================
                # EXTRACT CLEAN ANSWER
                # ====================================================

                final_answer = extract_clean_text(
                    final_message.content
                )


                # ====================================================
                # DETECT TOOLS USED
                # ====================================================

                tools_used = []

                for message in response["messages"]:

                    if hasattr(message, "tool_calls"):

                        for tool_call in message.tool_calls:

                            tool_name = tool_call.get("name")

                            if tool_name:

                                tools_used.append(
                                    tool_name
                                )


                # Remove duplicates

                tools_used = list(
                    dict.fromkeys(tools_used)
                )


                # ====================================================
                # SHOW TOOL USAGE
                # ====================================================

                if tools_used:

                    st.caption(
                        "🔧 **Tools used:** "
                        + ", ".join(tools_used)
                    )

                else:

                    st.caption(
                        "ℹ️ No external tools were used."
                    )


                # ====================================================
                # SHOW EXPLANATION OF TOOL USAGE
                # ====================================================

                if "retrieve_from_pdf" in tools_used:

                    st.caption(
                        "📄 PDF retrieval was used."
                    )


                if "tavily_search" in tools_used:

                    st.caption(
                        "🌐 Current web search was used."
                    )


                # ====================================================
                # DISPLAY CLEAN ANSWER
                # ====================================================

                st.markdown(final_answer)


                # ====================================================
                # SAVE ANSWER
                # ====================================================

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_answer
                    }
                )


            except Exception as e:

                st.error(
                    f"""
                    ❌ Error while generating response:

                    {str(e)}
                    """
                )