import streamlit as st
import tempfile
import os
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
from pdf_reader import read_pdf
from chunking import create_chunks
from vector_store import create_vector_store, retrieve_chunk
from gemini_service import ask_gemini


st.write("API KEY FOUND:", os.getenv("YOUR_API_KEY"))

try:
    genai.configure(api_key=os.getenv("YOUR_API_KEY"))

    test_model = genai.GenerativeModel("gemini-2.0-flash")

    if st.button("🧪 Test Gemini Connection"):
        response = test_model.generate_content("Say hello")
        st.success(response.text)

except Exception as e:
    st.error(f"Gemini Error: {e}")

# Page Config
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #A0A0A0;
    font-size: 18px;
    margin-bottom: 30px;
}

.answer-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #1E1E1E;
    color: white;
    border: 1px solid #333;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    "<div class='title'>🤖 RAG PDF AI Assistant</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Upload any PDF and ask questions instantly using Gemini + FAISS</div>",
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    st.markdown("---")

    st.markdown("""
    ### 🚀 Features

    ✅ PDF Question Answering

    ✅ FAISS Vector Search

    ✅ Gemini AI

    ✅ Semantic Retrieval

    ✅ Fast Responses
    """)

# Main Section
question = st.text_input(
    "💬 Ask a Question",
    placeholder="Example: How many joints are used in the robot arm?"
)

col1, col2, col3 = st.columns([1,1,1])

with col2:
    submit = st.button("🔍 Get Answer", use_container_width=True)

# Processing
if submit:

    if uploaded_file is None:
        st.warning("⚠️ Please upload a PDF first.")
    
    elif question.strip() == "":
        st.warning("⚠️ Please enter a question.")
    
    else:

        with st.spinner("🔍 Processing document..."):

            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            # Backend Flow
            text = read_pdf(pdf_path)

            chunks = create_chunks(text)

            index, embeddings = create_vector_store(chunks)

            context = retrieve_chunk(
                question,
                index,
                chunks
            )

            answer = ask_gemini(
                context,
                question
            )

        st.success("✅ Answer Generated")

        st.markdown("## 🤖 AI Response")

        st.markdown(
            f"""
            <div class="answer-box">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("📄 Retrieved Context"):
            st.write(context)

# Footer
st.markdown("---")

st.markdown(
    """
    <center>
    Made with ❤️ using Streamlit, FAISS and Gemini AI
    </center>
    """,
    unsafe_allow_html=True
)