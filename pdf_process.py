#Main file

#importing libraries
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI



# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



#checking the API Key
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment variables.")



# Streamlit UI
st.set_page_config("Intelli-PDF")
st.header("Here is Intelli-PDF using Gemini (LangChain Framework)")
st.subheader("Upload any pdf of choice and click submit.Then ask any question IntelliPDF will give you answer.")




#extracting text from the uploaded pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if hasattr(pdf, 'name') and hasattr(pdf, 'read'):  # Streamlit uploader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
        else:
            loader = PyPDFLoader(pdf)  # File path provided directly

        docs = loader.load()
        text += "\n".join([doc.page_content for doc in docs])

        # Clean up temp file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return text



#breaking the text into chunks

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)



#creating vector database(FAISS) and storing the embedded text
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



#system prompt
def get_QA_chain():
    prompt_template = """
    You are a helpful PDF assistant. You answer all questions related to the content of the uploaded PDF files clearly and accurately.
    
    If the uploaded file is a medical report, include relevant remarks, suggestions, results, and provide a likely diagnosis or treatment plan,
    considering the patient's age if possible. You may suggest medications or further tests when appropriate.
    Try to be precise and to the point.
    Always cite relevant parts of the document or provide links if applicable.

    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain





#user prompt
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_QA_chain()
    response = chain({
        "input_documents": docs,
        "question": user_question
    }, return_only_outputs=True)

    st.write("Reply:", response["output_text"])





#main function
def main():
    user_question = st.text_input("Ask any question related to the uploaded PDF(s):")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload multiple PDF files", accept_multiple_files=True)
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed and indexed!")






if __name__ == "__main__":
    main()
