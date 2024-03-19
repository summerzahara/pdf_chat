from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()

def read_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def split_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def embed_chunks(chunk):
    embeddings = OpenAIEmbeddings()
    embeddings.embed_documents(chunk)
    db = FAISS.from_texts(chunk, embeddings)
    return db

def retrieve_data(db, query):
    retriever = db.as_retriever()
    response = retriever.get_relevant_documents(query)
    return response
