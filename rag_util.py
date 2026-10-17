import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# loading the environment variables
load_dotenv()

working_directory = os.path.dirname(os.path.abspath((__file__)))

# load the embedding model
embedding = HuggingFaceEmbeddings()

# load the llm
llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0.0
)

def process_document_to_croma_db(filename, persist_directory=None):
    # load the pdf document using unstructred document loader
    loader = UnstructuredPDFLoader(filename)
    documents = loader.load()

    # splitting the text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 200
    )

    texts = text_splitter.split_documents(documents)

    # store the documents into croma db
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=persist_directory or f"{working_directory}/doc_vectorstore"
    )
    return 0

def answer_question(question, persist_directory=None):
    # loading the chromadb
    vector_db = Chroma(
        persist_directory=persist_directory or f"{working_directory}/doc_vectorstore",
        embedding_function=embedding
    )

    # create a retriever
    retriever = vector_db.as_retriever(
        search_kwargs = {"k" : 5}
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer the question using only the following context. "
            "If the context does not contain the answer, say that you could not find it in the document.\n\n"
            "<context>\n{context}\n</context>",
        ),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)

    # create retrival QA chain
    qa_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    response = qa_chain.invoke({"input" : question})
    return response["answer"]


# Backwards-friendly spelling for callers that prefer the shorter function name.
process_document = process_document_to_croma_db


def remove_document_store(persist_directory):
    """Remove a session's persisted vector store after closing its Chroma handle."""
    shutil.rmtree(persist_directory, ignore_errors=True)
