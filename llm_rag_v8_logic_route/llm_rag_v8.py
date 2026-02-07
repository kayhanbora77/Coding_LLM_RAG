import os
from dotenv import load_dotenv

# Set User Agent FIRST to avoid warnings
os.environ["USER_AGENT"] = "RAG-Learning-Bot/1.0"

# Load environment variables from .env file
load_dotenv()
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from bs4.filter import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# ==========================================
# CONFIGURATION CONSTANTS
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # From console.groq.com
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")  # From huggingface.co
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
RETRIEVAL_K = 4  # Increased to 4 to get better context for sub-questions
WEB_PATHS = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
CSS_CLASSES = ("post-content", "post-title", "post-header")


def process_documents(web_paths, css_classes, chunk_size, chunk_overlap):
    """Load and process documents from web sources."""
    # Load documents
    loader = WebBaseLoader(
        web_paths=web_paths,
        bs_kwargs=dict(
            parse_only=SoupStrainer(
                class_=css_classes
            )
        ),
    )
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(docs)

    return splits


def create_index(splits, embedding_model_name, k_value):
    """Create vector store and retriever from document splits."""
    # Create vector store
    embeddings = HuggingFaceEndpointEmbeddings(model=embedding_model_name)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})

    return vectorstore, retriever


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )


def choose_route(result):
    if "python_docs" in result.datasource.lower():
        ### Logic here
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        ### Logic here
        return "chain for js_docs"
    else:
        ### Logic here
        return "golang_docs"


def main():
    """Main function to execute the RAG pipeline."""
    # Setup environment
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

    print("Loading and processing documents...")
    splits = process_documents(WEB_PATHS, CSS_CLASSES, CHUNK_SIZE, CHUNK_OVERLAP)

    print("Creating index...")
    vectorstore, retriever = create_index(splits, EMBEDDING_MODEL_NAME, RETRIEVAL_K)

    # LLM with function call
    llm = ChatGroq(model=LLM_MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)

    # Prompt
    system = """You are an expert at routing a user question to the appropriate data source.

    Based on the programming language the question is referring to, route it to the relevant data source."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    # Define router
    router = prompt | structured_llm

    question = """Why doesn't the following code work:

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
    prompt.invoke("french")
    """

    result = router.invoke({"question": question})
    print(result)
    RouteQuery(datasource='python_docs')

    full_chain = router | RunnableLambda(choose_route)
    full_chain.invoke({"question": question})

if __name__ == "__main__":
    main()
