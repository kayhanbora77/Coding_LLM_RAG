import os

# Set User Agent FIRST to avoid warnings
os.environ["USER_AGENT"] = "RAG-Learning-Bot/1.0"

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from bs4.filter import SoupStrainer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def main():
    """Main function to execute the RAG pipeline."""
    # Setup environment
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

    print("Loading and processing documents...")
    splits = process_documents(WEB_PATHS, CSS_CLASSES, CHUNK_SIZE, CHUNK_OVERLAP)

    print("Creating index...")
    vectorstore, retriever = create_index(splits, EMBEDDING_MODEL_NAME, RETRIEVAL_K)

    examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel's was born in what country?",
        "output": "what is Jan Sindel's personal history?",
    },
    ]
    # We now transform these to example messages
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            # Few shot examples
            few_shot_prompt,
            # New question
            ("user", "{question}"),
        ]
    )

    generate_queries_step_back = prompt | ChatGroq(model=LLM_MODEL_NAME, temperature=0) | StrOutputParser()
    question = "What is task decomposition for LLM agents?"
    generate_queries_step_back.invoke({"question": question})

    # Response prompt
    response_prompt_template = """You are an expert of world knowledge.
    I am going to ask you a question. Your response should be comprehensive and
    not contradicted with the following context if they are relevant.
    Otherwise, ignore them if they are not relevant.

    # {normal_context}
    # {step_back_context}

    # Original Question: {question}
    # Answer:"""
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
    chain = (
        {
            # Retrieve context using the normal question
            "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
            # Retrieve context using the step-back question
            "step_back_context": generate_queries_step_back | retriever,
            # Pass on the question
            "question": lambda x: x["question"],
        }
        | response_prompt
        | ChatGroq(model=LLM_MODEL_NAME, temperature=0)
        | StrOutputParser()
    )

    chain.invoke({"question": question})


if __name__ == "__main__":
    main()