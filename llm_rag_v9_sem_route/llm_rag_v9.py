import os
import numpy as np
from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field
from bs4.filter import SoupStrainer

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# ==========================================
# ENV SETUP
# ==========================================
os.environ["USER_AGENT"] = "RAG-Learning-Bot/1.0"
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "llama-3.1-8b-instant"


# ==========================================
# UTILS
# ==========================================
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return (a @ b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b))


def build_prompt_router(embeddings, prompt_templates, prompt_embeddings):
    def prompt_router(input):
        query_embedding = embeddings.embed_query(input["query"])
        similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
        most_similar = prompt_templates[int(similarity.argmax())]

        print("Using MATH" if "mathematician" in most_similar else "Using PHYSICS")

        return PromptTemplate.from_template(most_similar)

    return prompt_router


def main():
    physics_template = """You are a very smart physics professor.
    You are great at answering questions about physics in a concise and easy to understand manner.
    When you don't know the answer, you admit it.

    Question:
    {query}
    """

    math_template = """You are a very good mathematician.
    You break down hard problems into components and solve them step by step.

    Question:
    {query}
    """

    prompt_templates = [physics_template, math_template]

    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_NAME
    )

    prompt_embeddings = embeddings.embed_documents(prompt_templates)

    router = build_prompt_router(
        embeddings,
        prompt_templates,
        prompt_embeddings
    )

    chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(router)
        | ChatGroq(model=LLM_MODEL_NAME)
        | StrOutputParser()
    )

    print(chain.invoke("What's a black hole?"))


if __name__ == "__main__":
    main()
