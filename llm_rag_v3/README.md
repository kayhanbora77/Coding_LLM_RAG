# RAG Learning Bot v3

An advanced Retrieval-Augmented Generation (RAG) system that leverages LangChain, Groq, and Hugging Face to provide intelligent question-answering capabilities based on external documents.

## Overview

This RAG system retrieves information from web documents and uses a Large Language Model to answer questions about the content. The system implements a multi-query approach to improve retrieval quality by generating multiple perspectives of a user's question.

## Features

- **Web Document Loading**: Fetches content from specified URLs using WebBaseLoader
- **Intelligent Chunking**: Splits documents into manageable chunks for processing
- **Vector Storage**: Uses Chroma for efficient similarity search
- **Multi-Query Enhancement**: Generates multiple query variations to improve retrieval
- **Deduplication**: Removes duplicate retrieved documents
- **Modular Architecture**: Clean separation of concerns with dedicated functions

## Prerequisites

- Python 3.8 or higher
- API keys for:
  - [Groq Cloud](https://console.groq.com)
  - [Hugging Face Hub](https://huggingface.co)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd llm_rag_v3
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Or install the required packages individually:
   ```bash
   pip install langchain-huggingface langchain-community langchain-core langchain-groq langchain-text-splitters tiktoken bs4 chromadb
   ```

## Configuration

Before running the application, you need to configure your API keys in the `llm_rag_v3.py` file:

1. Replace `GROQ_API_KEY` with your actual Groq API key
2. Replace `HUGGINGFACEHUB_API_TOKEN` with your actual Hugging Face token

## Usage

Run the script directly:
```bash
python3 llm_rag_v3.py
```

The system will:
1. Load documents from the configured web sources
2. Split documents into chunks
3. Create a vector store for similarity search
4. Test the retriever with a sample question
5. Generate answers to your questions using the RAG pipeline

## Architecture

The system is composed of several key components:

- **Document Loader**: Fetches content from web sources using BeautifulSoup filtering
- **Text Splitter**: Breaks documents into smaller chunks for vector storage
- **Embedding Model**: Converts text to vectors using Hugging Face embeddings
- **Vector Store**: Chroma database for efficient similarity search
- **Query Generator**: Creates multiple query variations to improve retrieval
- **Retriever**: Fetches relevant documents based on the query
- **LLM Chain**: Combines retrieved context with user questions for answer generation

## Customization

You can customize the following aspects:

- **Data Sources**: Modify `WEB_PATHS` to load content from different URLs
- **Content Filtering**: Update `CSS_CLASSES` to target specific HTML elements
- **Chunking Strategy**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` parameters
- **Models**: Change `EMBEDDING_MODEL_NAME` and `LLM_MODEL_NAME` to use different models
- **Retrieval Settings**: Modify `RETRIEVAL_K` to control the number of documents retrieved

## Dependencies

- `langchain-huggingface`: For Hugging Face embeddings
- `langchain-community`: Community components for LangChain
- `langchain-core`: Core LangChain components
- `langchain-groq`: Integration with Groq's LLM API
- `tiktoken`: Tokenization utilities
- `bs4`: Web scraping with BeautifulSoup
- `chromadb`: Vector database
- `langchain-text-splitters`: Text chunking utilities

## How It Works

1. **Indexing Phase**:
   - Load web documents
   - Split into chunks
   - Create embeddings
   - Store in vector database

2. **Query Phase**:
   - Generate multiple query perspectives
   - Retrieve relevant documents
   - Deduplicate results
   - Combine context with question
   - Generate answer using LLM

## Example Output

The system will output information about the number of documents retrieved and the final answer to the question "What is task decomposition for LLM agents?"

## Troubleshooting

- Ensure your API keys are valid and have sufficient quota
- Check that all dependencies are installed correctly
- Verify that the target websites are accessible
- Make sure the CSS selectors match the target website structure

## License

This project is open-source and available under the MIT License.
