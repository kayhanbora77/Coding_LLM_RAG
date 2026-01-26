# RAG Learning Bot v5

An advanced Retrieval-Augmented Generation (RAG) system that implements Task Decomposition techniques for improved question-answering capabilities using LangChain, Groq, and Hugging Face.

## Overview

This RAG system retrieves information from web documents and uses a Large Language Model to answer complex questions through task decomposition. The system breaks down complex questions into sub-questions, processes each individually, and synthesizes the results for comprehensive answers.

## Features

- **Web Document Loading**: Fetches content from specified URLs using WebBaseLoader
- **Intelligent Chunking**: Splits documents into manageable chunks for processing
- **Vector Storage**: Uses Chroma for efficient similarity search
- **Task Decomposition**: Breaks complex questions into sub-questions for better processing
- **Sub-question Processing**: Individual processing of each sub-question with RAG
- **Synthesis**: Combines answers from sub-questions into a comprehensive final answer
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
   cd llm_rag_v5
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Or install the required packages individually:
   ```bash
   pip install langchain-huggingface langchain-community langchain-core langchain-groq langchain-text-splitters bs4 chromadb tiktoken
   ```

## Configuration

Before running the application, you need to configure your API keys in the `llm_rag_v5.py` file:

1. Replace `GROQ_API_KEY` with your actual Groq API key
2. Replace `HUGGINGFACEHUB_API_TOKEN` with your actual Hugging Face token

## Usage

Run the script directly:
```bash
python3 llm_rag_v5.py
```

The system will:
1. Load documents from the configured web sources
2. Split documents into chunks
3. Create a vector store for similarity search
4. Decompose the main question into sub-questions
5. Process each sub-question individually with RAG
6. Synthesize answers from sub-questions into a final comprehensive answer

## Architecture

The system is composed of several key components:

- **Document Loader**: Fetches content from web sources using BeautifulSoup filtering
- **Text Splitter**: Breaks documents into smaller chunks for vector storage
- **Embedding Model**: Converts text to vectors using Hugging Face embeddings
- **Vector Store**: Chroma database for efficient similarity search
- **Question Decomposer**: Generates sub-questions from complex questions
- **Sub-question Processor**: Processes each sub-question with RAG
- **Synthesizer**: Combines sub-question answers into final response
- **LLM Chain**: Processes questions and generates answers

## Advanced Technique

### Task Decomposition
The system implements task decomposition by:
1. Taking a complex question and breaking it into multiple sub-questions
2. Processing each sub-question individually against the document corpus
3. Collecting answers for each sub-question
4. Synthesizing all sub-answers into a comprehensive final response

This approach allows for more thorough exploration of the document corpus and more comprehensive answers to complex questions.

## Customization

You can customize the following aspects:

- **Data Sources**: Modify `WEB_PATHS` to load content from different URLs
- **Content Filtering**: Update `CSS_CLASSES` to target specific HTML elements
- **Chunking Strategy**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` parameters
- **Models**: Change `EMBEDDING_MODEL_NAME` and `LLM_MODEL_NAME` to use different models
- **Retrieval Settings**: Modify `RETRIEVAL_K` to control the number of documents retrieved
- **Decomposition Logic**: Adjust the template in `template_decomposition` to change how questions are broken down

## Dependencies

- `langchain-huggingface`: For Hugging Face embeddings
- `langchain-community`: Community components for LangChain
- `langchain-core`: Core LangChain components
- `langchain-groq`: Integration with Groq's LLM API
- `bs4`: Web scraping with BeautifulSoup
- `chromadb`: Vector database
- `langchain-text-splitters`: Text chunking utilities
- `tiktoken`: Tokenization utilities

## How It Works

1. **Indexing Phase**:
   - Load web documents
   - Split into chunks
   - Create embeddings
   - Store in vector database

2. **Question Processing Phase**:
   - Decompose main question into sub-questions
   - Process each sub-question with RAG
   - Retrieve relevant documents for each sub-question
   - Generate answers for each sub-question
   - Synthesize all answers into final response

## Example Output

The system will output the decomposed sub-questions, individual answers to each, and a synthesized final answer to the complex question "What are the main components of an LLM-powered autonomous agent system?"

## Troubleshooting

- Ensure your API keys are valid and have sufficient quota
- Check that all dependencies are installed correctly
- Verify that the target websites are accessible
- Make sure the CSS selectors match the target website structure

## License

This project is open-source and available under the MIT License.
