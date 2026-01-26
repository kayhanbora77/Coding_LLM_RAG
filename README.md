# Coding LLM RAG Project

A comprehensive collection of Retrieval-Augmented Generation (RAG) implementations demonstrating progressive complexity and advanced techniques using LangChain, Groq, and Hugging Face.

## Overview

This project contains multiple versions of RAG systems, each building upon the previous with increasingly sophisticated techniques:

- **llm_rag_v1**: Basic RAG implementation with document loading and simple question answering
- **llm_rag_v2**: Enhanced with manual embedding similarity testing
- **llm_rag_v3**: Advanced multi-query approach for improved retrieval
- **llm_rag_v4**: Implementation of RAG-Fusion and Reciprocal Rank Fusion
- **llm_rag_v5**: Task Decomposition for handling complex questions

## Features

- **Progressive Learning**: Each version introduces new concepts and techniques
- **Multiple Approaches**: Various RAG methodologies and enhancements
- **Modular Architecture**: Clean, maintainable code structure
- **Comprehensive Documentation**: Each version includes detailed README
- **Best Practices**: Following Python and LangChain best practices

## Prerequisites

- Python 3.8 or higher
- API keys for:
  - [Groq Cloud](https://console.groq.com)
  - [Hugging Face Hub](https://huggingface.co)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Coding_LLM_RAG
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
Coding_LLM_RAG/
├── llm_rag_v1/          # Basic RAG implementation
│   ├── llm_rag_v1.py
│   └── README.md
├── llm_rag_v2/          # With manual embedding tests
│   ├── llm_rag_v2.py
│   └── README.md
├── llm_rag_v3/          # Multi-query enhancement
│   ├── llm_rag_v3.py
│   └── README.md
├── llm_rag_v4/          # RAG-Fusion and Reciprocal Rank Fusion
│   ├── llm_rag_v4.py
│   └── README.md
└── llm_rag_v5/          # Task Decomposition
    ├── llm_rag_v5.py
    ├── requirements.txt
    └── README.md
```

## Usage

Each version can be run independently:

```bash
# Navigate to the desired version directory
cd llm_rag_v1
python3 llm_rag_v1.py

# Or for other versions
cd ../llm_rag_v2
python3 llm_rag_v2.py
```

## Version Descriptions

### llm_rag_v1: Basic RAG
- Simple document loading and retrieval
- Basic question answering with context
- Foundational RAG concepts

### llm_rag_v2: With Embedding Tests
- Manual embedding similarity testing
- Cosine similarity calculations
- Enhanced with basic similarity metrics

### llm_rag_v3: Multi-Query Enhancement
- Generates multiple query perspectives
- Improved retrieval through query diversification
- Better handling of ambiguous questions

### llm_rag_v4: RAG-Fusion and RRF
- RAG-Fusion with multiple query generation
- Reciprocal Rank Fusion for result ranking
- Advanced retrieval enhancement techniques

### llm_rag_v5: Task Decomposition
- Complex question breakdown into sub-questions
- Individual processing of sub-questions
- Synthesis of answers from multiple sub-answers

## Dependencies

The project uses various LangChain components and external services:

- `langchain-huggingface`: For Hugging Face embeddings
- `langchain-community`: Community components for LangChain
- `langchain-core`: Core LangChain components
- `langchain-groq`: Integration with Groq's LLM API
- `bs4`: Web scraping with BeautifulSoup
- `chromadb`: Vector database
- `langchain-text-splitters`: Text chunking utilities
- `tiktoken`: Tokenization utilities
- `numpy`: Numerical computing for similarity calculations

## Customization

Each version can be customized:

- **Data Sources**: Modify URLs and CSS selectors for different content
- **Models**: Change embedding and LLM models
- **Chunking Strategy**: Adjust chunk sizes and overlap
- **Retrieval Settings**: Modify number of retrieved documents
- **Prompts**: Customize prompt templates for different behaviors

## Contributing

Feel free to contribute by:
- Adding new RAG techniques and approaches
- Improving existing implementations
- Adding more comprehensive examples
- Enhancing documentation

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- LangChain community for excellent RAG tools
- Hugging Face for embedding models
- Groq for fast LLM inference
