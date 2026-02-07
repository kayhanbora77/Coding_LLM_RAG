# RAG Learning Bot v6: Step-Back Prompting

An advanced Retrieval-Augmented Generation (RAG) system implementing Step-Back Prompting technique for enhanced reasoning and question answering.

## Overview

This implementation uses the Step-Back Prompting technique where the system first generates a more generic, step-back version of the question before answering. This approach helps in:
- Improving reasoning for complex questions
- Providing better context through multiple perspectives
- Generating more comprehensive answers

## Features

- **Step-Back Prompting**: Generates generic questions first for better reasoning
- **Dual Context Retrieval**: Uses both original and step-back questions for context
- **Few-Shot Learning**: Demonstrates prompt engineering with examples
- **Enhanced Answer Synthesis**: Combines multiple sources of information
- **Web Document Processing**: Loads and processes content from online sources

## Prerequisites

- Python 3.8 or higher
- API keys for:
  - [Groq Cloud](https://console.groq.com)
  - [Hugging Face Hub](https://huggingface.co)

## Installation

1. Ensure you're in the project root directory
2. Install required packages (if not already installed):
   ```bash
   pip install -r ../requirements.txt
   ```

## Configuration

Before running, configure your API keys in the `.env` file at the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

## Usage

Run the script directly:
```bash
python llm_rag_v6.py
```

The system will:
1. Load documents from the configured web source
2. Split documents into chunks and create vector store
3. Generate a step-back question for "What is task decomposition for LLM agents?"
4. Retrieve context using both original and step-back questions
5. Generate a comprehensive answer combining both contexts

## How Step-Back Prompting Works

The technique involves two phases:

1. **Step-Back Generation**:
   - Original question: "What is task decomposition for LLM agents?"
   - Generated step-back: "What is task decomposition?"

2. **Dual Retrieval**:
   - Retrieve context for original specific question
   - Retrieve context for generic step-back question

3. **Answer Synthesis**:
   - Combine both contexts
   - Generate comprehensive answer using all available information

## Architecture

The system implements:

- **Document Processing Pipeline**: Web loading → chunking → embedding → vector storage
- **Step-Back Prompt Engineering**: Few-shot examples for question generalization
- **Dual Retrieval System**: Two separate context retrieval streams
- **Context-Aware Answer Generation**: LLM that considers multiple information sources

## Key Components

- **Step-Back Prompt Template**: Generates generic questions from specific ones
- **Few-Shot Examples**: Demonstrates the step-back transformation pattern
- **Dual Retriever Chain**: Retrieves context from both question types
- **Response Synthesis**: Combines multiple contexts into coherent answers

## Customization

You can modify:

- **Target Question**: Change the `question` variable in the main function
- **Step-Back Examples**: Update the examples in the few-shot prompt
- **Data Sources**: Modify `WEB_PATHS` and `CSS_CLASSES` for different content
- **Models**: Adjust `EMBEDDING_MODEL_NAME` and `LLM_MODEL_NAME`
- **Retrieval Parameters**: Change `RETRIEVAL_K` for more/less context

## Example Workflow

1. **Input**: "What is task decomposition for LLM agents?"
2. **Step-Back**: "What is task decomposition?"
3. **Context Retrieval**:
   - Get specific context about LLM task decomposition
   - Get general context about task decomposition
4. **Answer Generation**: Synthesize comprehensive response

## Dependencies

Same as main project:
- `langchain-huggingface`
- `langchain-community`
- `langchain-core`
- `langchain-groq`
- `bs4`
- `chromadb`
- `langchain-text-splitters`
- `tiktoken`

## Advanced Usage

Try different questions to see the step-back effect:
```python
# In the main() function, change:
question = "Your complex question here"
```

## Troubleshooting

- Ensure API keys are properly configured in `.env`
- Check that the target website is accessible
- Verify all dependencies are installed
- Confirm sufficient API quota for Groq requests

## Related Versions

This builds upon:
- **v5**: Task Decomposition (handles complex multi-part questions)
- **v4**: RAG-Fusion (improved retrieval through multiple queries)
- **v3**: Multi-Query (query diversification)

## License

This project is open-source and available under the MIT License.
