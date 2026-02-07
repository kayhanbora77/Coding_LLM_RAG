# RAG Learning Bot v9: Semantic Routing

An advanced Retrieval-Augmented Generation (RAG) system implementing semantic routing for directing queries to the most appropriate processing chain based on content similarity.

## Overview

This implementation uses semantic routing where the system calculates the semantic similarity between the incoming query and predefined prompt templates to determine the most appropriate processing chain. This approach selects the best-suited prompt and processing logic based on content similarity rather than keyword matching.

## Features

- **Semantic Routing**: Calculates cosine similarity between queries and prompt templates
- **Dynamic Prompt Selection**: Automatically chooses the most relevant prompt template
- **Physics vs Math Specialization**: Demonstrates routing between different subject domains
- **Similarity-Based Decision Making**: Uses embedding similarity for routing decisions
- **Adaptive Processing Chains**: Different processing logic for different content types

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
   pip install scikit-learn
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
python llm_rag_v9.py
```

The system will:
1. Calculate embeddings for predefined prompt templates (physics and math)
2. Compare incoming query with these templates using cosine similarity
3. Select the most semantically similar prompt template
4. Process the query using the selected prompt template
5. Return the response from the appropriate processing chain

## How Semantic Routing Works

The routing system involves several phases:

1. **Template Embedding**:
   - Pre-compute embeddings for different prompt templates
   - Physics template: "You are a very smart physics professor..."
   - Math template: "You are a very good mathematician..."

2. **Query Analysis**:
   - Input question: "What's a black hole?"
   - Calculate embedding of the query
   - Compute cosine similarity with each template

3. **Similarity-Based Routing**:
   - Select template with highest similarity score
   - Route to appropriate processing chain
   - Apply domain-specific logic

## Architecture

The system implements:

- **Template Preprocessing**: Embeds all available prompt templates
- **Cosine Similarity Calculator**: Computes semantic similarity
- **Router Function**: Selects optimal processing chain
- **Domain-Specific Processing**: Different logic for different subjects

## Key Components

- **Cosine Similarity**: Uses sklearn for similarity computation
- **Embedding-Based Routing**: Leverages embeddings for semantic matching
- **Prompt Selection**: Dynamically chooses best prompt based on similarity
- **Domain Routing**: Directs queries to appropriate subject experts

## Customization

You can modify:

- **Prompt Templates**: Add new templates for different domains
- **Routing Logic**: Adjust similarity threshold or comparison method
- **Domains**: Extend to more subject areas (chemistry, biology, etc.)
- **Model Configuration**: Adjust `EMBEDDING_MODEL_NAME` and `LLM_MODEL_NAME`
- **Processing Chains**: Implement different logic for each domain

## Example Workflow

1. **Input**: "What's a black hole?"
2. **Embedding**: Calculate query embedding
3. **Similarity**: Compare with physics and math templates
4. **Routing**: Physics template has higher similarity
5. **Processing**: Route to physics expert prompt
6. **Response**: Process with physics-specific logic

## Dependencies

Same as main project plus:
- `scikit-learn`: For cosine similarity calculations
- `numpy`: For numerical computations
- `langchain-huggingface`
- `langchain-community`
- `langchain-core`
- `langchain-groq`
- `bs4`
- `chromadb`
- `langchain-text-splitters`

## Advanced Usage

Add more domains by extending the template list:
```python
# In main(), add more templates:
chemistry_template = """You are a chemistry expert..."""
biology_template = """You are a biology expert..."""
prompt_templates = [physics_template, math_template, chemistry_template, biology_template]
```

## Troubleshooting

- Ensure API keys are properly configured in `.env`
- Check that scikit-learn is installed: `pip install scikit-learn`
- Verify all dependencies are installed
- Confirm sufficient API quota for Groq requests

## Related Versions

This builds upon:
- **v8**: Logic-Based Query Routing (structured decision making)
- **v7**: HyDE (Hypothetical Document Embeddings) for improved retrieval
- **v6**: Step-Back Prompting (enhanced reasoning)

## License

This project is open-source and available under the MIT License.
