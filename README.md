# CoDeforge AlcheMist

![CoDeforge AlcheMist Logo](https://img.shields.io/badge/CoDeforge-AlcheMist-blue?style=for-the-badge)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.10+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.0.267+-green.svg)](https://github.com/hwchase17/langchain)

## 📋 Overview

CoDeforge AlcheMist is an advanced document processing application that transforms technical questions from PDF documents into executable C++ code solutions. Leveraging state-of-the-art language models (Gemma2-9b and LLama3-70b), this application extracts technical questions, generates corresponding code solutions, and compiles them into a downloadable PDF report complete with visual output representations.

## 🌟 Key Features

- **PDF Document Processing**: Upload and process technical PDFs to extract relevant questions
- **Dual-Model Architecture**: 
  - **Model A (Gemma2-9b-it)**: Transforms technical questions into clear, structured language
  - **Model M (LLama3-70b-8192)**: Generates complete, machine-readable C++ code solutions
- **Comprehensive PDF Reports**: Auto-generated reports with:
  - Formatted code solutions
  - Visual output representations
  - Execution success indicators
- **Smart API Key Management**: Load-balanced API key selection for reliable performance
- **Streamlit Web Interface**: Clean, responsive user interface for seamless interaction

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/CoDeforge-AlcheMist.git
cd CoDeforge-AlcheMist
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `config.json` file in the root directory with your API keys:
```json
{
  "GROQ_API_KEY": {
    "key1": "your-groq-api-key-1",
    "key2": "your-groq-api-key-2"
  },
  "GOOGLE_API_KEY": "your-google-api-key"
}
```

## 📚 Prerequisites

- Python 3.7+
- Libraries (will be installed via requirements.txt):
  - streamlit
  - langchain
  - langchain-groq
  - langchain-google-genai
  - langchain-community
  - fpdf
  - pillow
  - faiss-cpu

## 🚀 Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload a PDF document containing technical questions

4. The application will:
   - Process the document
   - Extract technical questions
   - Generate C++ code solutions
   - Create a downloadable PDF report

## 💻 How It Works

### Data Flow Architecture

1. **Document Ingestion Pipeline**:
   - PDF uploaded → PyPDFLoader → Document splitting (RecursiveCharacterTextSplitter)
   - Context creation from document chunks

2. **Question Processing Pipeline**:
   - Context → Gemma2-9b-it (Model A) → Structured questions
   - Vector embedding with Google's Embedding model
   - FAISS vector storage for efficient retrieval

3. **Code Generation Pipeline**:
   - Structured questions → LLama3-70b (Model M) → Complete C++ solutions
   - Solution extraction with custom parsers
   - Output visualization generation

4. **PDF Generation Pipeline**:
   - Questions + Solutions → FPDF document creation
   - Code formatting and syntax highlighting
   - Output visualization embedding

### Key Components

#### Model Selection and Implementation
```python
model_a = "gemma2-9b-it"  # Fixed model for question processing
models_for_model_m = ["llama3-70b-8192"]  # Models for code generation

llm_a = ChatGroq(groq_api_key=selected_groq_api_key, model_name=get_model_a())
llm_m = ChatGroq(groq_api_key=selected_groq_api_key, model_name=selected_model_m)
```

#### Prompt Engineering
The application uses carefully crafted prompts to guide the language models:

**Model A Prompt (Question Processing)**:
```
You are an expert in transforming technical questions into clear, 
structured, and easily understandable language.

**Format:**
```
**Question Structure Overview**:
q1
q2
q3

**Updated Question 1**: [Rephrased version]
```
```

**Model M Prompt (Code Generation)**:
```
You are an expert in generating complete, machine-readable C++ code 
for the provided questions.

**Response Format:**
```
**Question 1 - Code:**
```cpp
[Full code solution]
```

**Question 1 - Output:**
```
[Expected output of the code]
```
```
```

## 📊 Performance Metrics

The application measures and displays performance metrics for transparency:
- Processing time for PDF question extraction
- Code generation time
- Model information for traceability

## 🔒 Security Considerations

- API keys are stored in a separate config file (not committed to version control)
- Random selection of API keys to prevent rate limiting
- Local processing of PDF files for data privacy

## 🧩 Extensibility

The modular design allows for easy extension:
- Additional language models can be added to the model roster
- Support for other programming languages can be implemented
- Custom prompt templates can be modified for different document types

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Groq](https://groq.com/) for providing the API for language model inference
- [Google Generative AI](https://ai.google.dev/) for embedding models
- [LangChain](https://github.com/hwchase17/langchain) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web interface framework
