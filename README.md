# 🤖 PDF AI Assistant

A multilingual PDF processing application that leverages various AI models to analyze, summarize, and interact with PDF documents. Built with Python, Gradio, and LangChain.

## 🌟 Features

- **Multiple AI Models Support**:

  - OpenAI GPT-4
  - IBM Granite 3.1
  - Mistral Small 24B
  - SmolLM2 1.7B
  - Local Ollama models

- **Multilingual Interface**:

  - English
  - Español
  - Deutsch
  - Français
  - Português

- **Core Functionalities**:
  - 📝 Text extraction from PDFs
  - 💬 Interactive Q&A with document content
  - 📋 Document summarization
  - 👨‍💼 Customizable specialist advisor
  - 🔄 Dynamic chunk size and overlap settings

## 🛠️ Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd pdf-ai-assistant
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
# Create .env file
touch .env

# Add your API keys (if using)
WATSONX_APIKEY=your_watsonx_api_key
WATSONX_PROJECT_ID=your_watsonx_project_id
```

## 📦 Dependencies

- gradio
- langchain
- chromadb
- PyPDF2
- ollama (for local models)
- python-dotenv
- requests
- ibm-watsonx-ai

## 🚀 Usage

1. Start the application:

```bash
python app.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:7860)

3. Select your preferred:

   - Language
   - AI Model
   - Model Type (Local/API)

4. Upload a PDF file and process it

5. Use any of the three main features:
   - Ask questions about the document
   - Generate a comprehensive summary
   - Get specialized analysis using the custom advisor

## 💡 Features in Detail

### Q&A System

- Interactive chat interface
- Context-aware responses
- Source page references

### Summarization

- Chunk-based processing
- Configurable chunk sizes
- Comprehensive document overview

### Specialist Advisor

- Customizable expert roles
- Detailed analysis based on expertise
- Structured insights and recommendations

## 🔧 Configuration

The application supports various AI models:

- Local models via Ollama
- API-based models (OpenAI, IBM WatsonX)
- Hugging Face models

For Ollama local models, ensure:

```bash
ollama pull granite3.1-dense
ollama pull granite-embedding:278m
```

## 🌐 Language Support

The interface and AI responses are available in:

- English
- Spanish
- German
- French
- Portuguese

## 📝 License

[MIT License]

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
