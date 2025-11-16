# 🏥 Medical Chatbot with RAG, Chat History & Image Analysis

An intelligent medical chatbot powered by Retrieval-Augmented Generation (RAG) that provides accurate medical information through natural language conversations. Built using LangChain, Groq, and Streamlit with support for contextual chat history and medical image analysis.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- **🤖 RAG-Based Responses**: Retrieves relevant information from medical PDF documents using FAISS vector store
- **💬 Contextual Chat History**: Maintains conversation context for intelligent follow-up questions
- **📸 Medical Image Analysis**: Analyzes medical images using BLIP (Bootstrapping Language-Image Pre-training) model
- **🔍 Semantic Search**: Uses sentence transformers for accurate document retrieval
- **⚡ Fast Response**: Powered by Groq's LLaMA 3.1 model for quick inference
- **📄 PDF Processing**: Automatically processes medical PDF documents and creates searchable knowledge base
- **🎨 User-Friendly Interface**: Clean Streamlit interface with chat history management

## 🏗️ Architecture

```
User Query → Chat History Context → Vector Store Retrieval → LLM Processing → Response
                                  ↓
                            Image Analysis (if image uploaded)
```

**Key Components:**
- **Vector Store**: FAISS with HuggingFace embeddings (all-MiniLM-L6-v2)
- **LLM**: Groq LLaMA 3.1 (8B Instant)
- **Vision Model**: Salesforce BLIP Image Captioning
- **Document Processing**: PyPDF for PDF extraction
- **Framework**: LangChain for orchestration

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API Key ([Get it here](https://console.groq.com/))
- (Optional) Hugging Face API Key for image processing fallback

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SankalpGoel/medical_chatbot.git
cd medical-chatbot
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
streamlit
langchain
langchain-huggingface
langchain-community
langchain-groq
faiss-cpu
sentence-transformers
python-dotenv
Pillow
transformers
torch
torchvision
accelerate
requests
pypdf
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_hf_api_key_here  # Optional
```

**Get your API keys:**
- **Groq API**: Visit [https://console.groq.com/](https://console.groq.com/) → Sign up → Generate API key
- **Hugging Face** (optional): Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 5. Prepare Your Medical Documents

Add your medical PDF documents to the `data/` folder:

```bash
# The data folder should contain your PDF files
data/
└── The_GALE_ENCYCLOPEDIA_of_MEDICINE_5.pdf  # Example
```

### 6. Create Vector Store

Run the memory creation script to process PDFs and build the vector database:

```bash
python create_memory_for_llm.py
```

This will:
1. Load all PDF files from the `data/` directory
2. Split documents into chunks (500 chars with 50 char overlap)
3. Create embeddings using sentence-transformers
4. Store embeddings in FAISS vector database at `vectorstore/db_faiss/`

**Output:**
```
✅ Vector store created at: vectorstore/db_faiss/
✅ Processed [X] documents
✅ Created [Y] text chunks
```

## 🎮 Usage

### Start the Application

```bash
streamlit run medibot.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Chatbot

#### 💬 Text-Based Queries
1. Type your medical question in the chat input
2. Press Enter to get a response based on your PDF knowledge base
3. Ask follow-up questions - the bot remembers context!

**Example:**
```
You: What is diabetes?
Bot: [Provides detailed answer retrieved from medical PDFs]

You: What are its symptoms?  
Bot: [Understands "its" refers to diabetes from previous context]

You: How is it treated?
Bot: [Continues the conversation with treatment information]
```

#### 📸 Image Analysis
1. Click **"Browse files"** in the sidebar
2. Upload a medical image (PNG, JPG, JPEG)
3. Type your question about the image
4. Get AI-powered image analysis combined with knowledge base response

**Example:**
```
[Upload chest X-ray image]
You: What abnormalities can you see in this X-ray?
Bot: 
**Image Analysis:**
- General Description: [BLIP model description]
- Medical Context: [Contextual analysis]

**Knowledge Base Response:**
[Relevant information from medical PDFs]
```

#### 🗑️ Clear History
- Click **"🗑️ Clear Chat History"** button in sidebar to reset the conversation

## 📁 Project Structure

```
MEDICAL-CHATBOT/
├── .venv/                          # Virtual environment
├── data/                           # Source medical PDF documents
│   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_5.pdf
├── vectorstore/
│   └── db_faiss/                   # FAISS vector database
│       ├── index.faiss             # Vector index file
│       └── index.pkl               # Metadata pickle file
├── .env                            # Environment variables (API keys)
├── .python-version                 # Python version specification
├── connect_memory_with_llm.py     # [Legacy/Alternative connection script]
├── create_memory_for_llm.py       # PDF processing & vector store creation
├── medibot.py                      # Main Streamlit application
├── pyproject.toml                  # Project configuration
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
└── uv.lock                         # Dependency lock file
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `medibot.py` | Main Streamlit chatbot application with UI and chat logic |
| `create_memory_for_llm.py` | Processes PDFs and creates FAISS vector store |
| `connect_memory_with_llm.py` | Alternative/legacy script for connecting to vector store |
| `.env` | Stores API keys (never commit this file!) |
| `requirements.txt` | All Python package dependencies |
| `data/` | Directory containing medical PDF documents |
| `vectorstore/db_faiss/` | FAISS vector database for semantic search |

## 🔧 Configuration

### Adjusting Vector Store Parameters

Edit `create_memory_for_llm.py` to customize chunking:

```python
# Modify chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Increase for longer contexts (e.g., 1000)
    chunk_overlap=50     # Increase for better continuity (e.g., 100)
)
```

### Changing Retrieval Settings

Edit `medibot.py` to adjust retrieval:

```python
# Modify number of documents retrieved
vectorstore.as_retriever(search_kwargs={'k': 3})  # Change k=3 to k=5 for more context
```

### LLM Configuration

Customize the language model in `medibot.py`:

```python
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Change model here
    temperature=0.5,                # Lower (0.1) = more focused, Higher (0.9) = more creative
    max_tokens=512,                 # Increase for longer responses (e.g., 1024)
    api_key=GROQ_API_KEY,
)
```

### Available Groq Models
- `llama-3.1-8b-instant` (Default - Fast & efficient)
- `llama-3.1-70b-versatile` (More powerful, slower)
- `mixtral-8x7b-32768` (Long context window)



## 🎯 Performance Optimization

### Caching Strategy

The application uses Streamlit caching for optimal performance:

```python
@st.cache_resource  # Loads once per session
def get_vectorstore():
    # Vector store loaded once and reused

@st.cache_resource  # Model loaded once
def load_vision_model():
    # BLIP model cached in memory
```

### Adding More Documents

To add new medical PDFs after initial setup:

```bash
# 1. Add new PDFs to data/ folder
cp new_medical_document.pdf data/

# 2. Recreate vector store
python create_memory_for_llm.py

# 3. Restart the app
streamlit run medibot.py
```

## 🔒 Security & Privacy

### API Key Security
- ✅ Store API keys in `.env` file
- ✅ Add `.env` to `.gitignore`
- ❌ Never commit API keys to version control
- ❌ Never hardcode keys in source files

### Data Privacy
- All PDF processing happens locally
- Vector embeddings stored locally
- Only API calls: LLM inference (Groq) and optional image analysis (HF)
- No medical data sent to third parties

### Best Practices
```bash
# Always use environment variables
GROQ_API_KEY=your_key_here  # In .env file

# Never do this:
api_key = "gsk_hardcoded_key"  # ❌ WRONG!
```

## ⚠️ Medical Disclaimer

**IMPORTANT LEGAL NOTICE:**

This chatbot is an **AI-powered informational tool** and is **NOT** a substitute for professional medical advice, diagnosis, or treatment. 

- ❌ Do not use for medical emergencies
- ❌ Do not make treatment decisions based solely on chatbot responses
- ✅ Always consult qualified healthcare professionals
- ✅ Verify critical medical information with licensed providers

**For emergencies, call your local emergency number immediately.**

The developers and contributors of this project assume no liability for any medical decisions made based on information provided by this chatbot.


## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
MIT License - Free to use, modify, and distribute with attribution
```


## 🌟 Show Your Support

If you find this project helpful, please consider:

- ⭐ **Starring** the repository
- 🍴 **Forking** for your own projects
- 📢 **Sharing** with others
- 🐛 **Reporting bugs** to improve quality
- 💡 **Suggesting features** for enhancement

---

<div align="center">

**Made with ❤️ using LangChain, Groq, and Streamlit**

[⬆ Back to Top](#-medical-chatbot-with-rag-chat-history--image-analysis)

</div>


