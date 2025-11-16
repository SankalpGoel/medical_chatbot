import os
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import requests

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

from dotenv import load_dotenv
load_dotenv()


DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


@st.cache_resource
def load_vision_model():
    """Load BLIP vision model for image captioning and analysis"""
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        return processor, model
    except Exception as e:
        st.error(f"Error loading vision model: {str(e)}")
        return None, None


def analyze_image_with_blip(image, prompt):
    """Analyze image using BLIP model"""
    try:
        processor, model = load_vision_model()
        if processor is None or model is None:
            return "Vision model not available. Please install transformers and torch."
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate caption
        inputs = processor(image, return_tensors="pt")
        caption_ids = model.generate(**inputs, max_length=100)
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)
        
        # Generate conditional caption based on prompt
        text_prompt = f"a medical image showing"
        inputs = processor(image, text_prompt, return_tensors="pt")
        conditional_ids = model.generate(**inputs, max_length=100)
        conditional_caption = processor.decode(conditional_ids[0], skip_special_tokens=True)
        
        analysis = f"""**Image Analysis:**
- General Description: {caption}
- Medical Context: {conditional_caption}
- Note: This is an automated analysis. For accurate medical diagnosis, please consult a healthcare professional."""
        
        return analysis
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


def process_image_with_hf_api(image, prompt):
    """Process image using Hugging Face Inference API as fallback"""
    try:
        HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
        if not HF_API_KEY:
            return None
            
        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        
        # Convert image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        response = requests.post(API_URL, headers=headers, data=img_bytes)
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            caption = result[0].get('generated_text', 'No description available')
            return f"**Image Analysis:** {caption}"
        return None
        
    except Exception as e:
        return None


def convert_messages_to_langchain(messages):
    """Convert Streamlit messages to LangChain format"""
    langchain_messages = []
    for msg in messages:
        if msg['role'] == 'user':
            langchain_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            langchain_messages.append(AIMessage(content=msg['content']))
    return langchain_messages


def create_contextualize_question_prompt():
    """Create a prompt to reformulate questions based on chat history"""
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    return contextualize_q_prompt


def create_qa_prompt():
    """Create QA prompt with chat history support"""
    qa_system_prompt = """You are a medical assistant chatbot. Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

If an image is provided, analyze it in the context of the medical query and provide relevant insights.

{context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    return qa_prompt


def main():
    st.title("🏥 Medical Chatbot with History & Image Support")
    st.caption("Ask questions about medical topics or upload medical images for analysis")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()

    # Sidebar for image upload
    with st.sidebar:
        st.header("📸 Image Upload")
        uploaded_file = st.file_uploader("Upload a medical image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = ChatMessageHistory()
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'image' in message and message['image']:
                st.image(message['image'], width=300)

    # Chat input
    prompt = st.chat_input("Ask a medical question...")

    if prompt:
        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)
            if uploaded_file is not None:
                st.image(image, width=300)
        
        # Store user message
        user_message = {'role': 'user', 'content': prompt}
        if uploaded_file is not None:
            user_message['image'] = image
        st.session_state.messages.append(user_message)
        
        try:
            # Initialize components
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.5,
                max_tokens=512,
                api_key=GROQ_API_KEY,
            )
            
            # Handle image if uploaded
            image_analysis = None
            if uploaded_file is not None:
                with st.spinner("Analyzing image..."):
                    # Try local BLIP model first
                    image_analysis = analyze_image_with_blip(image, prompt)
                    
                    # Fallback to HF API if local fails
                    if image_analysis and "Error analyzing image" in image_analysis:
                        hf_analysis = process_image_with_hf_api(image, prompt)
                        if hf_analysis:
                            image_analysis = hf_analysis
            
            # Convert chat history to LangChain format
            chat_history = convert_messages_to_langchain(st.session_state.messages[:-1])
            
            # Create history-aware retriever
            contextualize_q_prompt = create_contextualize_question_prompt()
            history_aware_retriever = create_history_aware_retriever(
                llm, 
                vectorstore.as_retriever(search_kwargs={'k': 3}),
                contextualize_q_prompt
            )
            
            # Create QA chain with history
            qa_prompt = create_qa_prompt()
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            
            # Create RAG chain
            rag_chain = create_retrieval_chain(
                history_aware_retriever, 
                question_answer_chain
            )
            
            # Prepare input
            rag_input = {
                'input': prompt,
                'chat_history': chat_history
            }
            
            # Get response
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(rag_input)
                result = response["answer"]
                
                # Combine with image analysis if available
                if image_analysis:
                    result = f"{image_analysis}\n\n**Knowledge Base Response:**\n{result}"
            
            # Display assistant response
            with st.chat_message('assistant'):
                st.markdown(result)
            
            # Store assistant message
            st.session_state.messages.append({'role': 'assistant', 'content': result})
            
            # Update chat history
            st.session_state.chat_history.add_user_message(prompt)
            st.session_state.chat_history.add_ai_message(result)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()