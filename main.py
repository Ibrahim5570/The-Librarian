import os
import warnings
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# 1. Silence Logs & Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Environment Setup (Using Streamlit Secrets)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "BookRecommenderBot"

# 3. Cached Initializers 
# We use @st.cache_resource so the DB and Model don't reload on every interaction
@st.cache_resource
def get_retriever():
    books = [
        {"title": "Project Hail Mary",
         "desc": "A lone astronaut must save the earth from disaster using science and a spider-like alien friend."},
        {"title": "Atomic Habits",
         "desc": "A guide to breaking bad habits and forming good ones through small, incremental changes."},
        {"title": "The Name of the Wind",
         "desc": "The story of Kvothe, a magically gifted young man who grows to be the most notorious wizard."},
        {"title": "Thinking, Fast and Slow",
         "desc": "Explains the two systems that drive the way we think: System 1 (fast/intuitive) and System 2 (slow/logical)."},
        {"title": "Dune",
         "desc": "Set on the desert planet Arrakis, the story of Paul Atreides and the 'spice' melange."}
    ]
    docs = [Document(page_content=f"Title: {b['title']}\nDescription: {b['desc']}") for b in books]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

@st.cache_resource
def get_llm():
    repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.1,
    )
    return ChatHuggingFace(llm=llm)

# 4. Main Application logic
def main():
    st.set_page_config(page_title="Book Librarian", page_icon="📚")
    st.title("📚 AI Book Librarian")
    st.markdown("I can recommend books from my database. What are you looking for?")

    # --- Memory Setup ---
    # This automatically persists messages in st.session_state["chat_messages"]
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    
    # Initialize components
    retriever = get_retriever()
    llm = get_llm()

    # Prompt Template
    template = """You are a helpful expert book librarian. 
    Answer the user's request based ONLY on the following context.

    CONTEXT:
    {context}

    CHAT HISTORY:
    {history}

    USER REQUEST: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Helper function for history formatting
    def get_history_string(_):
        # Retrieve the last 6 messages from Streamlit history
        return "\n".join([f"{m.type}: {m.content}" for m in msgs.messages[-6:]])

    # RAG Pipeline
    chain = (
        {
            "context": lambda x: "\n\n".join([d.page_content for d in retriever.invoke(x["question"])]),
            "history": get_history_string,
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- Chat Interface ---
    # Display message history on rerun
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Handle User Input
    if user_query := st.chat_input("Ask me about a book..."):
        # 1. Display User Message
        st.chat_message("human").write(user_query)
        
        # 2. Run the Chain
        with st.chat_message("ai"):
            with st.spinner("Searching the library..."):
                response = chain.invoke({"question": user_query})
                st.write(response)
        
        # 3. Update Memory
        msgs.add_user_message(user_query)
        msgs.add_ai_message(response)

if __name__ == "__main__":
    main()