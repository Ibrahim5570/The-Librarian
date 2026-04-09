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

# 2. Environment Setup
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]


# 3. Cached Initializers
@st.cache_resource
def get_retriever():
    # This represents your local "trusted" database
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
    return vectorstore.as_retriever(search_kwargs={"k": 2})  # Reduced k to reduce noise


@st.cache_resource
def get_llm():
    repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.4,  # Slightly higher for better conversational flow
    )
    return ChatHuggingFace(llm=llm)


# 4. Main Application logic
def main():
    st.set_page_config(page_title="Book Librarian", page_icon="📚")
    st.title("📚 AI Book Librarian")
    st.markdown("I am an expert librarian. I can help you with my local collection or general book inquiries.")

    # --- Memory Setup ---
    msgs = StreamlitChatMessageHistory(key="chat_messages")

    # Initialize components
    retriever = get_retriever()
    llm = get_llm()

    # --- UPDATED PROMPT TEMPLATE ---
    template = """You are a helpful expert book librarian. 

    DIRECTIONS:
    1. If the user asks about a book found in the CONTEXT, prioritize that information.
    2. If the user asks about a book NOT in the context (like 'Credence'), use your general knowledge to provide an accurate description.
    3. If the user asks for a content advisory or mentions "mature content/smut," provide an honest assessment of the book's themes. 
    4. Do not recommend children's books as substitutes for adult-themed inquiries.
    5. Use the CHAT HISTORY to stay on track with the conversation.

    CONTEXT FROM COLLECTION:
    {context}

    CHAT HISTORY:
    {history}

    USER REQUEST: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def get_history_string(_):
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
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if user_query := st.chat_input("Ask me about a book..."):
        st.chat_message("human").write(user_query)

        with st.chat_message("ai"):
            with st.spinner("Searching the library..."):
                response = chain.invoke({"question": user_query})
                st.write(response)

        # Update Memory
        msgs.add_user_message(user_query)
        msgs.add_ai_message(response)


if __name__ == "__main__":
    main()