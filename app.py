import streamlit as st
import pandas as pd
import warnings
import os
import sys

# --- SILENCE THE NOISE ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

# This prevents Streamlit from trying to 'inspect' image modules it doesn't need
if "torchvision" not in sys.modules:
    sys.modules["torchvision"] = None

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- SECRETS SETUP ---
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# --- PAGE SETUP ---
st.set_page_config(page_title="The Librarian", page_icon="📖", layout="wide")
st.title("The Librarian")
st.markdown("---")

# --- INITIALIZE BOT (Cached) ---
@st.cache_resource
def init_bot():
    df = pd.read_csv('books_data.csv')
    df = df.fillna("") 
    
    docs = []
    for _, row in df.iterrows():
        content = f"Title: {row['Name']}\nAuthor: {row['Author']}\nTarget Age: {row['Age']}\nDescription: {row['Description']}"
        metadata = {"link": row['Link'], "title": row['Name']}
        docs.append(Document(page_content=content, metadata=metadata))

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm_engine = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens=512,
        temperature=0.5,
        huggingfacehub_api_token=hf_token
    )
    llm = ChatHuggingFace(llm=llm_engine)

    return retriever, llm

try:
    retriever, llm = init_bot()
except Exception as e:
    st.error(f"Failed to load bot: {e}")
    st.stop()

# --- SESSION STATE (The Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt_input := st.chat_input("What kind of book are you looking for?"):
    # Display user message
    st.chat_message("user").markdown(prompt_input)
    
    # 1. Format the Chat History for the LLM
    # We take the last few messages and turn them into a string
    chat_history_str = ""
    for msg in st.session_state.messages[-5:]: # Last 5 messages for context
        chat_history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"

    # 2. Get Context from Vector DB
    context_docs = retriever.invoke(prompt_input)
    context_text = ""
    links_text = "\n\n**Found in our collection:**"

    for d in context_docs:
        context_text += f"\n---\n{d.page_content}\n"
        links_text += f"\n* [{d.metadata['title']}]({d.metadata['link']})"

    # 3. The UPDATED Prompt (Including History)
    template = """You are a helpful, wise, and witty book librarian. 

    STRICT RULES:
    1. Use the CHAT HISTORY to understand the context of the conversation.
    2. If the user refers to "previous prompts" or "that book", look in CHAT HISTORY.
    3. If the user's request matches books in the CONTEXT below, recommend them specifically.
    4. If the user asks for something NOT in context, use your own knowledge.

    CHAT HISTORY:
    {history}

    CONTEXT FROM COLLECTION:
    {context}

    USER QUESTION: 
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # 4. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Browsing the shelves..."):
            # We pass history, context, and question into the chain
            response = chain.invoke({
                "history": chat_history_str, 
                "context": context_text, 
                "question": prompt_input
            })

            full_response = f"{response}\n\n{links_text}"
            st.markdown(full_response)

    # 5. Save to session state AFTER generating
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    st.session_state.messages.append({"role": "assistant", "content": full_response})