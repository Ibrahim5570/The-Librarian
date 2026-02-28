import streamlit as st
import pandas as pd
import warnings
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- SILENCE THE NOISE ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- PAGE SETUP ---
st.set_page_config(page_title="The Librarian", page_icon="📖", layout="wide")
st.title("📖 The Librarian")
st.markdown("---")


# --- INITIALIZE BOT (Cached) ---
@st.cache_resource
def init_bot():
    # 1. API Setup
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    # 2. Load the 1,200 Books CSV
    df = pd.read_csv('books_data.csv')
    df = df.fillna("")  # Clean up empty cells

    # 3. Create Searchable Documents
    docs = []
    for _, row in df.iterrows():
        # We combine multiple columns so the bot knows EVERYTHING about the book
        content = f"Title: {row['Name']}\nAuthor: {row['Author']}\nTarget Age: {row['Age']}\nDescription: {row['Description']}"

        # We store the link in metadata so the bot can use it later
        metadata = {"link": row['Link'], "title": row['Name']}
        docs.append(Document(page_content=content, metadata=metadata))

    # 4. Create Vector Database (This might take a minute the first time)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 5. Setup the Brain (Llama 3.1)
    llm_engine = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens=512,
        temperature=0.5,  # Slightly more creative for better talking
        huggingfacehub_api_token=hf_token
    )
    llm = ChatHuggingFace(llm=llm_engine)

    return retriever, llm


# Initialize
try:
    retriever, llm = init_bot()
except Exception as e:
    st.error(f"Failed to load bot: {e}")
    st.stop()

# --- CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt_input := st.chat_input("What kind of book are you looking for?"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # 1. Get Context from CSV
    context_docs = retriever.invoke(prompt_input)
    context_text = ""
    links_text = "\n\n**Found in our collection:**"

    for d in context_docs:
        context_text += f"\n---\n{d.page_content}\n"
        links_text += f"\n* [{d.metadata['title']}]({d.metadata['link']})"

    # 2. The "Smart" Prompt
    template = """You are a helpful, wise, and witty book librarian. 

    STRICT RULES:
    1. If the user's request matches books in the CONTEXT below, recommend them specifically. Mention their target age.
    2. If the user asks for something NOT in the context (like Urdu literature or a specific genre we don't have), do NOT say 'I don't know.' Instead, use your own broad knowledge to give a great recommendation.
    3. Always be polite and encourage reading.

    CONTEXT FROM COLLECTION:
    {context}

    USER QUESTION: 
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # 3. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Browsing the shelves..."):
            response = chain.invoke({"context": context_text, "question": prompt_input})

            # Combine the AI's talk with the actual links we found
            full_response = f"{response}\n\n{links_text}"
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})