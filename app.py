import streamlit as st
import pandas as pd
import warnings
import os
import sys

# --- ENVIRONMENT & SILENCE LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

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
st.set_page_config(page_title="The Librarian", page_icon="📚", layout="wide")
st.title("The Librarian")
st.markdown("Your guide into the wonderful world of books!")
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm_engine = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens=1024,
        temperature=0.1,  # Keep it extremely factual and low-creativity
        huggingfacehub_api_token=hf_token
    )
    llm = ChatHuggingFace(llm=llm_engine)

    return retriever, llm


try:
    retriever, llm = init_bot()
except Exception as e:
    st.error(f"Failed to load bot: {e}")
    st.stop()

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT LOGIC ---
if prompt_input := st.chat_input("What can I help you with?..."):
    st.chat_message("user").markdown(prompt_input)

    recent_messages = st.session_state.messages[-6:]
    chat_history_str = ""
    for msg in recent_messages:
        role = "Parent" if msg["role"] == "user" else "Librarian"
        chat_history_str += f"{role}: {msg['content']}\n"

    context_docs = retriever.invoke(prompt_input)
    context_text = ""
    links_text = "\n\n**Related from our local collection:**"
    for d in context_docs:
        context_text += f"\n---\n{d.page_content}\n"
        links_text += f"\n* [{d.metadata['title']}]({d.metadata['link']})"

    # --- THE PROMPT ---
    template = """You are "The Librarian." Your primary mission is to protect minors by providing parents with explicit, unvarnished truths about book content. Your mission includes being as helpful as you can to parents inquiring about books to keep their children safe, informed, and helped.

    #STRICT OPERATING RULES:
    #1. TRUTH OVER POLITENESS: Do not sugarcoat mature themes. If a book has adult content (like Haunting Adeline or Credence), you must explicitly flag it as 18+.
    #2. CULTURAL NEUTRALITY: Do not assume the user's origin. If the cultural context is unclear, ask before using specific cultural greetings.
    #3. PARENTAL ALLY: Use the CHAT HISTORY to remember the user's concerns about their children's ages and needs.

    IDENTITY & MISSION:
    - You are an expert in content ratings. 
    - You do NOT sugarcoat. If a book contains graphic sexual violence, stalking, or "dark romance" tropes, you MUST state this clearly to the parent.
    - You are a tool for parental informed consent.
    
    CULTURAL CONTEXT:
    - If the user's cultural background is not yet clear from the CHAT HISTORY, ask politely about their preferences/origins to ensure appropriate guidance. 
    - If the user speaks in Romanized Urdu, respond helpfully but avoid repetitive phrases like "Main samajh gaya".

    RESPONSE PROTOCOL:
    1. If asked about 'Haunting Adeline' or 'Credence', IMMEDIATELY flag them as 18+ Adult Content. 
    2. Explicitly mention that 'Haunting Adeline' involves extreme "dark romance" themes like non-consensual behavior and stalking that is NOT suitable for a 15-year-old.
    3. If a book is NOT in the Local Collection, use your internal database to give a CONTENT ADVISORY.
    4. Never recommend a children's book as a "substitute" for a mature query unless you explicitly explain WHY the original book was rejected.

    CHAT HISTORY:
    {history}

    LOCAL COLLECTION CONTEXT:
    {context}

    PARENT'S QUESTION: 
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    with st.chat_message("assistant"):
        with st.spinner("Analyzing content risks..."):
            try:
                response = chain.invoke({
                    "history": chat_history_str,
                    "context": context_text,
                    "question": prompt_input
                })

                found_local_match = any(d.metadata['title'].lower() in response.lower() for d in context_docs)
                full_response = f"{response}{links_text}" if found_local_match else response
                st.markdown(full_response)

                st.session_state.messages.append({"role": "user", "content": prompt_input})
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(
                    "The safety filter was triggered. This usually happens when discussing extremely dark themes. Try rephrasing to ask specifically for a 'content rating' or 'trigger warnings'.")
