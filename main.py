import warnings
import os
from dotenv import load_dotenv
load_dotenv()
# 1. Silence TensorFlow logs (0 = all, 1 = no info, 2 = no warning, 3 = no errors)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Silence Python/Torch warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import sys
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# 1. Setup Environment
# ---------------------------------------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "BookRecommenderBot"

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory


# 2. Database Setup
# ---------------------------------------------------------
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
    print("Indexing database...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


# 3. Model Setup
# ---------------------------------------------------------
def get_llm():
    # Switching to Llama 3.1 which is much more stable on the API right now
    repo_id = "meta-llama/Llama-3.1-8B-Instruct"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        # Keep the temperature low for consistent librarian answers
        temperature=0.1,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )

    return ChatHuggingFace(llm=llm)


# 4. Main Execution logic
# ---------------------------------------------------------
def main():
    print("--- Initializing Book Bot ---")
    retriever = get_retriever()
    llm = get_llm()
    history = ChatMessageHistory()

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
        msgs = history.messages[-6:]  # Keep last 3 rounds of conversation
        return "\n".join([f"{m.type}: {m.content}" for m in msgs])

    # Helper function for retrieval
    def retrieve_context(query):
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])

    # The Unified RAG Pipeline
    chain = (
            {
                "context": lambda x: retrieve_context(x["question"]),
                "history": get_history_string,
                "question": lambda x: x["question"]
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    print("\nThe Librarian: Hello! I can recommend books from my database. What are you looking for?")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("The Librarian: Happy reading!")
                break

            response = chain.invoke({"question": user_input})

            # Update history
            history.add_user_message(user_input)
            history.add_ai_message(response)

            print(f"The Librarian: {response}")

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()