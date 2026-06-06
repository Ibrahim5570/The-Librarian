# 📚 The Librarian

A Streamlit-based chatbot designed to help parents make informed decisions about the books their children want to read. The Librarian provides honest, unfiltered content advisories, flagging mature themes, age-inappropriate content, and potential triggers so parents can say yes or no with full information.

---

## What It Does

- Accepts a parent's question about any book (e.g. *"My 3-year-old struggles with reading and I am afraid they'll fall behind"*)
- Searches a local curated book collection using semantic similarity (FAISS + HuggingFace Embeddings)
- Uses a large language model (Llama 3.3 70B via Groq) to generate a honest content advisory
- Flags 18+ content, dark romance tropes, graphic violence, non-consensual themes, and other mature material explicitly
- Links relevant books from the local collection when a match is found
- Maintains conversational history so it remembers the child's age and the parent's concerns across the session

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Llama 3.3 70B Versatile (via Groq API) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | FAISS |
| LLM Framework | LangChain |
| Data | CSV (`books_data.csv`) |

---

## Project Structure

```
chatbot/
├── app.py                  # Main application
├── books_data.csv          # Local book collection (Name, Author, Age, Description, Link)
├── .streamlit/
│  └── secrets.toml        # API keys (never commit this)
├── README.md   
└── requirements.txt
```

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/the-librarian.git
cd the-librarian
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API keys

Create a `.streamlit/secrets.toml` file:
```toml
HUGGINGFACEHUB_API_TOKEN = "your_hf_token_here"
GROQ_API_KEY = "your_groq_api_key_here"
```

Get your keys here:
- HuggingFace token → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Groq API key (free) → [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
python -m streamlit run app.py
```

---

## books_data.csv Format

The local collection CSV must have these columns:

| Column | Description |
|---|---|
| `Name` | Book title |
| `Author` | Author name |
| `Age` | Target age range (e.g. `13+`, `18+`) |
| `Description` | Short description of the book |
| `Link` | URL to book page or purchase link |

---

## Example Usage

> **Parent:** My 14 year old daughter is asking permission to read Haunting Adeline. What is it about?

> **The Librarian:** ⚠️ **Not appropriate for a 14-year-old.** Haunting Adeline is an 18+ dark romance novel that contains graphic depictions of stalking, non-consensual sexual behavior, and sexual violence. The male lead's obsession with the protagonist is romanticized throughout the book, which makes it particularly harmful for teenage readers who may internalize unhealthy relationship dynamics. This book is not suitable for minors under any circumstances.

---

## Notes

- The bot uses conversational memory (last 6 messages) to stay aware of the child's age and the parent's concerns throughout the session
- If a book is not in the local collection, the LLM falls back on its own training knowledge to provide a content advisory
- The app is designed for parents  not children

---

## License

MIT License. Free to use, modify, and distribute.
