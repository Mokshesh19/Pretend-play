# FanFiction RAG

## Overview
FanFictionRAG is a **Retrieval-Augmented Generation (RAG) system** designed for **fan fiction character roleplay**. This project extracts knowledge from fan fiction PDFs, builds a **vector database** for efficient search, and generates responses using **OpenAI's GPT-3.5-turbo** while staying in character.

## Features
- **PDF Parsing & Indexing**: Extracts character-related content from fan fiction PDFs.
- **Vector Search with FAISS**: Retrieves relevant context from stored documents.
- **Character-Aware Prompting**: Ensures the AI stays true to the character’s background and speech.
- **OpenAI API Integration**: Generates immersive, in-character responses.

## Project Structure
```
📂 project-root
 ├── 📂 data                  # PDF documents for indexing
 ├── 📜 fanfiction_rag.py     # Main implementation
 ├── 📜 README.md             # Documentation
```

## Setup & Installation
### 1. Clone the Repository
```sh
git clone https://github.com/your-username/Pretend-play
cd FanFictionRAG
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```
Or manually install required libraries:
```sh
pip install openai langchain langchain_community faiss-cpu python-dotenv
```

### 3. Set Up API Key
Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage
### Running the System
```sh
python fanfiction_rag.py
```
You'll be prompted to **ask the character a question**.

### Example Interaction
```
> What was your biggest challenge?

As El, I recall facing my darkest fears when I had to confront...

Sources: book1.pdf, book3.pdf
```

## How It Works
1. **Loads PDFs** from the specified directory.
2. **Extracts character information** using regex.
3. **Splits text into chunks** and creates a **FAISS vector store**.
4. **Searches relevant chunks** based on user queries.
5. **Generates in-character responses** using OpenAI’s API.

## Customization
- Modify `character_name` to use different characters.
- Change `pdf_directory` to process different fan fiction collections.
- Adjust `temperature` and `max_tokens` in `generate_response()` for fine-tuning response style.

## Future Enhancements
- Expand character analysis with **NLP-based entity recognition**.
- Implement **multi-character roleplay**.
- Improve response coherence using **fine-tuned language models**.

## License
This project is **MIT Licensed**. Feel free to contribute!

