Local RAG System with LightRAG and Streamlit UI
This project implements a complete, local Retrieval-Augmented Generation (RAG) system. It uses the LightRAG library to build a knowledge base from your documents, a local LLM (served via LM Studio) for generation, a FastAPI server to handle the logic, and a Streamlit web interface for user interaction.

✨ Features
Local First: Runs entirely on your local machine. All documents and conversations remain private.

Powered by LightRAG: Leverages knowledge graphs to provide more contextually aware and accurate answers.

Flexible LLM Backend: Connects to any OpenAI-compatible API server, perfect for use with LM Studio, Ollama, or vLLM.

Multi-Format Document Support: Ingests and processes .pdf, .docx, and .txt files.

Interactive UI: A user-friendly, web-based chat interface built with Streamlit.

Live Document Uploads: Add new documents to the knowledge base on the fly through the UI without restarting the server.

Knowledge Graph Visualization: Includes a script to export the knowledge graph as an interactive HTML file, allowing you to visually explore the relationships between concepts in your documents.

🏗️ Architecture
The application is composed of three main components that run simultaneously:

LM Studio (AI Backend): Runs on Windows and serves the large language models (one for chat, one for embeddings) on a local OpenAI-compatible API endpoint.

LightRAG API Server (The Brain): A Python FastAPI server running in WSL2. It handles document ingestion, knowledge graph management, and processing queries.

Streamlit UI (The Frontend): A Python Streamlit application running in WSL2 that provides the user-facing chat interface. It communicates with the LightRAG API Server.

📋 Prerequisites
Windows 11 with WSL2 (Ubuntu distribution).

LM Studio installed on Windows.

Python 3.10 or higher installed in your WSL2/Ubuntu environment.

🚀 Setup & Installation
These are the one-time setup steps to get the project running from scratch.

Install Git:
Open a WSL2/Ubuntu terminal and ensure git is installed.

sudo apt update && sudo apt install git -y

Clone the LightRAG Repository:
This project is built directly from the LightRAG source code. Clone it into your home directory.

cd ~
git clone [https://github.com/HKUDS/LightRAG.git](https://github.com/HKUDS/LightRAG.git)

Navigate to the Project Directory:

cd LightRAG

Create and Activate a Python Virtual Environment:
This isolates the project's dependencies.

python3 -m venv venv
source venv/bin/activate

Your terminal prompt should now start with (venv).

Install All Dependencies:
This command installs LightRAG in editable mode along with all server and UI dependencies.

pip install -e ".[api]" streamlit requests pypdf python-docx pyvis numpy python-dotenv

⚙️ Configuration
Before launching, configure the server and the local LLM.

1. Configure the server.py script:
Open the server.py file in a text editor (nano server.py).

DATA_DIR: Set this variable to the full path of the Windows folder containing your initial documents. Remember to use the /mnt/c/ format (e.g., /mnt/c/Users/YourName/Documents/MyFiles).

BASE_URL: Update the IP address in this URL to match the one provided by your LM Studio server logs when Serve on Local Network is enabled.

2. Configure LM Studio:
Launch LM Studio on Windows.

Load your chosen models:

Chat Model: mistralai/mistral-nemo-instruct-2407

Embedding Model: nomic-ai/nomic-embed-text-v1.5

Go to the Local Server tab (<-->).

Ensure the setting Serve on Local Network is turned ON.

Click Start Server.

▶️ Running the Application
You will need two separate WSL2 terminals, both navigated to the ~/LightRAG directory and with the (venv) activated.

Terminal 1: Start the LightRAG API Server

uvicorn server.py:app --host 0.0.0.0 --port 8000

This server will first process the documents in your DATA_DIR and then wait for requests.

Terminal 2: Start the Streamlit UI

streamlit run ui.py

This will provide a URL (http://localhost:8501) that you can open in your Windows web browser.

You can now chat with your documents via the web interface!

🛠️ Usage and Management
Chatting and Uploading
Ask questions about your documents in the main chat input.

Use the sidebar to upload new .pdf, .docx, or .txt files, which are added to the knowledge base instantly.

Visualizing the Knowledge Graph
Stop the Uvicorn server (Ctrl+C).

Run the visualization script:

python visualize.py

This creates a knowledge_graph.html file. Copy it to your Windows Desktop to view it:

cp knowledge_graph.html /mnt/c/Users/YourName/Desktop/

Resetting the Knowledge Base
Stop the Uvicorn server (Ctrl+C).

Delete the storage directory:

rm -rf rag_storage/

Restart the server. It will create a new, empty knowledge base and re-process the files from your DATA_DIR.

Creating a New, Separate Knowledge Base
Stop the Uvicorn server.

Edit server.py and change the WORKING_DIR variable from "./rag_storage" to a new name, like "./new_project_kb".

Restart the server. It will create and use this new directory.