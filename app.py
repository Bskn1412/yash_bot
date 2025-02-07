# pip install flask flask-socketio eventlet
# pip install google-search-results faiss-cpu sentence-transformers google-generativeai
#  AIzaSyDrKmxWwvKOIiwYf1xJqU4cLaVm85ijXVw  //for gemini flash 2.0 x
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable WebSocket

# Configure API Keys
GEMINI_API_KEY = "AIzaSyBNAZnOE_670PQ1NO0Inq87L1rGjewCqls"
genai.configure(api_key=GEMINI_API_KEY)

# Load Embedding Model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample Documents
art_projects = [
    "How to create a cardboard castle...",
    "How to make a leaf painting...",
    "How to craft a clay pot...",
   " How to make an origami animal",
]

# Convert documents to embeddings
embeddings = embed_model.encode(art_projects, convert_to_numpy=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Retrieve relevant documents
def retrieve_relevant_docs(query, top_k=2):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [art_projects[i] for i in indices[0]]

# Stream response from Gemini AI
def stream_response(query, retrieved_projects):
    context = "\n".join(retrieved_projects)
    
    prompt = f"""
    You are an AI interactive assistant specializing in art projects. Answer the following query step by step.

    Context:
    {context}

    If the context does not fully answer, generate a detailed response.

    Question: {query}
    Answer:
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt, stream=True)

    full_response = ""  # Store full response

    for chunk in response:
        full_response += chunk.text  # Collect all response parts
    formatted_response = full_response.replace("1.", "\n1.").replace("2.", "\n2.").replace("3.", "\n3.")
    socketio.emit("bot_response", {"message": formatted_response})  # Send full response at once


@socketio.on("ask_question")
def handle_question(data):
    query = data.get("query", "")
    retrieved_docs = retrieve_relevant_docs(query)

    socketio.emit("bot_response", {"message": "Thinking..."})  # Initial loading message
    stream_response(query, retrieved_docs)  # Stream AI response

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
