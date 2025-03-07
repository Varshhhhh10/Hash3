import os
import nest_asyncio
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

from llama_index.core import SimpleDirectoryReader, Settings, SummaryIndex, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

nest_asyncio.apply()

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = "enter_your_key"

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is missing. Set it in a .env file.")

# Initialize Flask app
app = Flask(__name__, static_folder="static")

# Load documents
documents = SimpleDirectoryReader(input_files=["College Website Data.pdf"]).load_data()

# Split documents into nodes
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# Configure OpenAI settings with API key
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)

# Create summary and vector indexes
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

# Create query engines
summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
vector_query_engine = vector_index.as_query_engine()

# Define query engine tools
summary_tool = QueryEngineTool.from_defaults(query_engine=summary_query_engine,
                                             description="Useful for summarization questions related to College data")

vector_tool = QueryEngineTool.from_defaults(query_engine=vector_query_engine,
                                            description="Useful for retrieving specific details about the College.")

# Configure router query engine
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[summary_tool, vector_tool],
    verbose=True
)

# Serve HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Handle Chat Requests
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_query = data.get("question", "")  # Changed "query" → "question" to match frontend

    if not user_query:
        return jsonify({"error": "No question provided"}), 400

    response = query_engine.query(user_query)
    return jsonify({"answer": str(response)})  # Changed "response" → "answer" to match frontend

# Run Flask server
if __name__ == '__main__':
    app.run(debug=True)
