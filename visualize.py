import asyncio
import logging
import os
import numpy as np

# We will use a library called 'pyvis' to draw the graph.
# You may need to install it first: pip install pyvis
from pyvis.network import Network

from lightrag.lightrag import LightRAG
from lightrag.utils import EmbeddingFunc # Import the required class

# Setup logging and the working directory, same as our server script
logging.basicConfig(level=logging.INFO)
WORKING_DIR = "./rag_storage"
OUTPUT_FILENAME = "knowledge_graph.html"

# --- DUMMY FUNCTIONS ---
# We provide placeholder async functions to satisfy the LightRAG constructor.
async def dummy_llm_func(prompt: str, **kwargs) -> str:
    return "This is a dummy function."

async def dummy_embedding_func(texts: list[str], **kwargs) -> np.ndarray:
    # We create a fake embedding vector of the correct dimension (768).
    return np.zeros((len(texts), 768))

async def generate_graph_visualization():
    """
    This function loads the existing RAG storage and exports the
    knowledge graph as an interactive HTML file.
    """
    if not os.path.exists(WORKING_DIR):
        print(f"Error: Working directory '{WORKING_DIR}' not found.")
        print("Please run the server first to process some documents and create the storage.")
        return

    print("Initializing LightRAG to load existing storage...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=dummy_llm_func,
        embedding_func=EmbeddingFunc(
            func=dummy_embedding_func,
            embedding_dim=768, # Must match the dimension used in your server
            max_token_size=8192
        )
    )

    # This loads the data and creates the storage client objects
    await rag.initialize_storages()
    print("Storage loaded successfully.")

    print("Exporting knowledge graph data...")
    # THE FIX: Access the graph object from *within* the storage client object.
    graph = rag.chunk_entity_relation_graph._graph

    # The check now correctly uses graph.nodes without calling it as a function.
    if not graph or not graph.nodes:
        print("Knowledge graph is empty or could not be loaded. Please process some documents first.")
        await rag.finalize_storages()
        return

    # Create a pyvis network object for visualization
    print(f"Found {len(graph.nodes)} nodes and {len(graph.edges)} relationships. Creating visualization...")
    
    net = Network(height="900px", width="100%", notebook=True, cdn_resources="in_line", directed=True)
    
    # This is a good layout algorithm for knowledge graphs
    net.force_atlas_2based(gravity=-60, spring_length=250, overlap=0.5)

    # Add nodes (entities) and edges (relationships) to the pyvis graph
    for node, data in graph.nodes(data=True):
        # We can add extra information to the hover tooltip
        title = f"Type: {data.get('type', 'N/A')}\nDescription: {data.get('description', '')}"
        net.add_node(node, label=data.get('name', node), title=title)

    for source, target, data in graph.edges(data=True):
        # We can show the relationship description on hover
        net.add_edge(source, target, title=data.get('description', ''))
    
    print(f"Saving interactive graph to {OUTPUT_FILENAME}...")
    net.save_graph(OUTPUT_FILENAME)

    await rag.finalize_storages()
    
    print(f"\nSuccess! Your interactive knowledge graph has been saved.")
    print(f"To view it, copy '{OUTPUT_FILENAME}' from this WSL2 directory to your Windows Desktop and open it in a web browser.")


if __name__ == "__main__":
    asyncio.run(generate_graph_visualization())
