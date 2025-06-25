import streamlit as st
import requests
import time

# --- Page Configuration ---
# Using a wide layout to accommodate the sidebar and chat comfortably
st.set_page_config(
    page_title="LightRAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Server URLs ---
# Central place to define the addresses of your FastAPI server
QUERY_URL = "http://localhost:8000/query"
UPLOAD_URL = "http://localhost:8000/upload"

# =============================================================================
# --- SIDEBAR for Document Uploads ---
# =============================================================================
with st.sidebar:
    st.title("📎 Document Management")
    st.caption("Add new documents to the knowledge base on the fly.")
    
    # The file uploader widget allows multiple file types
    uploaded_file = st.file_uploader(
        "Upload a .pdf, .docx, or .txt file",
        type=["pdf", "docx", "txt"]
    )
    
    # This block handles the file once it's uploaded by the user
    if uploaded_file is not None:
        # Show a spinner while the file is being processed
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            try:
                # Prepare the file to be sent in a POST request
                files_to_send = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Send the file to your FastAPI server's /upload endpoint
                response = requests.post(UPLOAD_URL, files=files_to_send)
                
                if response.status_code == 200:
                    st.success(f"Success! '{uploaded_file.name}' was added to the knowledge base.")
                    time.sleep(2) # Pause for 2 seconds to let the user read the message
                    st.rerun() # Rerun the script to clear the uploader
                else:
                    st.error(f"Error uploading file: {response.text}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

# =============================================================================
# --- MAIN CHAT INTERFACE ---
# =============================================================================
st.title("🤖 Chat With Your Documents")
st.caption("Powered by LightRAG, LM Studio, and Streamlit")

# Initialize chat history in Streamlit's session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about your documents, or upload a new one using the sidebar."}]

# Display all past messages from the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the message is from the assistant and has sources, display them in an expander
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("View Cited Sources"):
                for i, source in enumerate(message["sources"]):
                    st.info(f"Source {i+1}: (from Document: {source.get('doc_id', 'N/A')})")
                    st.caption(source["text"])

# The chat input box at the bottom of the screen
if prompt := st.chat_input("Ask a question..."):
    
    # 1. Add user's new message to the history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send the user's question to the /query endpoint
                response = requests.post(QUERY_URL, json={"query_str": prompt})
                
                if response.status_code == 200:
                    response_json = response.json()
                    # Extract both the answer and the sources list from the response
                    answer = response_json.get("answer", "Sorry, I couldn't get a valid answer.")
                    sources = response_json.get("sources", [])
                    
                    st.markdown(answer)
                    
                    # Display the sources for the *current* answer in an expander
                    if sources:
                        with st.expander("View Cited Sources"):
                            for i, source in enumerate(sources):
                                st.info(f"Source {i+1}: (from Document: {source.get('doc_id', 'N/A')})")
                                st.caption(source["text"])
                    
                    # 3. Add the complete assistant message (with sources) to the history
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

                else:
                    error_answer = f"Error from server: {response.status_code} - {response.text}"
                    st.markdown(error_answer)
                    st.session_state.messages.append({"role": "assistant", "content": error_answer})

            except requests.exceptions.ConnectionError:
                conn_error_answer = "Connection Error: Could not connect to the LightRAG server. Is it running?"
                st.markdown(conn_error_answer)
                st.session_state.messages.append({"role": "assistant", "content": conn_error_answer})
