import streamlit as st
from agent.runner import run_agent_pipeline

# Set page title and layout
st.set_page_config(page_title="Modexa AI Chat", layout="centered")
st.title("Modexa AI – Your Agentic Data Scientist")

# Initialize chat history in session state with default assistant message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi I'm Modexa AI, your agentic data scientist here to answer your data questions!"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Say something...")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        plan, response = run_agent_pipeline(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
