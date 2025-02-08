import streamlit as st
from chatbot import generate_response

st.title("Fine-tuned GPT Chatbot 🤖")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Ask me anything:")

if st.button("Send"):
    if user_input:
        response = generate_response(user_input)

        # Store conversation
        st.session_state.chat_history.append(f"You: {user_input}")
        st.session_state.chat_history.append(f"Bot: {response}")

# Display chat history
for msg in st.session_state.chat_history:
    st.write(msg)
