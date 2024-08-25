import streamlit as st
import json
import requests
st.title("Trav.AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
st.logo("images/logo1.png")

with st.sidebar:
    st.title("Navigation")
    st.link_button("Travel Chat Bot", "https://streamlit.io/gallery")
    st.link_button("Activity Tracker", "https://streamlit.io/gallery")
    st.link_button("Map Tracker","https://streamlit.io/gallery")


if prompt := st.chat_input("What would you like trav.ai to help you with?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    inputs = {
    "input_text": prompt}
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream=requests.post(url="http://127.0.0.1:8000/query",data=json.dumps(inputs),timeout=1200)                                     #crew.kickoff(inputs=inputs)
        response = st.markdown(stream.json())
    st.session_state.messages.append({"role": "assistant", "content": response})
    sentiment_mapping = ["one", "two", "three", "four", "five"]
    selected = st.feedback("stars")

# if selected is not None:
#         st.markdown(f"You rated this trav.ai experience {sentiment_mapping[selected]} star(s).")


