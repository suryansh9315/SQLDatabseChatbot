import streamlit as st
from langchain_helper import get_chain

st.title("AtliQ T Shirts: Database Q&A 👕")

question = st.text_input("Question: ")

if question:
    chain = get_chain()
    response = chain.invoke({"question": question})

    st.header("Answer")
    st.write(response)





