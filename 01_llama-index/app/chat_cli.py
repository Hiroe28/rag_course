import streamlit as st
from dotenv import load_dotenv
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
import logging
import sys
from common_setup import get_llm_and_embed_model
from langchain.memory import ConversationBufferMemory
import textwrap

class ChatUI:
    def __init__(self):
        st.set_page_config(page_title="Basic RAG", layout="wide")
        st.header("Basic RAG")
        self.data_dir = "data"
        st.session_state.temperature = 0.0
        st.session_state.language = "Japanese"
        st.session_state.role = "あなたは親切なAIアシスタントです。"
        self.setup_index()

    def setup_index(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        load_dotenv()
        llm, embed_model = get_llm_and_embed_model()
        documents = SimpleDirectoryReader(self.data_dir, recursive=True).load_data()
        self.index = VectorStoreIndex.from_documents(documents)

    def side(self):
        st.sidebar.button("＋ 新しい会話を始める", key="clear", on_click=self.chat_reset)
        st.sidebar.title("設定")
        
        st.session_state.temperature = st.sidebar.slider(
            "Temperature:\n\n高いほど会話のランダム性が高くなる",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.01,
        )
        st.session_state.role = st.sidebar.text_area(
            label="System Message:\n\n会話の前提として与えるAIの役割",
            max_chars=None,
            value=st.session_state.role
        )
        languages = sorted(["Japanese", "English", "Chinese", "German", "French"])
        st.session_state.language = st.sidebar.selectbox(
            label="Language:\n\n指定した言語で返答を行いやすくなる",
            options=languages,
            index=languages.index(st.session_state.language),
        )

    def chat_reset(self):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.setup_index()

    def conversation(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Please input your question."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            synth = get_response_synthesizer(streaming=True)
            query_engine = RetrieverQueryEngine.from_args(
                self.index,
                response_synthesizer=synth,
                retriever=self.index.as_retriever(similarity_top_k=1)
            )
            
            streaming_response = query_engine.query(prompt)
            
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                for text in streaming_response.response_gen:
                    full_response += text
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.memory.save_context({"input": prompt}, {"output": full_response})

def main():
    load_dotenv(r"secret\.env", override=True)
    ui = ChatUI()
    ui.side()
    ui.conversation()

if __name__ == "__main__":
    main()