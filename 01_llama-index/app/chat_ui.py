import streamlit as st
from dotenv import load_dotenv
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
import logging
import sys
from common_setup import get_llm_and_embed_model
from langchain.memory import ConversationBufferMemory

class ChatUI:
    def __init__(self):
        # Streamlitのページ設定
        st.set_page_config(page_title="Basic RAG", layout="wide")
        st.header("Basic RAG")
        
        # 初期セッションステートの設定
        self.data_dir = "data"
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.0
        if "language" not in st.session_state:
            st.session_state.language = "Japanese"
        if "role" not in st.session_state:
            st.session_state.role = "あなたは親切なAIアシスタントです。"
        
        self.setup_index()

    def setup_index(self):
        # ログ設定
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        
        # 環境変数の読み込み
        load_dotenv()
        
        # モデルとエンベディングの取得
        llm, embed_model = get_llm_and_embed_model()
        
        # グローバル設定を更新
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # ドキュメントの読み込み
        documents = SimpleDirectoryReader(self.data_dir, recursive=True).load_data()
        self.index = VectorStoreIndex.from_documents(documents)

    def side(self):
        # サイドバーに設定項目を追加
        st.sidebar.button("＋ 新しい会話を始める", key="clear", on_click=self.chat_reset)
        st.sidebar.title("設定")
        
        # Temperatureスライダー
        st.session_state.temperature = st.sidebar.slider(
            "Temperature:\n\n高いほど会話のランダム性が高くなる",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.01,
        )
        
        # Systemメッセージテキストエリア
        st.session_state.role = st.sidebar.text_area(
            label="System Message:\n\n会話の前提として与えるAIの役割",
            max_chars=None,
            value=st.session_state.role
        )
        
        # 言語選択ボックス
        languages = sorted(["Japanese", "English", "Chinese", "German", "French"])
        st.session_state.language = st.sidebar.selectbox(
            label="Language:\n\n指定した言語で返答を行いやすくなる",
            options=languages,
            index=languages.index(st.session_state.language),
        )

    def chat_reset(self):
        # チャット履歴とメモリをリセット
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.setup_index()

    def conversation(self):
        # メッセージとメモリの初期化
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # 過去のメッセージを表示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # ユーザーの入力を受け取る
        if prompt := st.chat_input("Please input your question."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            synth = get_response_synthesizer(streaming=True)

            # VectorStoreIndexからリトリーバーを作成
            retriever = self.index.as_retriever(similarity_top_k=1)

            # RetrieverQueryEngineのインスタンス化
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=synth
            )
            
            streaming_response = query_engine.query(prompt)
            
            # レスポンスをリアルタイムで表示
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                for text in streaming_response.response_gen:
                    full_response += text
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            
            # メッセージとメモリに保存
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.memory.save_context({"input": prompt}, {"output": full_response})

def main():
    # 環境変数の読み込み
    load_dotenv(r"secret\.env", override=True)
    
    # UIの初期化と設定
    ui = ChatUI()
    ui.side()
    ui.conversation()

if __name__ == "__main__":
    main()
