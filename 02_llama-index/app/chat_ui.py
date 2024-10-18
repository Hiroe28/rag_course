# chat_ui4.py
import streamlit as st

from common_setup import get_llm_and_embed_model, load_environment
from langchain.memory import ConversationBufferMemory
from llama_index.core import Settings
from llama_index.core.response_synthesizers.type import ResponseMode
from rag import setup_rag
from vector_save import load_existing_index

@st.cache_resource
def load_cached_index():
    return load_existing_index()

class ChatUI:

    def __init__(self):
        st.set_page_config(page_title="RAG-based QA System", layout="wide")
        st.header("RAG-based QA System with Precise File References")

        self.config = load_environment()
        self.llm, self.embed_model = get_llm_and_embed_model()

        # グローバル設定を更新
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # キャッシュされたインデックスの読み込み
        self.vector_index, self.keyword_index = load_cached_index()
        
        if self.vector_index is None or self.keyword_index is None:
            st.error("インデックスが見つかりません。vector_save.pyを実行してインデックスを作成してください。")
            st.stop()

        # RAG設定の初期化
        if "similarity_top_k" not in st.session_state:
            st.session_state.similarity_top_k = 5
        if "response_mode" not in st.session_state:
            st.session_state.response_mode = ResponseMode.COMPACT
        if "structured_answer_filtering" not in st.session_state:
            st.session_state.structured_answer_filtering = True
        if "hybrid_mode" not in st.session_state:
            st.session_state.hybrid_mode = "OR"
        
        self.query_engine = setup_rag(
            self.vector_index,
            self.keyword_index,
            similarity_top_k=st.session_state.similarity_top_k,
            response_mode=st.session_state.response_mode,
            structured_answer_filtering=st.session_state.structured_answer_filtering,
            hybrid_mode=st.session_state.hybrid_mode
        )

        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.0
        if "language" not in st.session_state:
            st.session_state.language = "Japanese"
        if "role" not in st.session_state:
            st.session_state.role = "あなたは親切で知識豊富なAIアシスタントです。与えられたファイルの情報を基に正確に回答します。"

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
            value=st.session_state.role
        )

        languages = sorted(["Japanese", "English", "Chinese", "German", "French"])

        st.session_state.language = st.sidebar.selectbox(
            label="Language:\n\n指定した言語で返答を行いやすくなる",
            options=languages,
            index=languages.index(st.session_state.language),
        )

        # RAG設定
        st.sidebar.title("RAG設定")

        hybrid_modes = ["AND", "OR"]
        st.session_state.hybrid_mode = st.sidebar.selectbox(
            "Hybrid Mode:\n\nベクトル検索とキーワード検索の組み合わせ方",
            options=hybrid_modes,
            index=hybrid_modes.index(st.session_state.hybrid_mode),
        )

        st.session_state.similarity_top_k = st.sidebar.slider(
            "Similarity Top K:\n\n検索する類似ドキュメントの数",
            min_value=1,
            max_value=20,
            value=st.session_state.similarity_top_k,
            step=1,
        )

        response_modes = [mode.value for mode in ResponseMode]
        st.session_state.response_mode = ResponseMode(st.sidebar.selectbox(
            "Response Mode:\n\n回答生成のモード",
            options=response_modes,
            index=response_modes.index(st.session_state.response_mode.value),
        ))

        st.session_state.structured_answer_filtering = st.sidebar.checkbox(
            "Structured Answer Filtering",
            value=st.session_state.structured_answer_filtering,
            help="構造化された回答のフィルタリングを行うかどうか"
        )

        # Azure OpenAI か通常の OpenAI かを表示
        st.sidebar.markdown(f"**使用中のサービス:** {'Azure OpenAI' if self.config['use_azure'] else 'OpenAI'}")
        st.sidebar.markdown(f"**モデル:** {self.config['chat_model']}")

    def chat_reset(self):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # インデックスを再作成するのではなく、既存のインデックスを読み込む
        self.vector_index, self.keyword_index = load_existing_index()
        
        if self.vector_index is None or self.keyword_index is None:
            st.error("インデックスが見つかりません。vector_save.pyを実行してインデックスを作成してください。")
            st.stop()
        
        self.query_engine = setup_rag(
            self.vector_index,
            self.keyword_index,
            similarity_top_k=st.session_state.similarity_top_k,
            response_mode=st.session_state.response_mode,
            structured_answer_filtering=st.session_state.structured_answer_filtering,
            hybrid_mode=st.session_state.hybrid_mode
        )
        
    def conversation(self):  
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        memory = st.session_state.memory

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("質問を入力してください。"):
            with st.chat_message("user"):
                st.write(prompt)

            st.session_state.messages.append({"role": "user", "content": prompt})

            try:
                response = self.query_engine.query(prompt)
                ai_response = response.response
                sources = response.source_nodes
            except Exception as e:
                ai_response = "エラーが発生しました。"
                sources = []
                st.error(f"クエリ処理中にエラーが発生しました: {str(e)}")

            with st.chat_message("assistant"):
                st.write(ai_response)

            st.session_state.messages.append({"role": "assistant", "content": ai_response})

            st.session_state.memory.save_context({"input": prompt}, {"outputs": ai_response})

            if sources:

                st.subheader("参照したチャンク情報:")
                
                # タブを使用してチャンク情報を表示
                tabs = st.tabs([f"チャンク {i+1}" for i in range(len(sources))])
                
                for i, (tab, node) in enumerate(zip(tabs, sources)):
                    with tab:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(label="類似度スコア", value=f"{node.score:.4f}")
                        with col2:
                            st.info(f"**ファイル名:** {node.metadata.get('filename', '不明')}")
                        
                        # サマリーの表示を修正
                        summary = node.metadata.get('summary', '要約なし')
                        if summary != '要約なし':
                            st.success(f"**要約:** {summary}")
                        else:
                            st.warning("このチャンクには要約がありません。")
                        
                        with st.expander("チャンクの内容を表示"):
                            st.code(node.text, language="plaintext")


def main():
    # load_dotenv(r"secret\.env", override=True)
    ui = ChatUI()
    ui.side()
    ui.conversation()

if __name__ == "__main__":
    main()