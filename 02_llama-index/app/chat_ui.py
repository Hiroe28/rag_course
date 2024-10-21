# chat_ui.py
import streamlit as st
from llama_index.core import Settings
from common_setup import get_llm_and_embed_model, load_environment
from langchain.memory import ConversationBufferMemory
from llama_index.core.response_synthesizers.type import ResponseMode
from rag import setup_rag, load_index_and_documents

@st.cache_resource
def load_cached_index_and_documents():
    return load_index_and_documents()

class ChatUI:

    def __init__(self):
        st.set_page_config(page_title="RAG-based QA System", layout="wide")
        st.header("RAG-based QA System with Precise File References")

        self.config = load_environment()
        self.llm, self.embed_model = get_llm_and_embed_model()

        # グローバル設定を更新
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # キャッシュされたインデックスとドキュメントのロード
        self.vector_index, self.documents = load_cached_index_and_documents()

        # セッション状態の初期化
        self.initialize_session_state()

        # 初回のみクエリエンジンの設定を行う
        if "query_engine" not in st.session_state:
            self.setup_query_engine()
        else:
            # セッション状態からクエリエンジンを取得
            self.query_engine = st.session_state.query_engine

    def initialize_session_state(self):
        if "similarity_top_k" not in st.session_state:
            st.session_state.similarity_top_k = 5
        if "response_mode" not in st.session_state:
            st.session_state.response_mode = ResponseMode.COMPACT
        if "structured_answer_filtering" not in st.session_state:
            st.session_state.structured_answer_filtering = True
        if "use_auto_retriever" not in st.session_state:
            st.session_state.use_auto_retriever = True
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.0
        if "language" not in st.session_state:
            st.session_state.language = "Japanese"
        if "role" not in st.session_state:
            st.session_state.role = "あなたは親切で知識豊富なAIアシスタントです。与えられたファイルの情報を基に正確に回答します。"
        if "use_keywords" not in st.session_state:
            st.session_state.use_keywords = False


    def side(self):
        st.sidebar.title("設定")
        
        temperature = st.sidebar.slider(
            "Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.01,
            help="高いほど会話のランダム性が高くなります。低い値では一貫性のある回答が、高い値では創造的な回答が得られやすくなります。"
        )

        role = st.sidebar.text_area(
            label="System Message:",
            value=st.session_state.role,
            help="AIアシスタントの役割や振る舞いを定義します。ここで指定した内容に基づいてAIが応答を生成します。"
        )

        languages = sorted(["Japanese", "English", "Chinese", "German", "French"])
        language = st.sidebar.selectbox(
            label="Language:",
            options=languages,
            index=languages.index(st.session_state.language),
            help="AIアシスタントが応答を生成する際に使用する言語を指定します。"
        )

        # RAG設定
        st.sidebar.title("RAG設定")

        use_auto_retriever = st.sidebar.radio(
            "検索方法:",
            options=["自動検索", "再帰的検索"],
            index=0 if st.session_state.use_auto_retriever else 1,
            format_func=lambda x: x,
            help="自動検索：AIが自動的にフィルターを選択します。再帰的検索：文書の階層構造を利用して検索します。"
        ) == "自動検索"

        use_keywords = st.sidebar.checkbox(
            "キーワード検索を使用",
            value=st.session_state.use_keywords,
            help="キーワードを使用して検索精度を向上させます。自動検索モードでのみ有効です。"
        )
        # キーワード検索は自動検索モードでのみ有効にする
        if not use_auto_retriever:
            use_keywords = False
            st.sidebar.info("キーワード検索は自動検索モードでのみ使用できます。")


        similarity_top_k = st.sidebar.slider(
            "Similarity Top K:",
            min_value=1,
            max_value=20,
            value=st.session_state.similarity_top_k,
            step=1,
            help="検索する類似ドキュメントの数を指定します。多いほど広範囲の情報を参照しますが、処理時間が長くなる可能性があります。"
        )

        response_modes = [mode.value for mode in ResponseMode]
        response_mode = ResponseMode(st.sidebar.selectbox(
            "Response Mode:",
            options=response_modes,
            index=response_modes.index(st.session_state.response_mode.value),
            help="回答生成のモードを指定します。各モードの特徴は以下の通りです：\n"
                 "- refine: 各テキストチャンクを順に使用して回答を生成し、逐次的に洗練させます。詳細な回答が得られますが、LLM呼び出しが多くなります。\n"
                 "- compact: テキストチャンクを大きな塊に結合してからrefineを適用します。refineより高速で、コンテキストウインドウを効率的に利用します。\n"
                 "- simple_summaraize: 全テキストチャンクを1つに結合し、1回のLLM呼び出しで回答を生成します。コンテキストウインドウのサイズ制限に注意が必要です。\n"
                 "- tree_summaraize: テキストチャンクから木構造のインデックスを構築し、クエリを考慮しながらボトムアップで要約します。\n"
                 "- generation: コンテキストを無視し、LLMのみで回答を生成します。\n"
                 "- no_text: 最終的な回答を生成せず、検索されたコンテキストノードのみを返します。\n"
                 "- context_only: 全テキストチャンクを連結した文字列を返します。\n"
                 "- accumulate: 各テキストチャンクに対して個別に回答を生成し、それらを連結します。\n"
                 "- compact_accumulate: テキストチャンクを結合してからACCUMULATEを適用します。ACCUMULATEより高速です。"
        ))

        structured_answer_filtering = st.sidebar.checkbox(
            "Structured Answer Filtering",
            value=st.session_state.structured_answer_filtering,
            help="回答の質を向上させるフィルタリング機能を有効にします。\n"
                 "- 有効時：無関係な情報や「わかりません」といった回答を除外し、より適切な情報に焦点を当てます。\n"
                 "- 特に'refine'や'compact'モードで効果的です。"
        )

        # 設定変更の検出
        settings_changed = (
            temperature != st.session_state.temperature or
            role != st.session_state.role or
            language != st.session_state.language or
            use_auto_retriever != st.session_state.use_auto_retriever or
            use_keywords != st.session_state.use_keywords or
            similarity_top_k != st.session_state.similarity_top_k or
            response_mode != st.session_state.response_mode or
            structured_answer_filtering != st.session_state.structured_answer_filtering
        )

        if settings_changed:
            st.sidebar.warning("設定が変更されました。適用するには「RAG設定を適用」ボタンを押してください。")

        # Azure OpenAI か通常の OpenAI かを表示
        st.sidebar.markdown(f"**使用中のサービス:** {'Azure OpenAI' if self.config['use_azure'] else 'OpenAI'}")
        st.sidebar.markdown(f"**モデル:** {self.config['chat_model']}")

        # RAG設定が変更された場合、クエリエンジンを再設定
        if st.sidebar.button("RAG設定を適用"):
            st.session_state.temperature = temperature
            st.session_state.role = role
            st.session_state.language = language
            st.session_state.use_auto_retriever = use_auto_retriever
            st.session_state.use_keywords = use_keywords
            st.session_state.similarity_top_k = similarity_top_k
            st.session_state.response_mode = response_mode
            st.session_state.structured_answer_filtering = structured_answer_filtering
            
            self.setup_query_engine()
            st.success("RAG設定が適用されました。")

    def setup_query_engine(self):
        self.query_engine, _ = setup_rag(
            self.vector_index,
            self.documents,
            similarity_top_k=st.session_state.similarity_top_k,
            response_mode=st.session_state.response_mode,
            structured_answer_filtering=st.session_state.structured_answer_filtering,
            use_auto_retriever=st.session_state.use_auto_retriever,
            use_keywords=st.session_state.use_keywords,
            system_message=st.session_state.role,
            language=st.session_state.language,
            temperature=st.session_state.temperature
        )
        # query_engine をセッション状態に保存
        st.session_state.query_engine = self.query_engine
        

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

                # デバッグ情報の出力
                if st.session_state.use_keywords and st.session_state.use_auto_retriever:
                    if hasattr(self.query_engine, '_custom_retriever') and hasattr(self.query_engine._custom_retriever, 'get_extracted_keywords'):
                        extracted_keywords = self.query_engine._custom_retriever.get_extracted_keywords(prompt)
                        st.info(f"抽出されたキーワード: {extracted_keywords}")
                    else:
                        st.warning("キーワード抽出機能が利用できません")
                        st.info(f"query_engine type: {type(self.query_engine)}")
                        st.info(f"retriever type: {type(self.query_engine._custom_retriever) if hasattr(self.query_engine, '_custom_retriever') else 'No retriever'}")

                if not ai_response.strip():
                    ai_response = "申し訳ありません。その質問に対する適切な回答が見つかりませんでした。質問を別の方法で言い換えるか、より具体的な情報を提供していただけますか？"

            except Exception as e:
                ai_response = "申し訳ありません。回答の生成中にエラーが発生しました。しばらくしてからもう一度お試しください。"
                sources = []
                st.error(f"クエリ処理中にエラーが発生しました: {str(e)}")

            with st.chat_message("assistant"):
                st.write(ai_response)

            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.session_state.memory.save_context({"input": prompt}, {"outputs": ai_response})

            if sources:
                st.subheader("参照したチャンク情報:")
                
                tabs = st.tabs([f"チャンク {i+1}" for i in range(len(sources))])
                
                for i, (tab, node) in enumerate(zip(tabs, sources)):
                    with tab:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(label="類似度スコア", value=f"{node.score:.4f}")
                        with col2:
                            st.info(f"**ファイル名:** {node.metadata.get('filename', '不明')}")
                    
                        # ファイルレベルのサマリー表示
                        file_summary = node.metadata.get('summary', '要約なし')
                        if file_summary != '要約なし':
                            st.success(f"**ファイル要約:** {file_summary}")
                        else:
                            st.warning("このファイルには要約がありません。")

                        # チャンク単位のサマリー表示
                        chunk_summary = node.metadata.get('chunk_summary', '要約なし')
                        if chunk_summary != '要約なし':
                            st.success(f"**チャンク要約:** {chunk_summary}")
                        else:
                            st.warning("このチャンクには要約がありません。")
                                

def main():
    ui = ChatUI()
    ui.side()
    ui.conversation()

if __name__ == "__main__":
    main()

