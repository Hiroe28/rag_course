import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from dotenv import load_dotenv
import logging
import sys
from common_setup import get_llm_and_embed_model

# スクリプトのディレクトリを取得
script_dir = os.path.dirname(os.path.abspath(__file__))
# プロジェクトのルートディレクトリを取得（スクリプトの親ディレクトリ）
project_root = os.path.dirname(script_dir)
# dataディレクトリへのパスを作成
data_dir = os.path.join(project_root, "data")

def setup_index():
    # ログ設定
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(f"データディレクトリ: {data_dir}")
    
    # 環境変数の読み込み
    load_dotenv()
    
    # モデルとエンベディングを取得
    llm, embed_model = get_llm_and_embed_model()
    
    # グローバル設定を更新
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # ドキュメントの読み込み
    reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
    documents = reader.load_data()
    logging.info(f"読み込まれたドキュメント数: {len(documents)}")
    
    for doc in documents:
        logging.info(f"読み込まれたファイル: {doc.metadata.get('file_name', 'Unknown')}")
    
    # インデックスの作成
    index = VectorStoreIndex.from_documents(documents)
    return index

def query_index(index):
    # クエリエンジンの設定
    query_engine = index.as_query_engine()
    
    while True:
        # ユーザーからのクエリ入力
        query = input("質問を入力してください（終了するには 'quit' と入力）: ")
        
        if query.lower() == 'quit':
            break
        
        # クエリを実行
        response = query_engine.query(query)
        
        # 結果を表示
        print("\nQuery:", query)
        print("Answer:", response)
        print("\n" + "-"*50 + "\n")

def main():
    index = setup_index()
    query_index(index)

if __name__ == "__main__":
    main()
