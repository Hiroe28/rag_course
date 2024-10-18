# vector_save.py
import argparse
import logging
import os
import sys
from datetime import datetime

from common_setup import get_llm_and_embed_model
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import get_response_synthesizer

# ログレベルをINFOに設定し、デバッグメッセージを減らす
logging.basicConfig(level=logging.INFO)

# スクリプトのディレクトリを取得
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# プロジェクトのルートディレクトリ（スクリプトの親ディレクトリ）を取得
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# 各ディレクトリのパスを設定
PERSIST_DIR = os.path.join(ROOT_DIR, "storage")
METADATA_FILE = os.path.join(PERSIST_DIR, "metadata.txt")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# document_loader モジュールのパスをシステムパスに追加
sys.path.append(SCRIPT_DIR)

def save_metadata(docs_list):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        f.write("ファイル名\t最終更新日時\tサマリー\n")
        f.write("-" * 100 + "\n")
        for doc in docs_list:
            filename = doc.metadata['filename']
            modification_time = datetime.fromtimestamp(doc.metadata['modification_time']).strftime('%Y-%m-%d %H:%M:%S')
            summary = doc.metadata.get('summary', 'サマリーなし')
            f.write(f"{filename}\t{modification_time}\t{summary}\n")
    logging.info(f"メタデータが{METADATA_FILE}に保存されました。")

def generate_summary(doc, llm):
    """ドキュメントの要約を生成する関数"""
    summary_template = (
        "以下の文書の要約を100字程度で作成してください。重要なポイントを簡潔に説明してください：\n\n{context}"
    )
    Settings.llm = llm
    
    summary_index = SummaryIndex.from_documents([doc])
    summary_query_engine = summary_index.as_query_engine(
        response_synthesizer=get_response_synthesizer(
            summary_template=summary_template,
        )
    )
    
    summary_response = summary_query_engine.query("このドキュメントの要約を100文字程度で作成してください。")
    return str(summary_response)


def load_existing_index():
    """既存のインデックスを読み込む関数"""
    if os.path.exists(os.path.join(PERSIST_DIR, "vector", "docstore.json")) and \
       os.path.exists(os.path.join(PERSIST_DIR, "keyword", "docstore.json")):
        logging.info("既存のインデックスを読み込み中...")
        try:
            # VectorStoreIndexを読み込む
            vector_storage_context = StorageContext.from_defaults(persist_dir=os.path.join(PERSIST_DIR, "vector"))
            vector_index = load_index_from_storage(vector_storage_context, index_id="vector_index")
            
            # SimpleKeywordTableIndexを読み込む
            keyword_storage_context = StorageContext.from_defaults(persist_dir=os.path.join(PERSIST_DIR, "keyword"))
            keyword_index = load_index_from_storage(keyword_storage_context, index_id="keyword_index")
            
            logging.info("インデックスの読み込みが完了しました。")
            return vector_index, keyword_index
        except Exception as e:
            logging.error(f"インデックスの読み込みエラー: {e}")
            return None, None
    else:
        logging.info("既存のインデックスが見つかりません。")
        return None, None



def create_or_update_index(summarize=False, chunk_size=1024, chunk_overlap=40, force_update=False):
    llm, embed_model = get_llm_and_embed_model()
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 既存のインデックスをチェック
    if not force_update:
        existing_indexes = load_existing_index()
        if all(existing_indexes):
            return existing_indexes

    # SimpleDirectoryReaderでドキュメントを読み込み、メタデータを自動で取得
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR, 
        recursive=True,
        file_metadata=lambda file_path: {
            "filename": os.path.relpath(file_path, DATA_DIR),
            "modification_time": os.path.getmtime(file_path)
        },
        filename_as_id=True
    )
    
    docs_list = reader.load_data()

    # 既存のメタデータをロード
    existing_metadata = {}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            next(f)  # ヘッダーをスキップ
            next(f)  # 区切り線をスキップ
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    filename, modification_time = parts[:2]
                    existing_metadata[filename] = float(datetime.strptime(modification_time, '%Y-%m-%d %H:%M:%S').timestamp())

    updated = False

    for doc in docs_list:
        filename = doc.metadata['filename']
        modification_time = doc.metadata['modification_time']

        # メタデータが存在しないか、更新されている場合
        if filename not in existing_metadata or abs(existing_metadata.get(filename, 0) - modification_time) > 1:
            logging.info(f"新しいまたは更新されたファイルが検出されました: {filename}")
            updated = True
            
            if summarize:
                summary = generate_summary(doc, llm)
                doc.metadata['summary'] = summary
                logging.debug(f"{filename}の要約が生成されました。")
            else:
                doc.metadata['summary'] = 'サマリー生成スキップ'
        else:
            logging.debug(f"{filename}に変更がないため、要約生成をスキップしました。")
            if 'summary' in existing_metadata:
                doc.metadata['summary'] = existing_metadata['summary']

    if updated or not os.path.exists(os.path.join(PERSIST_DIR, "vector", "docstore.json")):
        logging.info("全ドキュメントでインデックスを更新中...")
        storage_context = StorageContext.from_defaults()

        # SentenceSplitterを使用してテキストを分割（チャンクサイズとオーバーラップを引数から設定）
        text_splitter = SentenceSplitter(
            separator='。',  # 日本語の句点で区切る
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # VectorStoreIndexを作成
        vector_index = VectorStoreIndex.from_documents(
            docs_list,
            storage_context=storage_context,
            text_splitter=text_splitter
        )

        # SimpleKeywordTableIndexを作成
        keyword_index = SimpleKeywordTableIndex.from_documents(
            docs_list,
            storage_context=storage_context
        )

        # 両方のインデックスを保存（インデックスIDを指定）
        vector_index.set_index_id("vector_index")
        vector_index.storage_context.persist(persist_dir=os.path.join(PERSIST_DIR, "vector"))
        
        keyword_index.set_index_id("keyword_index")
        keyword_index.storage_context.persist(persist_dir=os.path.join(PERSIST_DIR, "keyword"))

        logging.info("インデックスの更新が完了しました。")
    else:
        logging.info("変更がないため、既存のインデックスを読み込みます。")
        vector_index, keyword_index = load_existing_index()

    # メタデータを保存
    save_metadata(docs_list)

    return vector_index, keyword_index





def display_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        print("\nメタデータファイルが見つかりません。")

if __name__ == "__main__":
    # サマリーを生成せずに実行（デフォルト）
    # python vector_save.py --chunk-size 4096 --chunk-overlap 100
    # サマリーを生成する場合
    # python vector_save.py --summarize --chunk-size 4096 --chunk-overlap 100

    parser = argparse.ArgumentParser(description="Update vector index with customizable options")
    parser.add_argument("--summarize", action="store_true", help="Generate summaries for documents")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=40, help="Chunk overlap for text splitting")
    parser.add_argument("--force-update", action="store_true", help="Force update the index even if it exists")
    args = parser.parse_args()

    index = create_or_update_index(
        summarize=args.summarize,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_update=args.force_update
    )
    if index:
        print("インデックスの作成または更新が正常に完了しました。")
        display_metadata()
    else:
        print("インデックスの作成または更新に失敗しました。")
