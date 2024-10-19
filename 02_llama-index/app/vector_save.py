import argparse
import logging
import os
import sys
from datetime import datetime
import hashlib
import json
import pickle

from common_setup import get_llm_and_embed_model
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    SimpleKeywordTableIndex,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import get_response_synthesizer

# ログ設定
logging.basicConfig(level=logging.INFO)

# パス設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PERSIST_DIR = os.path.join(ROOT_DIR, "storage")
METADATA_FILE = os.path.join(PERSIST_DIR, "metadata.json")
DATA_DIR = os.path.join(ROOT_DIR, "data")
DOCUMENT_STORE_FILE = os.path.join(PERSIST_DIR, "document_store.pkl")

def generate_consistent_id(file_path, content):
    """ファイルパスとコンテンツに基づいて一貫したIDを生成"""
    return hashlib.md5((file_path + content).encode()).hexdigest()

def load_document_store():
    if os.path.exists(DOCUMENT_STORE_FILE):
        with open(DOCUMENT_STORE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_document_store(document_store):
    with open(DOCUMENT_STORE_FILE, 'wb') as f:
        pickle.dump(document_store, f)

def create_new_indexes():
    storage_context = StorageContext.from_defaults()
    vector_index = VectorStoreIndex([], storage_context=storage_context)
    vector_index.set_index_id("vector_index")
    
    keyword_index = SimpleKeywordTableIndex([], storage_context=storage_context)
    keyword_index.set_index_id("keyword_index")
    
    return vector_index, keyword_index

def create_or_update_index(summarize=False, chunk_size=1024, chunk_overlap=40, force_update=False):
    llm, embed_model = get_llm_and_embed_model()
    Settings.llm = llm
    Settings.embed_model = embed_model

    vector_index, keyword_index = load_or_create_indexes(force_update)
    document_store = load_document_store()
    file_metadata = load_file_metadata()

    # データディレクトリが存在しない場合は作成
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logging.warning(f"データディレクトリが存在しないため、作成しました: {DATA_DIR}")

    # 現在のファイルリストを取得
    current_files = set(os.path.relpath(os.path.join(root, file), DATA_DIR)
                        for root, _, files in os.walk(DATA_DIR)
                        for file in files)

    # データディレクトリが空かどうかチェック
    if not current_files:
        logging.warning(f"データディレクトリが空です: {DATA_DIR}")
        # 既存のインデックスとメタデータをクリア
        vector_index, keyword_index = create_new_indexes()
        document_store.clear()
        file_metadata.clear()
        save_document_store(document_store)
        save_file_metadata(file_metadata)
        return vector_index, keyword_index, file_metadata

    # 削除されたファイルを特定
    deleted_files = set(file_metadata.keys()) - current_files

    # 削除されたファイルの処理
    for deleted_file in deleted_files:
        logging.info(f"削除されたファイルを処理中: {deleted_file}")
        # このファイルに関連するドキュメントIDを見つける
        related_doc_ids = [doc_id for doc_id, doc in document_store.items()
                           if doc.metadata['filename'] == deleted_file]
        
        for doc_id in related_doc_ids:
            try:
                # インデックスから削除
                vector_index.delete_ref_doc(doc_id)
            except KeyError as e:
                logging.warning(f"ベクトルインデックスからの削除に失敗しました: {e}")
            except Exception as e:
                logging.error(f"ベクトルインデックスの削除中に予期しないエラーが発生しました: {e}")

            try:
                keyword_index.delete_ref_doc(doc_id)
            except KeyError as e:
                logging.warning(f"キーワードインデックスからの削除に失敗しました: {e}")
            except Exception as e:
                logging.error(f"キーワードインデックスの削除中に予期しないエラーが発生しました: {e}")

            # ドキュメントストアから削除
            document_store.pop(doc_id, None)
        
        # メタデータから削除
        file_metadata.pop(deleted_file, None)
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR, 
        recursive=True,
        file_metadata=lambda file_path: {
            "filename": os.path.relpath(file_path, DATA_DIR),
            "modification_time": os.path.getmtime(file_path)
        }
    )
    
    new_docs = reader.load_data()

    text_splitter = SentenceSplitter(
        separator='。',
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for doc in new_docs:
        filename = doc.metadata['filename']
        current_mod_time = doc.metadata['modification_time']
        
        if filename not in file_metadata or current_mod_time > file_metadata[filename]['modification_time'] or force_update:
            doc_id = generate_consistent_id(filename, doc.text)
            doc.id_ = doc_id

            if filename not in file_metadata:
                file_metadata[filename] = {
                    'modification_time': current_mod_time,
                    'chunks': 0,
                    'total_tokens': 0,
                    'summary': ''
                }
            
            if summarize and not file_metadata[filename]['summary']:
                file_metadata[filename]['summary'] = generate_summary(doc, llm)
                doc.metadata['summary'] = file_metadata[filename]['summary']

            vector_index.insert(doc, text_splitter=text_splitter)
            keyword_index.insert(doc)
            
            document_store[doc_id] = doc
            logging.info(f"Updated/Inserted document: {doc_id}")
            
            file_metadata[filename]['chunks'] += 1
            file_metadata[filename]['total_tokens'] += len(doc.text.split())
            file_metadata[filename]['modification_time'] = current_mod_time
        else:
            logging.info(f"Skipping unchanged file: {filename}")

    # インデックスの保存
    vector_index.storage_context.persist(persist_dir=os.path.join(PERSIST_DIR, "vector"))
    keyword_index.storage_context.persist(persist_dir=os.path.join(PERSIST_DIR, "keyword"))

    # ドキュメントストアの保存
    save_document_store(document_store)

    # ファイルメタデータの保存
    save_file_metadata(file_metadata)

    return vector_index, keyword_index, file_metadata

def load_file_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_file_metadata(file_metadata):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(file_metadata, f, ensure_ascii=False, indent=2)
    logging.info(f"ファイルメタデータが{METADATA_FILE}に保存されました。")


def load_or_create_indexes(force_update=False):
    if not force_update and os.path.exists(os.path.join(PERSIST_DIR, "vector", "docstore.json")) and \
       os.path.exists(os.path.join(PERSIST_DIR, "keyword", "docstore.json")):
        logging.info("既存のインデックスを読み込み中...")
        try:
            vector_storage_context = StorageContext.from_defaults(persist_dir=os.path.join(PERSIST_DIR, "vector"))
            vector_index = load_index_from_storage(vector_storage_context, index_id="vector_index")
            
            keyword_storage_context = StorageContext.from_defaults(persist_dir=os.path.join(PERSIST_DIR, "keyword"))
            keyword_index = load_index_from_storage(keyword_storage_context, index_id="keyword_index")
            
            logging.info("インデックスの読み込みが完了しました。")
        except Exception as e:
            logging.error(f"インデックスの読み込みエラー: {e}")
            vector_index, keyword_index = create_new_indexes()
    else:
        logging.info("新しいインデックスを作成します。")
        vector_index, keyword_index = create_new_indexes()

    return vector_index, keyword_index

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

def save_metadata(file_metadata):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(file_metadata, f, ensure_ascii=False, indent=2)
    logging.info(f"メタデータが{METADATA_FILE}に保存されました。")

def display_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print("ファイル名\t最終更新日時\tチャンク数\t合計トークン数\tサマリー")
        print("-" * 120)
        for filename, info in metadata.items():
            print(f"{filename}\t{datetime.fromtimestamp(info['modification_time']).strftime('%Y-%m-%d %H:%M:%S')}\t{info['chunks']}\t{info['total_tokens']}\t{info['summary'][:50]}...")
    else:
        print("\nメタデータファイルが見つかりません。")



def load_existing_index():
    """
    既存のベクトルインデックスとキーワードインデックスを読み込む関数
    この関数は外部のスクリプトから呼び出すことができます。
    """
    logging.info("既存のインデックスを読み込み中...")
    try:
        vector_storage_context = StorageContext.from_defaults(persist_dir=os.path.join(PERSIST_DIR, "vector"))
        vector_index = load_index_from_storage(vector_storage_context, index_id="vector_index")
        
        keyword_storage_context = StorageContext.from_defaults(persist_dir=os.path.join(PERSIST_DIR, "keyword"))
        keyword_index = load_index_from_storage(keyword_storage_context, index_id="keyword_index")
        
        logging.info("インデックスの読み込みが完了しました。")
        return vector_index, keyword_index
    except Exception as e:
        logging.error(f"インデックスの読み込みエラー: {e}")
        return None, None


if __name__ == "__main__":
    # サマリーを生成せずに実行（デフォルト）
    # python app\vector_save.py --chunk-size 4096 --chunk-overlap 100
    # サマリーを生成する場合
    # python app\vector_save.py --summarize --chunk-size 4096 --chunk-overlap 100


    parser = argparse.ArgumentParser(description="Update vector index with customizable options")
    parser.add_argument("--summarize", action="store_true", help="Generate summaries for documents")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=40, help="Chunk overlap for text splitting")
    parser.add_argument("--force-update", action="store_true", help="Force update the index even if it exists")
    args = parser.parse_args()

    try:
        vector_index, keyword_index, file_metadata = create_or_update_index(
            summarize=args.summarize,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            force_update=args.force_update
        )
        if vector_index and keyword_index:
            print("インデックスの作成または更新が正常に完了しました。")
            save_metadata(file_metadata)
            display_metadata()
        else:
            print("インデックスの作成または更新に失敗しました。")
    except Exception as e:
        logging.error(f"予期しないエラーが発生しました: {e}")
        raise
