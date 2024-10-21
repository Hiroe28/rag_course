#vector_save.py
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
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
    Document
)
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import get_response_synthesizer

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def create_new_index():
    storage_context = StorageContext.from_defaults()
    vector_index = VectorStoreIndex([], storage_context=storage_context)
    vector_index.set_index_id("vector_index")
    return vector_index

def create_or_update_index(summarize=False, extract_keywords=False, num_keywords=10, chunk_size=1024, chunk_overlap=40, force_update=False):
    llm, embed_model = get_llm_and_embed_model()
    Settings.llm = llm
    Settings.embed_model = embed_model

    vector_index = load_or_create_index(force_update)
    document_store = load_document_store()
    file_metadata = load_file_metadata()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.warning(f"データディレクトリが存在しないため、作成しました: {DATA_DIR}")

    current_files = set(os.path.relpath(os.path.join(root, file), DATA_DIR)
                        for root, _, files in os.walk(DATA_DIR)
                        for file in files)

    if not current_files:
        logger.warning(f"データディレクトリが空です: {DATA_DIR}")
        vector_index = create_new_index()
        document_store.clear()
        file_metadata.clear()
        save_document_store(document_store)
        save_file_metadata(file_metadata)
        return vector_index, file_metadata

    deleted_files = set(file_metadata.keys()) - current_files

    for deleted_file in deleted_files:
        logger.info(f"削除されたファイルを処理中: {deleted_file}")
        related_doc_ids = [doc_id for doc_id, doc in document_store.items()
                           if doc.metadata['filename'] == deleted_file]
        
        for doc_id in related_doc_ids:
            try:
                vector_index.delete_ref_doc(doc_id)
            except KeyError as e:
                logger.warning(f"ベクトルインデックスからの削除に失敗しました: {e}")
            except Exception as e:
                logger.error(f"ベクトルインデックスの削除中に予期しないエラーが発生しました: {e}")

            document_store.pop(doc_id, None)
        
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

    # ファイル名ごとにドキュメントをグループ化
    docs_by_filename = {}
    for doc in new_docs:
        filename = doc.metadata['filename']
        if filename not in docs_by_filename:
            docs_by_filename[filename] = doc
        else:
            # 既存のドキュメントとマージする
            existing_doc = docs_by_filename[filename]
            existing_doc.text += "\n\n" + doc.text
            existing_doc.metadata['num_sections'] = existing_doc.metadata.get('num_sections', 1) + 1
            # logger.info(f"Merged duplicate section for filename: {filename}. Total sections: {existing_doc.metadata['num_sections']}")

    text_splitter = SentenceSplitter(
        separator='。',
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    keyword_extractor = KeywordExtractor(keywords=num_keywords, llm=llm) if extract_keywords else None

    for filename, doc in docs_by_filename.items():
        current_mod_time = doc.metadata['modification_time']
        
        if filename not in file_metadata or current_mod_time > file_metadata[filename]['modification_time'] or force_update:
            if filename not in file_metadata:
                file_metadata[filename] = {
                    'modification_time': current_mod_time,
                    'chunks': 0,
                    'total_chars': 0,
                    'summary': '',
                    'keywords': []
                }
            
            if summarize and not file_metadata[filename]['summary']:
                file_metadata[filename]['summary'] = generate_summary(doc, llm)
                doc.metadata['summary'] = file_metadata[filename]['summary']

            if extract_keywords:
                keywords = keyword_extractor.extract([doc])[0]
                file_metadata[filename]['keywords'] = keywords
                doc.metadata['keywords'] = keywords

            # ドキュメントをチャンクに分割
            chunks = text_splitter.split_text(doc.text)
            num_chunks = len(chunks)
            total_chars = sum(len(chunk) for chunk in chunks)  # 文字数で計算

            # 各チャンクをノードとして作成
            nodes = []
            for chunk in chunks:
                node_id = generate_consistent_id(filename, chunk)
                node = Document(text=chunk, metadata=doc.metadata, id_=node_id)
                nodes.append(node)
                document_store[node_id] = node  # document_storeを更新

            # ノードをインデックスに挿入
            vector_index.insert_nodes(nodes)

            logger.info(f"Updated/Inserted document: {filename} with {num_chunks} chunks")

            # メタデータを更新
            file_metadata[filename]['chunks'] = num_chunks
            file_metadata[filename]['total_chars'] = total_chars  # 文字数を保存
            file_metadata[filename]['modification_time'] = current_mod_time
        else:
            logger.info(f"Skipping unchanged file: {filename}")

    vector_index.storage_context.persist(persist_dir=os.path.join(PERSIST_DIR, "vector"))

    save_document_store(document_store)
    save_file_metadata(file_metadata)

    return vector_index, file_metadata

def load_file_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_file_metadata(file_metadata):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(file_metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"ファイルメタデータが{METADATA_FILE}に保存されました。")

def load_or_create_index(force_update=False):
    if not force_update and os.path.exists(os.path.join(PERSIST_DIR, "vector", "docstore.json")):
        logger.info("既存のインデックスを読み込み中...")
        try:
            vector_storage_context = StorageContext.from_defaults(persist_dir=os.path.join(PERSIST_DIR, "vector"))
            vector_index = load_index_from_storage(vector_storage_context, index_id="vector_index")
            logger.info("インデックスの読み込みが完了しました。")
        except Exception as e:
            logger.error(f"インデックスの読み込みエラー: {e}")
            vector_index = create_new_index()
    else:
        logger.info("新しいインデックスを作成します。")
        vector_index = create_new_index()

    return vector_index

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


def display_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print("ファイル名\t最終更新日時\tチャンク数\t合計文字数\tサマリー\tキーワード")
        print("-" * 140)
        for filename, info in metadata.items():
            keywords = info.get('keywords', [])
            if isinstance(keywords, dict):
                # キーワードが辞書型の場合、キーを使用
                keywords = list(keywords.keys())
            elif not isinstance(keywords, list):
                # リストでない場合、空リストに設定
                keywords = []
            
            keyword_str = ", ".join(keywords[:5])  # 最初の5つのキーワードのみ表示
            summary = info.get('summary', '')[:50] + "..." if info.get('summary') else ''
            
            print(f"{filename}\t{datetime.fromtimestamp(info['modification_time']).strftime('%Y-%m-%d %H:%M:%S')}\t{info.get('chunks', 0)}\t{info.get('total_chars', 0)}\t{summary}\t{keyword_str}")
    else:
        print("\nメタデータファイルが見つかりません。")

def load_existing_index():
    """既存のベクトルインデックスを読み込む関数"""
    logger.info("既存のインデックスを読み込み中...")
    try:
        vector_storage_context = StorageContext.from_defaults(persist_dir=os.path.join(PERSIST_DIR, "vector"))
        vector_index = load_index_from_storage(vector_storage_context, index_id="vector_index")
        logger.info("インデックスの読み込みが完了しました。")
        return vector_index
    except Exception as e:
        logger.error(f"インデックスの読み込みエラー: {e}")
        return None



if __name__ == "__main__":
    # このプログラムは、テキストデータをベクトルインデックスとして保存・更新します。
    # 使用可能な引数とその説明、およびコマンドの例は以下の通りです。

    # 1. --summarize
    #    ドキュメントのサマリー（要約）を生成するかどうかを指定します。
    #    デフォルトではサマリーは生成されませんが、このフラグを追加することで、
    #    ドキュメントの内容を簡潔に要約します。
    #    例:
    #        python app\vector_save.py --summarize --chunk-size 4096 --chunk-overlap 100
    #    この例では、サマリーを生成し、チャンクサイズを4096、チャンクオーバーラップを100に設定しています。
    #
    # 2. --chunk-size
    #    ドキュメントをチャンク（分割された小さな部分）に分割する際のサイズを指定します。
    #    チャンクのサイズは文字数で指定されます。デフォルト値は1024です。
    # 3. --chunk-overlap
    #    チャンクを分割する際に、隣り合うチャンクの重なり部分のサイズを指定します。
    #    これにより、文脈を保ちながらテキストを分割できます。デフォルト値は40です。
    #    例:
    #        python app\vector_save.py --chunk-size 4096 --chunk-overlap 100
    #    この例では、チャンクサイズを2048に設定して実行します。
    #
    # 4. --force-update
    #    既存のインデックスがあっても、それを無視して強制的に再作成・更新するオプションです。
    #    これを使うと、以前と同じファイルがあっても再度インデックスを更新します。
    #    例:
    #        python app\vector_save.py --force-update --chunk-size 4096 --chunk-overlap 100
    #    この例では、インデックスの再作成を強制し、チャンクサイズとオーバーラップを指定して実行します。


    parser = argparse.ArgumentParser(description="Update vector index with customizable options")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=40, help="Chunk overlap for text splitting")
    parser.add_argument("--extract-keywords", action="store_true", help="Extract keywords from documents")
    parser.add_argument("--num-keywords", type=int, default=10, help="Number of keywords to extract")
    parser.add_argument("--summarize", action="store_true", help="Generate summaries for documents")
    parser.add_argument("--force-update", action="store_true", help="Force update the index even if it exists")
    args = parser.parse_args()

    try:
        vector_index, file_metadata = create_or_update_index(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            extract_keywords=args.extract_keywords,
            num_keywords=args.num_keywords,
            summarize=args.summarize,
            force_update=args.force_update
        )
        if vector_index:
            print("インデックスの作成または更新が正常に完了しました。")
            save_file_metadata(file_metadata)
            display_metadata()
        else:
            print("インデックスの作成または更新に失敗しました。")
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        raise
