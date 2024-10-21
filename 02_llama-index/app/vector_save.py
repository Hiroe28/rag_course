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
    VectorStoreIndex,
    load_index_from_storage,
    Document
)
from llama_index.core.extractors import KeywordExtractor, SummaryExtractor
from llama_index.core.node_parser import SentenceSplitter


# ログ設定
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# パス設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PERSIST_DIR = os.path.join(ROOT_DIR, "storage")
METADATA_FILE = os.path.join(PERSIST_DIR, "metadata.json")
DATA_DIR = os.path.join(ROOT_DIR, "data")
DOCUMENT_STORE_FILE = os.path.join(PERSIST_DIR, "document_store.pkl")

def generate_consistent_id(file_path, content):
    # ファイルパスとコンテンツに基づいて一貫したIDを生成
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

def create_or_update_index(summarize=False, summary_level='file', extract_keywords=False, num_keywords=10, chunk_size=1024, chunk_overlap=40, force_update=False):
    # LLMと埋め込みモデルを取得
    llm, embed_model = get_llm_and_embed_model()
    Settings.llm = llm
    Settings.embed_model = embed_model

    # インデックスを読み込むか新規作成
    vector_index = load_or_create_index(force_update)
    # ドキュメントストアとファイルメタデータを読み込む
    document_store = load_document_store()
    file_metadata = load_file_metadata()

    # データディレクトリの存在を確認
    ensure_data_directory()
    # 現在のファイルリストを取得
    current_files = get_current_files()
    
    # データディレクトリが空の場合、初期化して終了
    if not current_files:
        return handle_empty_data_directory(vector_index, document_store, file_metadata)

    # 削除されたファイルを処理
    handle_deleted_files(vector_index, document_store, file_metadata, current_files)
    
    # 新しいドキュメントを読み込む
    new_docs = load_new_documents()
    # ファイル名でドキュメントをグループ化
    docs_by_filename = group_documents_by_filename(new_docs)

    # テキスト分割器を作成
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    # キーワード抽出器を作成（必要な場合）
    keyword_extractor = create_keyword_extractor(extract_keywords, num_keywords, llm)

    # 各ドキュメントを処理
    for filename, doc in docs_by_filename.items():
        process_document(filename, doc, file_metadata, document_store, vector_index, text_splitter, 
                         keyword_extractor, summarize, summary_level, force_update, llm)

    # データを永続化
    persist_data(vector_index, document_store, file_metadata)

    # 更新されたインデックスとファイルメタデータを返す
    return vector_index, file_metadata

def ensure_data_directory():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.warning(f"データディレクトリが存在しないため、作成しました: {DATA_DIR}")

def get_current_files():
    return set(os.path.relpath(os.path.join(root, file), DATA_DIR)
               for root, _, files in os.walk(DATA_DIR)
               for file in files)

def handle_empty_data_directory(vector_index, document_store, file_metadata):
    logger.warning(f"データディレクトリが空です: {DATA_DIR}")
    vector_index = create_new_index()
    document_store.clear()
    file_metadata.clear()
    save_document_store(document_store)
    save_file_metadata(file_metadata)
    return vector_index, file_metadata

def handle_deleted_files(vector_index, document_store, file_metadata, current_files):
    deleted_files = set(file_metadata.keys()) - current_files
    for deleted_file in deleted_files:
        logger.info(f"削除されたファイルを処理中: {deleted_file}")
        delete_file_from_index(vector_index, document_store, deleted_file)
        file_metadata.pop(deleted_file, None)

def delete_file_from_index(vector_index, document_store, filename):
    related_doc_ids = [doc_id for doc_id, doc in document_store.items()
                       if doc.metadata['filename'] == filename]
    for doc_id in related_doc_ids:
        try:
            vector_index.delete_ref_doc(doc_id)
        except KeyError as e:
            logger.warning(f"ベクトルインデックスからの削除に失敗しました: {e}")
        except Exception as e:
            logger.error(f"ベクトルインデックスの削除中に予期しないエラーが発生しました: {e}")
        document_store.pop(doc_id, None)

def load_new_documents():
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR, 
        recursive=True,
        file_metadata=lambda file_path: {
            "filename": os.path.relpath(file_path, DATA_DIR),
            "modification_time": os.path.getmtime(file_path)
        }
    )
    return reader.load_data()

def group_documents_by_filename(docs):
    grouped_docs = {}
    for doc in docs:
        filename = doc.metadata['filename']
        if filename not in grouped_docs:
            grouped_docs[filename] = doc
        else:
            existing_doc = grouped_docs[filename]
            existing_doc.text += "\n\n" + doc.text
            existing_doc.metadata['num_sections'] = existing_doc.metadata.get('num_sections', 1) + 1
    return grouped_docs

def create_text_splitter(chunk_size, chunk_overlap):
    return SentenceSplitter(
        separator='。',
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

def create_keyword_extractor(extract_keywords, num_keywords, llm):
    return KeywordExtractor(keywords=num_keywords, llm=llm) if extract_keywords else None

def process_document(filename, doc, file_metadata, document_store, vector_index, text_splitter, 
                     keyword_extractor, summarize, summary_level, force_update, llm):
    current_mod_time = doc.metadata['modification_time']
    
    if should_process_document(filename, current_mod_time, file_metadata, force_update):
        initialize_file_metadata(filename, current_mod_time, file_metadata)
        chunks = text_splitter.split_text(doc.text)
        
        if summarize:
            generate_summaries(filename, doc, chunks, file_metadata, summary_level, llm)
        
        if keyword_extractor:
            extract_and_store_keywords(filename, doc, file_metadata, keyword_extractor)
        
        nodes = create_nodes(filename, doc, chunks, file_metadata, summarize, summary_level, document_store)
        
        vector_index.insert_nodes(nodes)
        
        update_file_metadata(filename, file_metadata, chunks, current_mod_time)
        logger.info(f"Updated/Inserted document: {filename} with {len(chunks)} chunks")
    else:
        logger.info(f"Skipping unchanged file: {filename}")

def should_process_document(filename, current_mod_time, file_metadata, force_update):
    return (filename not in file_metadata or 
            current_mod_time > file_metadata[filename]['modification_time'] or 
            force_update)

def initialize_file_metadata(filename, current_mod_time, file_metadata):
    if filename not in file_metadata:
        file_metadata[filename] = {
            'modification_time': current_mod_time,
            'chunks': 0,
            'total_chars': 0,
            'summary': '',
            'keywords': []
        }

def generate_summaries(filename, doc, chunks, file_metadata, summary_level, llm):
    if summary_level == 'file':
        file_metadata[filename]['summary'] = generate_summary(doc, llm)
        doc.metadata['summary'] = file_metadata[filename]['summary']
    elif summary_level == 'chunk':
        file_metadata[filename]['chunk_summaries'] = [generate_summary(chunk, llm) for chunk in chunks]

def extract_and_store_keywords(filename, doc, file_metadata, keyword_extractor):
    keywords = keyword_extractor.extract([doc])[0]
    file_metadata[filename]['keywords'] = list(keywords.values())[0] if isinstance(keywords, dict) else keywords
    doc.metadata['keywords'] = file_metadata[filename]['keywords']

def create_nodes(filename, doc, chunks, file_metadata, summarize, summary_level, document_store):
    nodes = []
    for i, chunk in enumerate(chunks):
        chunk_id = generate_consistent_id(filename, chunk)
        chunk_metadata = doc.metadata.copy()
        
        if summarize and summary_level == 'chunk':
            chunk_metadata['chunk_summary'] = file_metadata[filename]['chunk_summaries'][i]
        
        node = Document(text=chunk, metadata=chunk_metadata, id_=chunk_id)
        nodes.append(node)
        document_store[chunk_id] = node
    return nodes

def update_file_metadata(filename, file_metadata, chunks, current_mod_time):
    file_metadata[filename]['chunks'] = len(chunks)
    file_metadata[filename]['total_chars'] = sum(len(chunk) for chunk in chunks)
    file_metadata[filename]['modification_time'] = current_mod_time

def persist_data(vector_index, document_store, file_metadata):
    vector_index.storage_context.persist(persist_dir=os.path.join(PERSIST_DIR, "vector"))
    save_document_store(document_store)
    save_file_metadata(file_metadata)

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

def generate_summary(text, llm, max_length=100):
    prompt_template = (
        f"以下の文書の要約を{max_length}字程度で作成してください。"
        "重要なポイントを簡潔に説明してください：\n\n{{context_str}}"
    )
    
    extractor = SummaryExtractor(
        llm=llm,
        summaries=["self"],
        prompt_template=prompt_template
    )
    
    doc = Document(text=text) if isinstance(text, str) else text
    metadata_list = extractor.extract([doc])
    
    if metadata_list and "section_summary" in metadata_list[0]:
        return metadata_list[0]["section_summary"]
    else:
        return "要約の生成に失敗しました。"

def display_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print("ファイル名\t最終更新日時\tチャンク数\t合計文字数\tサマリー\tキーワード\tチャンクサマリー")
        print("-" * 160)
        for filename, info in metadata.items():
            keywords = info.get('keywords', [])
            if isinstance(keywords, dict):
                keywords = list(keywords.keys())
            elif not isinstance(keywords, list):
                keywords = []
            
            keyword_str = ", ".join(keywords[:5])
            summary = info.get('summary', '')[:50] + "..." if info.get('summary') else ''
            
            # チャンクサマリーの表示（最初の3つのみ）
            chunk_summaries = info.get('chunk_summaries', [])
            chunk_summary_str = "; ".join(chunk_summaries[:3]) + "..." if chunk_summaries else 'なし'
            
            print(f"{filename}\t{datetime.fromtimestamp(info['modification_time']).strftime('%Y-%m-%d %H:%M:%S')}\t{info.get('chunks', 0)}\t{info.get('total_chars', 0)}\t{summary}\t{keyword_str}\t{chunk_summary_str}")
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
    parser = argparse.ArgumentParser(description="Update vector index with customizable options")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=40, help="Chunk overlap for text splitting")
    parser.add_argument("--extract-keywords", action="store_true", help="Extract keywords from documents")
    parser.add_argument("--num-keywords", type=int, default=10, help="Number of keywords to extract")
    parser.add_argument("--summarize", action="store_true", help="Generate summaries for documents")
    parser.add_argument("--summary-level", choices=['file', 'chunk'], default='file', 
                        help="Level of summarization: 'file' for file-level summaries, 'chunk' for chunk-level summaries")
    parser.add_argument("--force-update", action="store_true", help="Force update the index even if it exists")
    args = parser.parse_args()

    try:
        vector_index, file_metadata = create_or_update_index(
            summarize=args.summarize,
            summary_level=args.summary_level,
            extract_keywords=args.extract_keywords,
            num_keywords=args.num_keywords,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
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
