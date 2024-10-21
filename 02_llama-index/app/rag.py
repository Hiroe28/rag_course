#rag.py
import logging
from typing import List, Dict, Any
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage,
    QueryBundle
)
from llama_index.core.retrievers import VectorIndexAutoRetriever, RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.vector_stores.types import VectorStoreInfo, MetadataInfo
from common_setup import get_llm_and_embed_model
from vector_save import PERSIST_DIR, load_file_metadata, load_existing_index
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_existing_index():
    logger.info("Loading existing index...")
    try:
        storage_context = StorageContext.from_defaults(persist_dir=os.path.join(PERSIST_DIR, "vector"))
        return load_index_from_storage(storage_context, index_id="vector_index")
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        return None

def create_auto_retriever(vector_index: VectorStoreIndex, similarity_top_k: int, use_keywords: bool) -> VectorIndexAutoRetriever:
    vector_store_info = VectorStoreInfo(
        content_info="Document content related to various topics",
        metadata_info=[
            MetadataInfo(name="filename", type="str", description="Name of the source file"),
            MetadataInfo(name="modification_time", type="float", description="Last modification time of the file"),
            MetadataInfo(name="summary", type="str", description="Summary of the document content"),
            MetadataInfo(name="keywords", type="list", description="Keywords extracted from the document")
        ]
    )

    retriever = VectorIndexAutoRetriever(
        index=vector_index,
        vector_store_info=vector_store_info,
        similarity_top_k=similarity_top_k
    )

    if use_keywords:
        retriever.keyword_match_threshold = 0.5  # この値は調整可能
        retriever.keyword_extraction_threshold = 0.7  # キーワード抽出のしきい値を設定
    else:
        retriever.keyword_match_threshold = None
        retriever.keyword_extraction_threshold = None
    return retriever

def create_recursive_retriever(vector_index: VectorStoreIndex, documents: List[Document], similarity_top_k: int) -> RecursiveRetriever:
    top_retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)

    doc_retrievers = {}
    for doc in documents:
        doc_index = VectorStoreIndex([doc])
        doc_retrievers[doc.id_] = doc_index.as_retriever(similarity_top_k=max(1, similarity_top_k // 2))

    return RecursiveRetriever(
        "vector",
        retriever_dict={"vector": top_retriever, **doc_retrievers},
        verbose=True,
    )

def setup_rag(
    vector_index: VectorStoreIndex,
    documents: List[Document],
    similarity_top_k: int = 5,
    response_mode: ResponseMode = ResponseMode.COMPACT,
    structured_answer_filtering: bool = True,
    use_auto_retriever: bool = True,
    use_keywords: bool = False,
    system_message: str = "",
    language: str = "Japanese",
    temperature: float = 0.0
):
    logger.info("Setting up RAG system")
    llm, embed_model = get_llm_and_embed_model()
    
    # Update LLM settings
    llm.temperature = temperature
    system_prompt = f"{system_message}\n\nPlease respond in {language}."
    llm.system_prompt = system_prompt
    
    Settings.llm = llm
    Settings.embed_model = embed_model

    file_metadata = load_file_metadata()

    if use_auto_retriever:
        retriever = create_auto_retriever(vector_index, similarity_top_k, use_keywords)
    else:
        retriever = create_recursive_retriever(vector_index, documents, similarity_top_k)

    response_synthesizer = get_response_synthesizer(
        response_mode=response_mode,
        structured_answer_filtering=structured_answer_filtering
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    logger.info("RAG system setup completed")
    return query_engine, file_metadata

def run_rag(
    query: str,
    similarity_top_k: int = 5,
    response_mode: ResponseMode = ResponseMode.COMPACT,
    structured_answer_filtering: bool = True,
    use_auto_retriever: bool = False,
    use_keywords: bool = False  # 新しいパラメータ
):
    vector_index = load_existing_index()
    if vector_index is None:
        logger.error("Failed to load existing index. Please run vector_save.py first.")
        return None, None

    documents = [Document.from_dict(node.dict()) for node in vector_index.docstore.docs.values()]
    query_engine, file_metadata = setup_rag(
        vector_index,
        documents,
        similarity_top_k=similarity_top_k,
        response_mode=response_mode,
        structured_answer_filtering=structured_answer_filtering,
        use_auto_retriever=use_auto_retriever,
        use_keywords=use_keywords  # 新しいパラメータを渡す
    )
    response = query_engine.query(query)
    
    logger.info(f"Query: {query}")
    logger.info(f"Response: {response}")

    return response, file_metadata

def get_document_summaries(index):
    summaries = []
    for node in index.docstore.docs.values():
        if 'summary' in node.metadata:
            summaries.append({
                'filename': node.metadata.get('filename', 'Unknown'),
                'summary': node.metadata['summary']
            })
    return summaries

if __name__ == "__main__":
    query = "文書の主要なトピックは何ですか？"
    use_auto_retriever = True
    use_keywords = True  # キーワードを使用するかどうかを指定
    response, file_metadata = run_rag(
        query, 
        use_auto_retriever=use_auto_retriever,
        use_keywords=use_keywords
    )
    
    if response:
        print(f"Query: {query}")
        print(f"Response: {response}")
        print(f"\nRetrieval Method: {'Auto' if use_auto_retriever else 'Recursive'}")
        print(f"Using Keywords: {'Yes' if use_keywords else 'No'}")
        print("\nDocument Summaries:")
        for filename, info in file_metadata.items():
            print(f"Filename: {filename}")
            print(f"Summary: {info['summary']}")
            print(f"Chunks: {info['chunks']}")
            print(f"Total characters: {info['total_chars']}")
            print("-" * 50)
    else:
        print("Failed to run RAG. Please ensure the vector index is created.")
