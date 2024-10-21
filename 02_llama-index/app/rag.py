#rag.py
import logging
from typing import List, Tuple
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings
)
from llama_index.core.retrievers import VectorIndexAutoRetriever, RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.vector_stores.types import VectorStoreInfo, MetadataInfo
from common_setup import get_llm_and_embed_model
from vector_save import PERSIST_DIR, load_file_metadata, load_existing_index
from llama_index.core.extractors import KeywordExtractor


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_index_and_documents() -> Tuple[VectorStoreIndex, List[Document]]:
    vector_index = load_existing_index()
    if vector_index is None:
        raise ValueError("インデックスが見つかりません。vector_save.pyを実行してインデックスを作成してください。")
    
    documents = [Document.from_dict(node.dict()) for node in vector_index.docstore.docs.values()]
    return vector_index, documents

def create_auto_retriever(vector_index: VectorStoreIndex, similarity_top_k: int, use_keywords: bool, llm) -> VectorIndexAutoRetriever:
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
        keyword_extractor = KeywordExtractor(keywords=10, llm=llm)
        
        def get_extracted_keywords(query):
            logger.debug(f"Extracting keywords for query: {query}")
            try:
                extracted = keyword_extractor.extract([Document(text=query)])
                raw_keywords = extracted[0].get("excerpt_keywords", "").split(", ")
                
                # キーワードの後処理
                processed_keywords = [
                    kw for kw in raw_keywords 
                    if 2 < len(kw) < 20  # 長さの制限
                ]
                processed_keywords = list(set(processed_keywords))  # 重複の削除
                
                logger.debug(f"Extracted keywords: {processed_keywords}")
                return processed_keywords
            except Exception as e:
                logger.error(f"Error extracting keywords: {e}")
                return []
        
        retriever.get_extracted_keywords = get_extracted_keywords
        retriever.keyword_filter = lambda keywords: [kw for kw in keywords if len(kw) > 1]
        retriever.keyword_match_threshold = 0.5
        logger.info("Keyword extraction enabled for retriever")
    else:
        retriever.keyword_match_threshold = None
        logger.info("Keyword extraction disabled for retriever")

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
    
    llm.temperature = temperature
    system_prompt = f"{system_message}\n\nPlease respond in {language}."
    llm.system_prompt = system_prompt
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    file_metadata = load_file_metadata()
    
    for doc in documents:
        filename = doc.metadata.get('filename')
        if filename in file_metadata:
            doc.metadata['summary'] = file_metadata[filename].get('summary', '')

    if use_auto_retriever:
        retriever = create_auto_retriever(vector_index, similarity_top_k, use_keywords, llm)
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

    query_engine._custom_retriever = retriever
    
    logger.info("RAG system setup completed")

    return query_engine, file_metadata

def run_rag(
    query: str,
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
    logger.info("Starting RAG process")
    
    query_engine, file_metadata = setup_rag(
        vector_index,
        documents,
        similarity_top_k=similarity_top_k,
        response_mode=response_mode,
        structured_answer_filtering=structured_answer_filtering,
        use_auto_retriever=use_auto_retriever,
        use_keywords=use_keywords,
        system_message=system_message,
        language=language,
        temperature=temperature
    )

    response = query_engine.query(query)
    
    logger.info(f"Query: {query}")
    if use_keywords and hasattr(query_engine._custom_retriever, 'get_extracted_keywords'):
        extracted_keywords = query_engine._custom_retriever.get_extracted_keywords(query)
        logger.debug(f"Extracted keywords: {extracted_keywords}")
    logger.debug(f"Retrieved documents: {[doc.metadata for doc in response.source_nodes]}")
    logger.info(f"Response: {response}")

    return response, query_engine, file_metadata

