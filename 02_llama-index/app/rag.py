# rag.py
import logging
from typing import Dict, List

from llama_index.core import QueryBundle, SimpleKeywordTableIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.schema import NodeWithScore

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)

class ImprovedCustomRetriever(BaseRetriever):
    
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_index: SimpleKeywordTableIndex,
        mode: str = "OR"
    ):
        self._vector_retriever = vector_retriever
        self._keyword_index = keyword_index
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        logger.info(f"Retrieving for query: {query_bundle.query_str}")
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_retriever = self._keyword_index.as_retriever(similarity_top_k=5)
        keyword_nodes = keyword_retriever.retrieve(query_bundle)

        logger.debug(f"Vector nodes count: {len(vector_nodes)}")
        logger.debug(f"Keyword nodes count: {len(keyword_nodes)}")

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict: Dict[str, NodeWithScore] = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
            if not retrieve_ids:
                logger.info("No nodes found in AND mode, falling back to OR mode")
                retrieve_ids = vector_ids.union(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        logger.info(f"Total nodes retrieved: {len(retrieve_nodes)}")
        return retrieve_nodes

def setup_rag(
    vector_index,
    keyword_index,
    similarity_top_k: int = 5,
    response_mode: ResponseMode = ResponseMode.COMPACT,
    structured_answer_filtering: bool = True,
    hybrid_mode: str = "OR"
):
    logger.info("Setting up RAG system")
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=similarity_top_k)
    custom_retriever = ImprovedCustomRetriever(vector_retriever, keyword_index, mode=hybrid_mode)

    response_synthesizer = get_response_synthesizer(
        response_mode=response_mode,
        structured_answer_filtering=structured_answer_filtering
    )

    query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )

    logger.info("RAG system setup completed")
    return query_engine

def get_document_summaries(index):
    """インデックス内のドキュメントのサマリーを取得する関数"""
    summaries = []
    for node in index.docstore.docs.values():
        if 'summary' in node.metadata:
            summaries.append({
                'filename': node.metadata.get('filename', 'Unknown'),
                'summary': node.metadata['summary']
            })
    return summaries

