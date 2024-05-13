from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.query_engine.custom import STR_OR_RESPONSE_TYPE
from llama_index.core.retrievers import BaseRetriever
from typing import List
class BlogSummaryQueryEngine(CustomQueryEngine):
    docstore: SimpleDocumentStore
    response_synthesizer: BaseSynthesizer
    chunk_size: int
    chunk_overlap: int

    def custom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
        document = self.docstore.get_document(doc_id=query_str)
        nodes = [NodeWithScore(node=node, score=1.0) for node in
                 SentenceSplitter(chunk_size=self.chunk_size,
                                  chunk_overlap=self.chunk_overlap).get_nodes_from_documents(documents=[document])]
        query = 'Summarize the information.'
        query_bundle = QueryBundle(query_str=query)
        response = self.response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
            streaming=True
        )
        return response

class CustomRetriever(BaseRetriever):
    def __init__(
            self,docstore: SimpleDocumentStore, chunk_size:int, chunk_overlap:int

    ) -> None:
        """Init params."""

        self._docstore = docstore
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        document = self._docstore.get_document(doc_id=query_bundle.query_str)
        nodes = [NodeWithScore(node=node, score=1.0) for node in
                 SentenceSplitter(chunk_size=self.chunk_size,
                                  chunk_overlap=self.chunk_overlap,include_metadata=False).get_nodes_from_documents(documents=[document])]
        return nodes

