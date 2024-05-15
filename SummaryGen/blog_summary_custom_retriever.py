from typing import List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.storage.docstore import SimpleDocumentStore


class BlogCustomRetriever(BaseRetriever):
    def __init__(
            self, docstore: SimpleDocumentStore, chunk_size: int, chunk_overlap: int

    ) -> None:
        """Init params."""

        self._docstore = docstore
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
                    Initialize LLM observability with deepeval platform.

                    Parameters:

                    Returns:

                    Examples:

                    Notes:

                """
        """Retrieve nodes given query."""
        document = self._docstore.get_document(doc_id=query_bundle.query_str)
        nodes = [NodeWithScore(node=node, score=1.0) for node in
                 SentenceSplitter(chunk_size=self.chunk_size,
                                  chunk_overlap=self.chunk_overlap, include_metadata=False).get_nodes_from_documents(
                     documents=[document])]
        return nodes
