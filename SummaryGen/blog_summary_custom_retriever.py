from typing import List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.storage.docstore import SimpleDocumentStore


class BlogCustomRetriever(BaseRetriever):
    """
        A custom retriever for blog documents that extends the BaseRetriever class.
    """

    def __init__(
            self, docstore: SimpleDocumentStore, chunk_size: int, chunk_overlap: int

    ) -> None:
        """
            Initializes the BlogCustomRetriever with the specified document store, chunk size, and chunk overlap.

                Parameters:
                    docstore (SimpleDocumentStore): The document store to use for retrieving documents.
                    chunk_size (int): The size of the chunks into which the document text is split.
                    chunk_overlap (int): The number of words that will overlap between consecutive chunks.
        """

        self._docstore = docstore
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieves nodes from documents based on the specified query.

        This method implements the retrieval logic by fetching a document based on the query string,
        splitting the document into chunks, and returning these chunks as nodes.

        Parameters:
            query_bundle (QueryBundle): An object containing the query string and additional metadata
                                        like embeddings which can be used for more advanced retrieval strategies.

        Returns:
            List[NodeWithScore]: A list of nodes, each associated with a score indicating their relevance
                                 to the query. Currently, all nodes are scored as 1.0 indicating equal relevance.
        """
        document = self._docstore.get_document(doc_id=query_bundle.query_str)
        nodes = [NodeWithScore(node=node, score=1.0) for node in
                 SentenceSplitter(chunk_size=self.chunk_size,
                                  chunk_overlap=self.chunk_overlap, include_metadata=False).get_nodes_from_documents(
                     documents=[document])]
        return nodes
