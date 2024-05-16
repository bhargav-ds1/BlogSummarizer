from llama_index.core import Settings, StorageContext
from typing import List, Union
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer, BaseSynthesizer
from llama_index.core.indices.prompt_helper import PromptHelper
from SummaryGen.fetch_blogs import FetchBlogs
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
import llama_index.core.query_engine as qe
from llama_index.core.base.response.schema import StreamingResponse, Response
from Observability import InitializeObservability
from dotenv import load_dotenv
import os
from SummaryGen.blog_summary_custom_retriever import BlogCustomRetriever
from SummaryGen.llm_model_provider import LLMProvider


class DocumentSummaryGenerator:
    """
    A class to generate summaries for documents fetched from blogs based on various response modes and configurations.

    Attributes:
    - refetch_blogs (bool): Whether to refetch the blogs from the source.
    - output_dir (str): Directory where the output and documents are stored.
    - summary_template_str (str): Prompt template string for generating summaries.
    - chunk_size (int): Size of the text chunk to process at one time.
    - chunk_overlap (int): Overlap between consecutive chunks.
    - streaming (bool): If True, enables streaming mode for response. Response will be streamed.
    - use_async (bool): If True, enables asynchronous response synthesis.
    - observ_provider (str): The provider for observability features.

    Constructor Parameters:
    - llm_args (dict, optional): Arguments to configure the language model.
    - refetch_blogs (bool, optional): Flag to refetch blogs, defaults to False.
    - output_dir (str, optional): Output directory path.
    - query_engine_type (str, optional): Type of query engine to use.
    - query_engine_kwargs (dict, optional): Additional kwargs for the query engine.
    - response_mode (str, optional): Mode of response handling, defaults to 'tree_summarize'. 'simple_summarize' can
    also be used as an alternative strategy.
    - chunk_size (int, optional): Chunk size for processing, defaults to 1024.
    - chunk_overlap (int, optional): Overlap size between chunks, defaults to 128.
    - streaming (bool, optional): Enable streaming mode, defaults to False.
    - summary_template_str (str, optional): Summary template string.
    - use_async (bool, optional): Enable asynchronous mode for LLM call during response synthesis, defaults to False.
    - observ_provider (str, optional): Observability provider, defaults to 'phoenix'.

    Examples:
    # Initialize the document summary generator with custom settings
    generator = DocumentSummaryGenerator({
        'model_name': 'gpt-3',
        'temperature': 0.7
    }, output_dir='/path/to/output', summary_template_str='Summarize: {content}')

    Notes:
    This class integrates various components like LLM, document retrieval, and query engines to provide a seamless
    document summarization experience. Ensure that all dependencies and environment variables are properly set up.
    """

    def __init__(self, llm_args: dict = None,
                 refetch_blogs: bool = False, output_dir: str = None, query_engine_type: str = None,
                 query_engine_kwargs: dict = None, response_mode: str = 'tree_summarize',
                 chunk_size: int = 1024, chunk_overlap: int = 128,
                 streaming: bool = False, summary_template_str: str = None, use_async: bool = False,
                 observ_provider: str = 'phoenix') -> None:
        super().__init__()
        root_dir = os.path.dirname(os.path.dirname(__file__))
        load_dotenv(root_dir + '/.envfile')
        self.observability = InitializeObservability(observ_provider=observ_provider)
        self.blog_fetcher = FetchBlogs()
        self.refetch_blogs = refetch_blogs
        self.output_dir = os.path.join(root_dir, output_dir)
        self.summary_template_str = summary_template_str
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.streaming = streaming
        ##############################
        self.llm = LLMProvider(**llm_args).get_llm_model()
        Settings.llm = self.llm
        ##############################
        try:
            self.response_mode = ResponseMode(response_mode)
        except Exception as e:
            print('Invalid Response mode:' + str(e))
        self.use_async = use_async
        self.response_synthesizer = self.get_response_synthesizer()
        ##############################
        self.docstore = self.get_documents()

        self.retriever = BlogCustomRetriever(docstore=self.docstore, chunk_size=self.chunk_size,
                                             chunk_overlap=self.chunk_overlap)
        if hasattr(qe, query_engine_type):
            self.query_engine_type = getattr(qe, query_engine_type)
        else:
            raise ModuleNotFoundError('The specified type of query engine is not available')
        try:
            self.query_engine = self.query_engine_type(response_synthesizer=self.response_synthesizer,
                                                       retriever=self.retriever)
        except Exception as e:
            print('Exception occured while creating the specified query engine:' + str(e))

    def get_response_synthesizer(self) -> BaseSynthesizer:
        """
            Returns the response synthesizer object by equipping it with the provided summary template, response mode,
            and limits the context sizes to the upper limit of the LLM context.

            Returns:
                - response synthesizer of type BaseSynthesizer
            Notes:
                - Different response modes can be used which changes the response created by the LLM. For the purpose
                of generating summaries (simple_summarize and tree_summarize) response modes can be helpful.

        """
        query_template_str = self.summary_template_str
        query_template = SelectorPromptTemplate(
            default_template=PromptTemplate(
                query_template_str, prompt_type=PromptType.SUMMARY
            ),
        )
        response_synthesizer = get_response_synthesizer(response_mode=self.response_mode,
                                                        summary_template=query_template,
                                                        prompt_helper=PromptHelper.from_llm_metadata(self.llm.metadata,
                                                                                                     chunk_size_limit=self.llm.metadata.context_window - 1000),
                                                        verbose=True, streaming=self.streaming,
                                                        use_async=self.use_async)
        return response_synthesizer

    def get_documents(self) -> SimpleDocumentStore:
        """
            Gets the blogs as documents and returns a SimpleDocumentStore object.

            Returns:
                - docstore:SimpleDocumentStore which contains the blogs as documents.
            Notes:
                - While many advanced document stores could be used for storing and retrieving the documents.
                A SimpleDocumentStore suffices the purpose of blog summary generation as no complex retrieval strategies
                are required.

        """
        if not os.path.exists(self.output_dir + '/docstore.json') or self.refetch_blogs:
            print('Fetching Blogs ...')
            blogs = self.blog_fetcher.fetch_blogs()
            docstore = SimpleDocumentStore()
            docstore.add_documents(blogs)
            StorageContext.from_defaults(docstore=docstore).persist(self.output_dir)
        else:
            print('Using stored blogs content')
            docstore = SimpleDocumentStore().from_persist_dir(self.output_dir)

            return docstore

    def get_titles(self) -> List[str]:
        """
            Returns the keys of the documents as the titles of the blogs.
            Returns:
                - list of blog titles
            Notes:
                - Depends on the get_documents and __init__ functions which gets and sets the docstore as a
                SimpleDocumentStore value.
                respectively.

        """
        return list(self.docstore.docs.keys())

    def get_summary_response(self, doc_id: str) -> Union[StreamingResponse, Response]:
        """
            queries the query_engine with the title of the blog to generate the response object containing the summary.
            Parameters:
                - id of the document which is the title of the blog.

            Returns:
                - response object containing the response from the LLM. It can be either streaming or normal response
            Notes:
                - This method depends on the __init__ as the query engine along with the retriever, response_synthesizer
                objects is created there.
        """
        response = self.query_engine.query(str_or_query_bundle=doc_id)
        # self.observability.collect_save_traces()
        return response
