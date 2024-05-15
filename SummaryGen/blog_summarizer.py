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
    def __init__(self, llm_args: dict = None,
                 refetch_blogs: bool = False, output_dir: str = None, query_engine_type: qe.BaseQueryEngine = None,
                 query_engine_kwargs: dict = None, response_mode: str = 'tree_summarize',
                 chunk_size: int = 1024, chunk_overlap: int = 128,
                 streaming: bool = False, summary_template_str: str = None, use_async: bool = False,
                 observ_provider: str = 'phoenix'):
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
        return list(self.docstore.docs.keys())

    def get_summary_response(self, doc_id: str) -> Union[StreamingResponse, Response]:
        response = self.query_engine.query(str_or_query_bundle=doc_id)
        # self.observability.collect_save_traces()
        return response

    def get_summary_txt(self) -> str:
        response = self.get_summary_response()
        streaming = isinstance(response, StreamingResponse)
        if streaming:
            response_txt = response.get_response()
        else:
            response_txt = str(response)
        return response_txt
