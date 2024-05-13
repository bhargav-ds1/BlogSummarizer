from transformers import AutoModelForCausalLM
from llama_index.core import Settings, StorageContext
from llama_index.core import DocumentSummaryIndex
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.indices.prompt_helper import PromptHelper
from SummaryGen.fetch_blogs import FetchBlogs
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
import llama_index.core.query_engine as qe
from llama_index.core.base.response.schema import StreamingResponse
from Observability import InitializeObservability
from dotenv import load_dotenv
import os
from SummaryGen.blog_summary_query_engine import BlogSummaryQueryEngine, CustomRetriever


class DocumentSummaryGenerator:
    def __init__(self, llm_provider: str, llm_model_name: str, llm_model_path: str,
                 offload_dir: str = './offload_dir', cache_dir: str = None,
                 local_files_only: bool = False, context_window: int = 4096, max_new_tokens: int = 256,
                 generate_kwargs: dict = None, tokenizer_max_length: int = 4096,
                 stopping_ids: tuple[int] = (50278, 50279, 50277, 1, 0),
                 embedding_provider: str = 'llama-index-huggingface',
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L12-v2',
                 embedding_model_path: str = 'Models/sentence-transformers/all-MiniLM-L12-v2',
                 refetch_blogs: bool = False, output_dir: str = None, query_engine_type: qe.BaseQueryEngine = None,
                 query_engine_kwargs: dict = None, response_mode: str = 'tree_summarize',
                 chunk_size: int = 1024, chunk_overlap: int = 128,
                 streaming: bool = False, summary_template_str: str = None, use_async: bool = False):
        super().__init__()
        root_dir = os.path.dirname(os.path.dirname(__file__))
        InitializeObservability(observ_provider='phoenix')
        load_dotenv(root_dir + '/.envfile')
        self.llm_provider = llm_provider
        self.llm_model_name = llm_model_name
        self.llm_model_path = llm_model_path
        self.offload_dir = offload_dir
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens
        self.generate_kwargs = generate_kwargs
        self.tokenizer_max_length = tokenizer_max_length
        self.stopping_ids = stopping_ids
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name
        self.embedding_model_path = embedding_model_path
        self.blog_fetcher = FetchBlogs()
        self.refetch_blogs = refetch_blogs
        self.output_dir = output_dir
        self.summary_template_str = summary_template_str
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.streaming = streaming
        ##############################
        self.llm = self.get_llm_model()
        Settings.llm = self.llm
        ##############################
        try:
            self.response_mode = ResponseMode(response_mode)
        except Exception as e:
            print('Invalid Response mode:' +str(e))
        self.use_async = use_async
        self.response_synthesizer = self.get_response_synthesizer()
        ##############################
        self.docstore = self.get_documents()
        # self.query_engine = BlogSummaryQueryEngine(docstore=self.docstore,
        #                                           response_synthesizer=self.response_synthesizer,
        #                                           chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
        #                                           )
        self.retriever = CustomRetriever(docstore=self.docstore, chunk_size=self.chunk_size,
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


    def get_llm_model(self):
        # option to use llm from different sources, HuggingFace, Langchain, AWS, etc.
        if self.llm_provider == 'langchain-openai':
            pass
        elif self.llm_provider == 'llama-index-huggingface':
            from llama_index.llms.huggingface import HuggingFaceLLM
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.llm_model_path,
                device_map="cpu",  # or a cuda enabled device or mps
                offload_folder=self.offload_dir,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            llm = HuggingFaceLLM(
                context_window=self.context_window,
                max_new_tokens=self.max_new_tokens,
                generate_kwargs=self.generate_kwargs,
                # system_prompt=system_prompt,
                # query_wrapper_prompt=query_wrapper_prompt,
                tokenizer_outputs_to_remove=['</s>'],
                tokenizer_name=self.llm_model_name,
                model_name=self.llm_model_name,
                device_map="cpu",
                # stopping_ids=list(self.stopping_ids),
                tokenizer_kwargs={"max_length": self.tokenizer_max_length},
                model=model
                # uncomment this if using CUDA to reduce memory usage
                # model_kwargs={"torch_dtype": torch.float16}
            )
        elif self.llm_provider == 'langchain-aws-bedrock':
            pass
        elif self.llm_provider == 'llama-index-openai':
            from llama_index.llms.openai import OpenAI
            llm = OpenAI(self.llm_model_name)
        elif self.llm_provider == 'llama-index-togetherai':
            from llama_index.llms.together import TogetherLLM
            llm = TogetherLLM(model=self.llm_model_name)
        return llm

    def get_embed_model(self):
        if self.embedding_provider == 'langchain-openai':
            pass
        elif self.embedding_provider == 'langchain-huggingface':
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        elif self.embedding_provider == 'llama-index-huggingface':
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            return HuggingFaceEmbedding(model_name=self.embedding_model_name, cache_folder=self.embedding_model_path)
        elif self.embedding_provider == 'langchain-aws-bedrock':
            pass

    def get_response_synthesizer(self):
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
                                                        verbose=True, streaming=self.streaming, use_async=self.use_async)
        return response_synthesizer

    def get_documents(self) -> DocumentSummaryIndex:
        if not os.path.exists(self.output_dir + '/docstore.json') or self.refetch_blogs:
            print('Fetching Blogs ...')
            blogs = self.blog_fetcher.fetch_blogs()
            docstore = SimpleDocumentStore()
            docstore.add_documents(blogs)
            StorageContext.from_defaults(docstore=docstore).persist(self.output_dir)
        else:
            print('Using stored blogs content')
            docstore = SimpleDocumentStore().from_persist_dir(self.output_dir)

            # index = load_index_from_storage(
            #    storage_context=StorageContext.from_defaults(persist_dir=self.output_dir))
        '''
        '''

        # query_engine = RetrieverQueryEngine(retriever=index.as_retriever(), response_synthesizer=response_synthesizer)
        # query_engine.query('Generate a summary of the blog. Find below the contents of the blog ')
        return docstore

    def get_titles(self):
        return list(self.docstore.docs.keys())

    def get_summary_response(self, doc_id):
        response = self.query_engine.query(str_or_query_bundle=doc_id)
        return response

    def get_summary_txt(self):
        response = self.get_summary_response()
        streaming = isinstance(response, StreamingResponse)
        if streaming:
            response_txt = response.get_response()
        else:
            response_txt = str(response)
        return response_txt
