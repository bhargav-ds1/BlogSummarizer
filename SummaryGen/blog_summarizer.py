from llama_index.core.query_engine.custom import STR_OR_RESPONSE_TYPE
from transformers import AutoModelForCausalLM
from llama_index.core import Settings, StorageContext
from llama_index.core import DocumentSummaryIndex, load_index_from_storage, VectorStoreIndex
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.response_synthesizers import BaseSynthesizer
from SummaryGen.fetch_blogs import FetchBlogs
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
import os

from Observability import InitializeObservability
from dotenv import load_dotenv
import os


# fetch blogs

#
class SummaryQueryEngine(CustomQueryEngine):
    docstore: SimpleDocumentStore
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str) -> STR_OR_RESPONSE_TYPE:
        document = self.docstore.get_document(doc_id=query_str)
        nodes = [NodeWithScore(node=node, score=1.0) for node in
                 SentenceSplitter().get_nodes_from_documents(documents=[document])]
        query = 'Summarize the information.'
        query_bundle = QueryBundle(query_str=query)
        response = self.response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
            streaming=True
        )
        return response


class DocumentSummaryGenerator:
    def __init__(self, llm_provider: str, llm_model_name: str, llm_model_path: str,
                 offload_dir: str = './offload_dir', cache_dir: str = None,
                 local_files_only: bool = False, context_window: int = 4096, max_new_tokens: int = 256,
                 generate_kwargs: dict = None, tokenizer_max_length: int = 4096,
                 stopping_ids: tuple[int] = (50278, 50279, 50277, 1, 0),
                 embedding_provider: str = 'llama-index-huggingface',
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L12-v2',
                 embedding_model_path: str = 'Models/sentence-transformers/all-MiniLM-L12-v2',
                 refetch_blogs: bool = False, output_dir: str = None):
        super().__init__()
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
        root_dir = os.path.dirname(__file__)
        InitializeObservability(observ_provider='phoenix')
        load_dotenv(root_dir + '/.envfile')

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

    def get_documents(self) -> DocumentSummaryIndex:
        llm = self.get_llm_model()
        embed_model = self.get_embed_model()
        Settings.llm = llm
        Settings.embed_model = embed_model
        if not os.path.exists(self.output_dir + '/docstore.json') or self.refetch_blogs:
            print('Fetching Blogs ...')
            blogs = self.blog_fetcher.fetch_blogs()
            # summary_generator = DocumentSummaryIndex.from_documents(documents=blogs[:2], show_progress=True,
            #                                                        embed_summaries=False,
            #                                                        )

            docstore = SimpleDocumentStore()
            docstore.add_documents(blogs)
            StorageContext.from_defaults(docstore=docstore).persist(self.output_dir)
            index = VectorStoreIndex.from_documents(documents=blogs,
                                                    show_progress=True)
            # index.storage_context.persist(persist_dir=self.output_dir)
            # summary_generator.storage_context.persist(persist_dir=self.output_dir)
        else:
            print('Using stored blogs content')
            docstore = SimpleDocumentStore().from_persist_dir(self.output_dir)

            # index = load_index_from_storage(
            #    storage_context=StorageContext.from_defaults(persist_dir=self.output_dir))
        '''query_template_str = ("The contents of the blog are provided as Context information below.\n"
                              "---------------------\n"
                              "{context_str}\n"
                              "---------------------\n"
                              "Given the information from multiple sources and not prior knowledge, "
                              "answer the query.\n"
                              "Query: {query_str}\n"
                              "Answer: ")
        query_template = SelectorPromptTemplate(
            default_template=PromptTemplate(
                query_template_str, prompt_type=PromptType.SUMMARY
            ),
        )
        '''
        response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.TREE_SUMMARIZE,
                                                        prompt_helper=PromptHelper.from_llm_metadata(llm.metadata,
                                                                                                     chunk_size_limit=llm.metadata.context_window - 1000),
                                                        verbose=True)
        # query_engine = RetrieverQueryEngine(retriever=index.as_retriever(), response_synthesizer=response_synthesizer)
        # query_engine.query('Generate a summary of the blog. Find below the contents of the blog ')
        return (docstore, response_synthesizer)

    def get_titles(self):
        docstore, response_synthesizer = self.get_documents()
        self.docstore = docstore
        self.response_synthesizer = response_synthesizer
        return list(docstore.docs.keys())

    def get_summary(self, doc_id):
        d = self.docstore.get_document(doc_id=doc_id)
        if 'summary' in d.metadata.keys():
            response = d.metadata['summary']
        else:
            query_engine = SummaryQueryEngine(docstore=self.docstore, response_synthesizer=self.response_synthesizer)
            response = query_engine.query(str_or_query_bundle=doc_id)
            d.metadata['summary'] = response
        return response
