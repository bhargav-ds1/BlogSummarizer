from dotenv import load_dotenv
import os
from Observability import InitializeObservability
from config import Config
from SummaryGen.fetch_blogs import FetchBlogs
import pandas as pd
from SummaryGen.blog_summarizer import DocumentSummaryGenerator


class JobLeadsBlogSummary:
    def __init__(self):
        self.root_dir = os.path.dirname(__file__)
        InitializeObservability(observ_provider='phoenix')
        load_dotenv(self.root_dir + '/.envfile')

    def make_summary_generator(self, llm_args: dict = None, data_args: dict = None,
                               ):
        summary_generator = DocumentSummaryGenerator(**llm_args, data_args=data_args).get_summary_generator()
        return summary_generator

    def makeEvaluationDataset(self, query_engine, rag_dataset_generation_args):
        pass

    def evaluate(self, query_engine, evaluation_args):
        pass


def get_summary_generator(arguments_config: dict = None):
    root_dir = os.path.dirname(__file__)
    InitializeObservability(observ_provider='phoenix')
    load_dotenv(root_dir + '/.envfile')
    summary_generator = DocumentSummaryGenerator(**arguments_config['llm_args'],
                                                 data_args=arguments_config['data_args']).get_summary_generator()
    return summary_generator


if __name__ == '__main__':
    get_summary_generator(arguments_config=Config)
    # notion_model.make_streamlit_app(query_engine=query_engine)

    # notion_model.cleanData(output_dir='./DataHouse/Data/notion-ingest-output',items_to_remove=["UncategorizedText"])
    # notion_model.storeEmbeddings(input_dir='./DataHouse/Data/notion-ingest-output',
    #                             store='chroma')
