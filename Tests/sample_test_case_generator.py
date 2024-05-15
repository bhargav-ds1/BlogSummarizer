from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
import phoenix as px
from llama_index.core.base.response.schema import StreamingResponse
import pandas as pd


def make_simple_eval_dataset() -> EvaluationDataset:
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=[''],
        context=['']
    )
    return EvaluationDataset(test_cases=[test_case])


def make_random_blog_eval_dataset(num_queries: int = 4) -> EvaluationDataset:
    from SummaryGen.blog_summarizer import DocumentSummaryGenerator
    from config import Config
    import random

    document_summarizer = DocumentSummaryGenerator(**Config['summarizer_args'], **Config['query_engine_args'])
    titles = document_summarizer.get_titles()
    blog_ids = random.sample(titles, num_queries)
    responses = []
    for id in blog_ids:
        responses.append(document_summarizer.get_summary_response(doc_id=id))
    a = [response.get_response() if isinstance(response, StreamingResponse) else response for response in
         responses]

    if document_summarizer.observability.observ_provider == 'phoenix':
        import phoenix as px
        span_df = px.active_session().get_spans_dataframe()
        return make_eval_dataset_from_phoenix_df(span_df=span_df)
    else:
        test_cases = [LLMTestCase(input=i, actual_output=j) for i, j in zip(titles, a)]
        return EvaluationDataset(test_cases=test_cases)


def make_eval_dataset_from_phoenix_df(span_df: pd.DataFrame = None,
                                      remove_duplicates: bool = True) -> EvaluationDataset:
    test_cases = []
    if span_df is None:
        try:
            span_df = px.active_session().get_trace_dataset().dataframe
        except Exception as e:
            print(
                'The phoenix client is not initialized. Provide a span_df or Call this function only when a phoenix client is initialized and contains valid spans')
            raise e
    span_df = span_df[span_df['name'] == 'llm']
    if remove_duplicates:
        span_df = span_df.sort_values('start_time', ascending=False).drop_duplicates('attributes.llm.input_messages')
    for id in span_df['context.trace_id'].unique():
        df = span_df[span_df['context.trace_id'] == id]
        llm_span = df[df['name'] == 'llm']
        test_cases.append(
            LLMTestCase(  # input="Given the information and not prior knowledge, summarize the blog.\n"
                #     "Summary: ",
                input=llm_span['attributes.llm.prompt_template.template'].iloc[0],
                actual_output=llm_span['attributes.output.value'].iloc[0],
                context=[eval(str(llm_span['attributes.llm.prompt_template.variables'].iloc[0]))['context_str']],
                retrieval_context=[
                    eval(str(llm_span['attributes.llm.prompt_template.variables'].iloc[0]))['context_str']]
            )
        )
    return EvaluationDataset(test_cases=test_cases)
