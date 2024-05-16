from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
import phoenix as px
from llama_index.core.base.response.schema import StreamingResponse
import pandas as pd


def make_simple_eval_dataset() -> EvaluationDataset:
    """
        Generates a simple evaluation dataset containing a predefined LLM test case.

            Returns:
                EvaluationDataset: A dataset containing one test case/cases with predefined input and actual output.
            Notes:
                Requires filling in appropriate details to specify a valid LLMTestCase.
    """
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=[''],
        context=['']
    )
    return EvaluationDataset(test_cases=[test_case])


def make_random_blog_eval_dataset(num_queries: int = 4) -> EvaluationDataset:
    """
        Generates an evaluation dataset containing random blog summaries. Gets random blog titles from the list of
        titles and generates summaries for them. The data required for generating the LLMTestCases is acquired from the
        traces logged by the observability provider (Currently defaults to phoenix).

            Parameters:
                num_queries (int): The number of random queries to generate summaries for.

            Returns:
                EvaluationDataset: A dataset containing test cases generated from random blog summaries.
            Notes:
                Useful to generate EvaluationDataset specifically to test the blog summarization and to evaluate the
                summarization performance using relevant metrics.
    """
    from SummaryGen.blog_summarizer import DocumentSummaryGenerator
    from config import Config
    import random

    document_summarizer = DocumentSummaryGenerator(**Config['summarizer_args'], **Config['query_engine_args'])
    titles = document_summarizer.get_titles()
    blog_ids = random.sample(titles, num_queries)
    responses = []
    for blog_id in blog_ids:
        responses.append(document_summarizer.get_summary_response(doc_id=blog_id))
    a = [response.get_response() if isinstance(response, StreamingResponse) else response for response in
         responses]

    if document_summarizer.observability.observ_provider == 'phoenix':
        span_df = px.active_session().get_spans_dataframe()
        return make_eval_dataset_from_phoenix_df(span_df=span_df)
    else:
        test_cases = [LLMTestCase(input=i, actual_output=j) for i, j in zip(titles, a)]
        return EvaluationDataset(test_cases=test_cases)


def make_eval_dataset_from_phoenix_df(span_df: pd.DataFrame = None,
                                      remove_duplicates: bool = True) -> EvaluationDataset:
    """
        Generates an evaluation dataset from a Phoenix span data frame containing trace data.

            Parameters:
                span_df (pd.DataFrame): DataFrame containing span data from Phoenix traces. If None, it attempts to
                    fetch the trace data.
                remove_duplicates (bool): Flag to remove duplicate entries based on LLM input messages. This is
                    required to get the latest trace which contains full response. As, the LLM may receive multiple
                    calls for each summarization task.

            Returns:
                EvaluationDataset: A dataset of test cases constructed from trace data.

            Raises:
                Exception: If Phoenix client is not initialized or does not contain valid spans.
    """
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
    for trace_id in span_df['context.trace_id'].unique():
        df = span_df[span_df['context.trace_id'] == trace_id]
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
