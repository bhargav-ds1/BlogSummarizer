import sys
import os

# Appending the parent directory to sys.path to enable imports from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (AnswerRelevancyMetric, SummarizationMetric, FaithfulnessMetric, HallucinationMetric,
                              ToxicityMetric)
from .deep_eval_custom_model import CustomEvaluationModel
from SummaryGen.llm_model_provider import LLMProvider
from dotenv import load_dotenv
from .config_test import Config
from .sample_test_case_generator import make_random_blog_eval_dataset
import pytest

# Load environment variables
root_dir = os.path.dirname(os.path.dirname(__file__))
load_dotenv(root_dir + '/.envfile')
# Initialize the custom evaluation model with configuration specified in the config_test file.
custom_eval_llm_model = CustomEvaluationModel(model=LLMProvider(**Config['eval_model_args']).get_llm_model())
evaluation_dataset = make_random_blog_eval_dataset(num_queries=4)
print(len(evaluation_dataset.test_cases))


@pytest.mark.parametrize(
    "test_case",
    evaluation_dataset,
)
def test_answer_relevancy(test_case: LLMTestCase):
    """
        Tests the answer relevancy of responses from a model against predefined test cases.
        Provides higher score for responses which are highly relevant to the provided query.

            Parameters:
                test_case (LLMTestCase): A test case object containing the input query, generated response and contexts.

        This function uses the AnswerRelevancyMetric to evaluate the response relevancy and asserts the test outcome.
    """
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=custom_eval_llm_model, async_mode=False)
    assert_test(test_case, [answer_relevancy_metric])


@pytest.mark.parametrize(
    "test_case",
    evaluation_dataset,
)
def test_summarization(test_case: LLMTestCase):
    """
        Tests the summarization quality of responses from a model against predefined test cases.
        Higher score if the provided summary response effectively summarizes the text provided as context.

            Parameters:
                test_case (LLMTestCase): A test case object containing the input query, generated response and contexts.
                As, summarization is the purpose of this project, the generated response is expected to be a summary.

        This function uses the SummarizationMetric to measure the accuracy of the summarization and asserts the test
        outcome.
    """
    summarization_metric = SummarizationMetric(threshold=0.5, model=custom_eval_llm_model)
    assert_test(test_case, metrics=[summarization_metric])


@pytest.mark.parametrize(
    "test_case",
    evaluation_dataset,
)
def test_faithfulness(test_case: LLMTestCase):
    """
        Tests the faithfulness of responses from a model against predefined test cases.
        Higher score if the actual summary output aligns with the contents of the retrieved context.

            Parameters:
                test_case (LLMTestCase): A test case object containing the input query, generated response and contexts.

        Notes:
        - This function uses the FaithfulnessMetric to assess whether the model's outputs are true to the original data
        and asserts the test outcome.
        - https://docs.confident-ai.com/docs/metrics-faithfulness

    """
    faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=custom_eval_llm_model)
    assert_test(test_case, metrics=[faithfulness_metric])


@pytest.mark.parametrize(
    "test_case",
    evaluation_dataset,
)
def test_hallucination(test_case: LLMTestCase):
    """
        Tests the hallucination rate of responses from a model against predefined test cases.
        Higher score, if actual summary output and the retrieved context are not comparable and new information is
        hallucinated and added by the LLM.
            Parameters:
                test_case (LLMTestCase): A test case object containing the input query, generated response and contexts.
        Notes:
            - This function uses the HallucinationMetric to evaluate the presence of hallucination in responses and asserts the test outcome.
            - https://docs.confident-ai.com/docs/metrics-hallucination
        """
    hallucination_metric = HallucinationMetric(threshold=0.5, model=custom_eval_llm_model)
    assert_test(test_case, metrics=[hallucination_metric])


@pytest.mark.parametrize(
    "test_case",
    evaluation_dataset,
)
def test_toxicity(test_case: LLMTestCase):
    """
        Tests the toxicity of responses from a model against predefined test cases.
        Higher score, if the response contains any toxic content.

            Parameters:
                test_case (LLMTestCase): A test case object containing the input query, generated response and contexts.

        Notes:
            - This function uses the ToxicityMetric to measure any toxicity in the responses and asserts the test outcome.
            - https://docs.confident-ai.com/docs/metrics-toxicity
        """
    toxicity_metric = ToxicityMetric(threshold=0.5, model=custom_eval_llm_model)
    assert_test(test_case, metrics=[toxicity_metric])
