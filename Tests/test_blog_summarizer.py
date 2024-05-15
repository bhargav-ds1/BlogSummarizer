import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (AnswerRelevancyMetric, SummarizationMetric, FaithfulnessMetric, HallucinationMetric,
                              ToxicityMetric)
from .deep_eval_custom_model import CustomEvaluationModel
from SummaryGen.llm_model_provider import LLMProvider
from dotenv import load_dotenv
from .testConfig import Config
from .sample_test_case_generator import make_random_blog_eval_dataset
import pytest

root_dir = os.path.dirname(os.path.dirname(__file__))
load_dotenv(root_dir + '/.envfile')

custom_eval_llm_model = CustomEvaluationModel(model=LLMProvider(**Config['eval_model_args']).get_llm_model())
evaluation_dataset = make_random_blog_eval_dataset(num_queries=2)
print(len(evaluation_dataset.test_cases))

@pytest.mark.parametrize(
    "test_case",
    evaluation_dataset,
)
def test_answer_relevancy(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=custom_eval_llm_model)
    assert_test(test_case, [answer_relevancy_metric])


@pytest.mark.parametrize(
    "test_case",
    evaluation_dataset,
)
def test_summarization(test_case: LLMTestCase):
    summarization_metric = SummarizationMetric(threshold=0.5, model=custom_eval_llm_model)
    assert_test(test_case, metrics=[summarization_metric])


@pytest.mark.parametrize(
    "test_case",
    evaluation_dataset,
)
def test_faithfulness(test_case: LLMTestCase):
    faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=custom_eval_llm_model)
    assert_test(test_case, metrics=[faithfulness_metric])


@pytest.mark.parametrize(
    "test_case",
    evaluation_dataset,
)
def test_hallucination(test_case: LLMTestCase):
    hallucination_metric = HallucinationMetric(threshold=0.5, model=custom_eval_llm_model)
    assert_test(test_case, metrics=[hallucination_metric])


@pytest.mark.parametrize(
    "test_case",
    evaluation_dataset,
)
def test_toxicity(test_case: LLMTestCase):
    toxicity_metric = ToxicityMetric(threshold=0.5, model=custom_eval_llm_model)
    assert_test(test_case, metrics=[toxicity_metric])


if __name__ == '__main__':
    test_answer_relevancy()
