from deepeval.models import DeepEvalBaseLLM
from llama_index.core.llms import LLM
from typing import Tuple
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter
class CustomEvaluationModel(DeepEvalBaseLLM):
    def __init__(
            self,
            model: LLM,
            *args,
            **kwargs,
    ):
        if isinstance(model, LLM):
            self.custom_model = model

        else:
            raise ValueError('Provide a valid LLM for evaluation.')

        super().__init__(self.custom_model.metadata.model_name, *args, **kwargs)

    def load_model(self, *args, **kwargs):
        return self.custom_model

    def generate(self, prompt: str):
        res = self.custom_model.complete(prompt)
        return res.text

    async def a_generate(self, prompt: str):
        res = self.custom_model.complete(prompt)
        return res.text

    def get_model_name(self, *args, **kwargs) -> str:
        return self.custom_model.metadata.model_name
