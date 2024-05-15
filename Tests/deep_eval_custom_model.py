from typing import Coroutine, Any

from deepeval.models import DeepEvalBaseLLM
from llama_index.core.llms import LLM


class CustomEvaluationModel(DeepEvalBaseLLM):
    def __init__(
            self,
            model: LLM,
            *args,
            **kwargs,
    ) -> None:
        if isinstance(model, LLM):
            self.custom_model = model

        else:
            raise ValueError('Provide a valid LLM for evaluation.')

        super().__init__(self.custom_model.metadata.model_name, *args, **kwargs)

    def load_model(self, *args, **kwargs) -> LLM:
        """
                    Initialize LLM observability with deepeval platform.

                    Parameters:

                    Returns:

                    Examples:

                    Notes:

                """
        return self.custom_model

    def generate(self, prompt: str) -> Coroutine[Any, Any, str]:
        """
                    Initialize LLM observability with deepeval platform.

                    Parameters:

                    Returns:

                    Examples:

                    Notes:

                """
        return self.a_generate(prompt=prompt)

    async def a_generate(self, prompt: str) -> str:
        """
                    Initialize LLM observability with deepeval platform.

                    Parameters:

                    Returns:

                    Examples:

                    Notes:

                """
        res = self.custom_model.complete(prompt)
        return res.text

    def get_model_name(self, *args, **kwargs) -> str:
        """
                    Initialize LLM observability with deepeval platform.

                    Parameters:

                    Returns:

                    Examples:

                    Notes:

                """
        return self.custom_model.metadata.model_name
