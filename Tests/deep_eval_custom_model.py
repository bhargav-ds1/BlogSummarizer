from typing import Coroutine, Any

from deepeval.models import DeepEvalBaseLLM
from llama_index.core.llms import LLM


class CustomEvaluationModel(DeepEvalBaseLLM):
    """
        A custom model class for evaluating Large Language Models (LLMs) based on the DeepEval framework.
        This class allows integration of LLM implementation from Together-AI to conform to the LLM interface into the
        DeepEval evaluation suite.

        Attributes:
            custom_model (LLM): The large language model instance to be evaluated.
        Notes:
            Defining this model is required to use an LLM model other than OpenAI's models as evaluation LLMs for the
            deepeval framework.
    """

    def __init__(
            self,
            model: LLM,
            *args,
            **kwargs,
    ) -> None:
        """
            Initializes the CustomEvaluationModel with a given LLM.

                Parameters:
                    model (LLM): An instance of a model that implements the LLM interface.

                Raises:
                    ValueError: If the provided model instance does not implement the LLM interface.
                """
        if isinstance(model, LLM):
            self.custom_model = model

        else:
            raise ValueError('Provide a valid LLM for evaluation.')

        super().__init__(self.custom_model.metadata.model_name, *args, **kwargs)

    def load_model(self, *args, **kwargs) -> LLM:
        """
            Loads the custom LLM model. In this context, it simply returns the initialized model.

                Returns:
                    LLM: The large language model instance.
        """
        return self.custom_model

    def generate(self, prompt: str) -> Coroutine[Any, Any, str]:
        """
            Asynchronously generates a response for a given prompt using the custom LLM.

                Parameters:
                    prompt (str): The input text to generate a response for.

                Returns:
                    Coroutine[Any, Any, str]: A coroutine that when awaited returns the generated string.

                Notes:
                    Same implementation for both async and non-async generation is used.
        """
        return self.a_generate(prompt=prompt)

    async def a_generate(self, prompt: str) -> str:
        """
            Asynchronous helper function that actually performs the text generation.

                Parameters:
                    prompt (str): The input text to generate a response for.

                Returns:
                    str: The generated text as a string.
                Notes:
                    Same implementation for both async and non-async generation is used.
        """
        res = self.custom_model.complete(prompt)
        return res.text

    def get_model_name(self, *args, **kwargs) -> str:
        """
            Retrieves the name of the custom model.

                Returns:
                    str: The name of the model as stored in the model's metadata.
                """
        return self.custom_model.metadata.model_name
