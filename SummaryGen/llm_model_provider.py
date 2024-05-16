from llama_index.core.llms import LLM
from llama_index.core.llms.mock import MockLLM


class LLMProvider:
    """
        A class to provide different implementations of large language models (LLMs) based on the specified provider.

        This class allows the selection and configuration of various LLMs such as those from HuggingFace, OpenAI, or custom
        implementations, with support for specific parameters like context window size, cache management, and token limits.

        Attributes:
            llm_provider (str): The name of the provider from which to source the LLM.
            llm_model_name (str): The name of the model to be used.
            llm_model_path (str): The filesystem path to the model, if using locally stored LLM and not loading from a
                                    standard source.
            offload_dir (str): Directory to offload parts of the model to manage memory usage.
            cache_dir (str): Directory to cache parts of the model to avoid re-downloading.
            local_files_only (bool): Flag to restrict loading models to local files only.
            context_window (int): The maximum number of tokens the model can consider in a single request.
            max_new_tokens (int): The maximum number of new tokens to be generated in the model's responses.
            generate_kwargs (dict): Additional keyword arguments which helps control LLM response generation.
            tokenizer_max_length (int): Maximum length of tokens for the tokenizer.
            stopping_ids (tuple[int]): Tuple of token IDs used to indicate the end of generation.
    """

    def __init__(self, llm_provider: str, llm_model_name: str, llm_model_path: str = None,
                 offload_dir: str = './offload_dir', cache_dir: str = None,
                 local_files_only: bool = False, context_window: int = 4096, max_new_tokens: int = 256,
                 generate_kwargs: dict = None, tokenizer_max_length: int = 4096,
                 stopping_ids: tuple[int] = (50278, 50279, 50277, 1, 0)) -> None:
        """
            Initializes the LLMProvider class with provided arguments and provides default values which are tested with
             a local Llama2 model downloaded from huggingface .
        """
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

    def get_llm_model(self) -> LLM:
        """
            Retrieves an LLM based on the specified provider and configuration.

                Returns:
                    LLM: An instance of an LLM class tailored to the specified configurations and provider.
        """
        # option to use llm from different sources, HuggingFace, Langchain, AWS, etc.
        # API provided by Together-AI is used to build and test this project
        llm = MockLLM()
        if self.llm_provider == 'langchain-openai':
            pass
        elif self.llm_provider == 'llama-index-huggingface':
            from llama_index.llms.huggingface import HuggingFaceLLM
            from transformers import AutoModelForCausalLM
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
        else:
            print('Please provide a valid LLM provider. Using mock LLM, this might result in unexpected results.')
        return llm
