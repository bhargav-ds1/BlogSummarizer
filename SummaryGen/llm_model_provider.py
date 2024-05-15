class LLMProvider:
    def __init__(self,llm_provider: str, llm_model_name: str, llm_model_path: str= None,
                 offload_dir: str = './offload_dir', cache_dir: str = None,
                 local_files_only: bool = False, context_window: int = 4096, max_new_tokens: int = 256,
                 generate_kwargs: dict = None, tokenizer_max_length: int = 4096,
                 stopping_ids: tuple[int] = (50278, 50279, 50277, 1, 0)):
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

    def get_llm_model(self):
        # option to use llm from different sources, HuggingFace, Langchain, AWS, etc.
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
            llm = TogetherLLM(model=self.llm_model_name,)
        return llm



