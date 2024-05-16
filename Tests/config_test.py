# Configuration file to perform tests

Config = {
    'eval_model_args': {'llm_provider': 'llama-index-togetherai',# llama-index-huggingface, llama-index-openai
                        'llm_model_name': 'meta-llama/Llama-3-70b-chat-hf',# meta-llama/Llama-2-7b-chat-hf, gpt-3.5-turbo
                        'llm_model_path': '',
                        'offload_dir': './offload_dir',
                        'cache_dir': '/Users/bhargavvankayalapati/Work/InHouseRAG/InHouseRAG/Models/meta-llama/Llama-2-7b-chat-hf',
                        'local_files_only': True, 'context_window': 4096,
                        'max_new_tokens': 512,
                        'generate_kwargs': {"temperature": 0.7, "top_k": 50, "top_p": 0.95,
                                            'do_sample': False},
                        'tokenizer_max_length': 4096,
                        'stopping_ids': (50278, 50279, 50277, 1, 0), },
}
