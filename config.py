# Configuration file to control the flow

Config = {
    'summarizer_args': {'llm_provider': 'llama-index-huggingface',
                        'llm_model_name': 'meta-llama/Llama-2-7b-chat-hf',
                        'llm_model_path': 'Models/meta-llama/Llama-2-7b-chat-hf',
                        'offload_dir': './offload_dir',
                        'cache_dir': '/Users/bhargavvankayalapati/Work/InHouseRAG/InHouseRAG/Models/meta-llama/Llama-2-7b-chat-hf',
                        'local_files_only': True, 'context_window': 4096,
                        'max_new_tokens': 512,
                        'generate_kwargs': {"temperature": 0.7, "top_k": 50, "top_p": 0.95,
                                            'do_sample': False},
                        'tokenizer_max_length': 4096,
                        'stopping_ids': (50278, 50279, 50277, 1, 0),
                        'embedding_provider': 'llama-index-huggingface',
                        'embedding_model_name': 'sentence-transformers/all-MiniLM-L12-v2',
                        'refetch_blogs': False,
                        'output_dir': 'Data/Blogs_content'
                        },
    'query_engine_args': {'query_engine_type': 'RetrieverQueryEngine',
                          'query_engine_kwargs': None,
                          'response_mode': 'compact', 'streaming': True,
                          'text_qa_template_str': None,
                          'refine_template_str': None, 'summary_template_str': None,
                          'simple_template_str': None,
                          'node_postprocessors': None,
                          'callback_manager': None,
                          'use_async': False},
}
