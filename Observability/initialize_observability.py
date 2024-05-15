import os.path
from typing import Optional
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import set_global_handler
from llama_index.core import Settings
import llama_index.core
import phoenix as px


class DefaultObservability:
    observ_provider = 'phoenix'
    observ_providers = ['deepeval', 'simple', 'phoenix']


# optimize class design
class InitializeObservability(DefaultObservability):
    def __init__(self, observ_provider: Optional[str] = 'phoenix'):
        self.observ_provider = observ_provider
        if self.observ_provider not in self.observ_providers:
            raise ValueError('Observability provider should be one of ' + ','.join(self.observ_providers))
        if self.observ_provider == 'deepeval':
            self.initializeDeepEval()
        if self.observ_provider == 'simple':
            self.initializeSimple()
        if self.observ_provider == 'phoenix':
            self.initializePhoenix()

    def initializeDeepEval(self) -> None:
        from llama_index.callbacks.deepeval import deepeval_callback_handler
        from llama_index.core.callbacks import CallbackManager
        CallbackManager([deepeval_callback_handler()])
        from llama_index.callbacks.arize_phoenix import arize_phoenix_callback_handler
        # set_global_handler('deepeval')

    def initializeSimple(self) -> None:
        set_global_handler('simple')

    def initializePhoenix(self) -> None:
        px.launch_app()
        llama_index.core.set_global_handler("arize_phoenix")

    def collect_save_traces(self) -> None:
        if self.observ_provider == 'phoenix':
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'Tests/phoenix_span_dataset.csv')
            px.active_session().get_spans_dataframe().to_csv(file_path)
