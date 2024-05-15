import os.path
from typing import Optional
import llama_index.core
import phoenix as px
from llama_index.core import set_global_handler


class DefaultObservability:
    """
            A Class which acts a base class to provide the default and available observability providers.
    """
    observ_provider = 'phoenix'
    observ_providers = ['deepeval', 'simple', 'phoenix']


# optimize class design
class InitializeObservability(DefaultObservability):
    """
        A Class to encapsulate different observability initializations based on provider.

        Constructor Parameters:
        - observ_provider (Optional[str]): The name of the observability provider. Default is 'phoenix'


        Examples:
            - (call) InitializeObservability('phoenix')
        Notes:
    """

    def __init__(self, observ_provider: Optional[str] = 'phoenix') -> None:
        """
                    Initializes the class with the observability provider name and calls the respective method.
        """
        self.observ_provider = observ_provider
        if self.observ_provider not in self.observ_providers:
            raise ValueError('Observability provider should be one of ' + ','.join(self.observ_providers))
        if self.observ_provider == 'deepeval':
            self.initializeDeepEval()
        if self.observ_provider == 'simple':
            self.initializeSimple()
        if self.observ_provider == 'phoenix':
            self.initializePhoenix()

    @staticmethod
    def initializeDeepEval() -> None:
        """
            Initialize LLM observability with deepeval platform as observability provider.
        """
        from llama_index.callbacks.deepeval import deepeval_callback_handler
        from llama_index.core.callbacks import CallbackManager
        CallbackManager([deepeval_callback_handler()])
        # set_global_handler('deepeval')

    @staticmethod
    def initializeSimple() -> None:
        """
                    Initialize the default LLM observability provided by llama-index.
        """
        set_global_handler('simple')

    @staticmethod
    def initializePhoenix() -> None:
        """
                   Initialize LLM observability with phoenix platform as observability provider.

        """
        px.launch_app()
        llama_index.core.set_global_handler("arize_phoenix")

    def collect_save_traces(self) -> None:
        """
            Saves the traces captured by the observability provider. Currently works for phoenix.
        """
        if self.observ_provider == 'phoenix':
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'Tests/phoenix_span_dataset.csv')
            px.active_session().get_spans_dataframe().to_csv(file_path)
