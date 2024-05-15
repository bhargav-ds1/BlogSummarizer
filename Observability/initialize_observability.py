import os.path
from typing import Optional
import llama_index.core
import phoenix as px
from llama_index.core import set_global_handler


class DefaultObservability:
    observ_provider = 'phoenix'
    observ_providers = ['deepeval', 'simple', 'phoenix']


# optimize class design
class InitializeObservability(DefaultObservability):
    """
        A simple bank account class that allows deposits, withdrawals, and balance checks.

        Attributes:
        - owner (str): Account owner's name.
        - balance (float): Account balance in dollars.

        Constructor Parameters:
        - owner (str): The name of the account owner.
        - balance (float, optional): Initial balance of the account. Default is 0.0.


        Examples:

        Notes:
        This class does not handle currency types or transaction histories, and should not be used for actual financial records.
        """

    def __init__(self, observ_provider: Optional[str] = 'phoenix') -> None:
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
            Initialize LLM observability with deepeval platform.

            Parameters:

            Returns:

            Examples:

            Notes:

        """
        from llama_index.callbacks.deepeval import deepeval_callback_handler
        from llama_index.core.callbacks import CallbackManager
        CallbackManager([deepeval_callback_handler()])
        # set_global_handler('deepeval')

    @staticmethod
    def initializeSimple() -> None:
        """
                    Initialize LLM observability with deepeval platform.

                    Parameters:

                    Returns:

                    Examples:

                    Notes:

                """
        set_global_handler('simple')

    @staticmethod
    def initializePhoenix() -> None:
        """
                    Initialize LLM observability with deepeval platform.

                    Parameters:

                    Returns:

                    Examples:

                    Notes:

                """
        px.launch_app()
        llama_index.core.set_global_handler("arize_phoenix")

    def collect_save_traces(self) -> None:
        """
                    Initialize LLM observability with deepeval platform.

                    Parameters:

                    Returns:

                    Examples:

                    Notes:

                """
        if self.observ_provider == 'phoenix':
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     'Tests/phoenix_span_dataset.csv')
            px.active_session().get_spans_dataframe().to_csv(file_path)
