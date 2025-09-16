from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract Base Class for all models.
    Ensures every model implements `load_model` and `unload_model`.
    """

    @abstractmethod
    def load_model(self, **kwargs):
        """
        Load model weights / initialize resources.
        """
        pass

    @abstractmethod
    def unload_model(self):
        """
        Free VRAM / GPU memory / resources used by the model.
        """
        pass
