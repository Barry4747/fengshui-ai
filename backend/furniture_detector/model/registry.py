from typing import Any, Dict, List
from dotenv import load_dotenv
from base_model_for_registry import BaseModel
from warnings import warn
import os
import yaml
import torch
import logging

load_dotenv()

# yaml class names mapped to actual model classes
CLASS_MAP = {}

logger = logging.getLogger(__name__)


class ModelManager:
    _instances: Dict[str, Any] = {}
    _models_map: Dict[str, Dict[str, Any]] = {} 
    _flat_models_map: Dict[str, Dict[str, Any]] = {}  
    _model_types: List[str] = []

    @classmethod
    def load_config(cls, config_path: str = None, class_map: Dict[str, Any] = None, default_vram=8):
        if not config_path:
            config_path = os.getenv('MODELS_YAML_PATH')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        models = config.get("models", {})

        for category, models_dict in models.items():
            cls._model_types.append(category)
            cls._models_map[category] = models_dict
            for model_name, data in models_dict.items():
                cls._flat_models_map[model_name] = data

        for name, klass in class_map.items():
            cls.register_class(name=name, klass=klass)

        cls.default_vram = default_vram

    @classmethod
    def list_models(cls, model_category: str = None):
        """
        Returns models per category if given,
        else return all models and the category they belong to
        """
        if not model_category:
            models: Dict[str, List[str]] = {}
            for category, v in cls._models_map.items():
                models[category] = list(v.keys())
            return models
        elif model_category not in cls._model_types:
            raise KeyError(f"There is no {model_category} category in config")

        return list(cls._models_map[model_category].keys())

    @staticmethod
    def _get_free_vram_gb() -> float:
        """ Returns free VRAM in GB."""
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info()
        return free / (1024**3) 

    @classmethod
    def get_model(cls, model_name: str, model_category: str = None):
        model_info = None
        if model_name not in cls._instances:
            if model_category:
                if model_category not in cls._model_types:
                    raise ValueError(f"Unknown category: {model_category}")
                elif model_name not in cls._models_map[model_category]:
                    raise ValueError(f"Model: {model_name} not found for category: {model_category}")
                else:
                    model_info = cls._models_map[model_category][model_name]
            else:
                model_info = cls._flat_models_map.get(model_name, None)
        
            if not model_info:
                raise ValueError(f"Couldn't read model info. Make sure name {model_name} and category {model_category} are correct")

            required_vram = model_info.get("required_vram", cls.default_vram)
            model_class_name = model_info["class"]

            free_gb = cls._get_free_vram_gb()
            if free_gb < float(required_vram):
                models = cls._find_models_to_unload(required_vram - free_gb)
                if models:
                    for m in models:
                        cls.unload_model(m)
                    free_gb = cls._get_free_vram_gb()

            if model_class_name not in CLASS_MAP:
                raise ValueError(f"Unknown class: {model_class_name}")
            model_class = CLASS_MAP[model_class_name]
            instance = model_class(**model_info['constructor_kwargs'])

            instance.load_model(**model_info['loader_kwargs'])

            cls._instances[model_name] = instance

        return cls._instances[model_name]
    

    @classmethod
    def unload_model(cls, model_name: str):
        if model_name in cls._instances:
            if hasattr(cls._instances[model_name], "unload_model"):
                cls._instances[model_name].unload_model()
            del cls._instances[model_name]
            logger.info(f"Unloaded model {model_name}")

    @classmethod
    def switch_model(cls, old_model: str, new_model: str):
        if old_model == new_model:
            return cls.get_model(old_model)
        cls.unload_model(old_model)
        return cls.get_model(new_model)
    
    @classmethod
    def _find_models_to_unload(cls, required_vram: float) -> list:
        """Finds models to unload to free up the required VRAM."""

        sorted_models = sorted(
            cls._instances.items(),
            key=lambda item: cls._flat_models_map.get(item[0], {}).get("required_vram", cls.default_vram),
            reverse=True
        )

        to_unload = []
        freed_vram = 0.0

        for model_name, instance in sorted_models:
            model_info = cls._flat_models_map.get(model_name, {})
            model_vram = model_info.get("required_vram", 10)
            to_unload.append(model_name)
            freed_vram += model_vram
            if freed_vram >= required_vram:
                break

        return to_unload

    @classmethod
    def register_class(cls, name: str, klass: type):
        """
        Register a model class in CLASS_MAP.
        Warns the user if klass does not inherit from BaseModel.
        """  

        if not issubclass(klass, BaseModel):
            warn(f"Class '{klass.__name__}' does not inherit from BaseModel. "
                 "This may cause errors when calling load_model or unload_model!", UserWarning)
        CLASS_MAP[name] = klass
