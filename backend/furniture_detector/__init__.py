from .model.registry import ModelManager
from .model.yolo import YOLOModel
from .model.dpt_beit import MiDaSModel

# models to load
models_dict = {
    "YOLOModel": YOLOModel,
    "MiDaSModel": MiDaSModel
}


ModelManager.load_config(class_map=models_dict)

