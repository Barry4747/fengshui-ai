from .model.registry import ModelManager
from .model.yolo import YOLOModel


# models to load
models_dict = {
    "YOLOModel": YOLOModel
}


ModelManager.load_config(class_map=models_dict)

