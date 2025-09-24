from .depth.model.registry import ModelManager
from .depth.model.yolo import YOLOModel
from .depth.model.dpt_beit import MiDaSModel
from .dimensions.SAM import SAMSegmenter

# models to load
models_dict = {
    "YOLOModel": YOLOModel,
    "MiDaSModel": MiDaSModel,
    "SAMSegmenter": SAMSegmenter
}


ModelManager.load_config(class_map=models_dict)

