from .utils import save_depth_with_pil
from .model.registry import ModelManager

def get_depth_image(image_path: str = None, model_name: str ="midas"):
    if not image_path:
        pass
    model = ModelManager.get_model(model_name=model_name, model_category='depth')

    results = model.predict_image(image_path)

    if isinstance(results, dict):
        for fname, depth in results.items():
            print(fname, "-> depth map shape:", depth.shape)
    else:
        save_depth_with_pil(results)

    return results