import torch
from ultralytics import YOLO
from .base_model_for_registry import BaseModel


class YOLOModel(BaseModel):
    """
    YOLOv8 model wrapper compatible with ModelManager.
    """

    def __init__(self, model_path: str = "runs/detect/train/weights/best.pt", device: int = 0):
        self.model_path = model_path
        self.device = device
        self.model = None

    def load_model(self, **kwargs):
        """
        Load YOLOv8 model.
        """
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        print(f"[YOLOModel] Loaded model from {self.model_path} on device {self.device}")

    def unload_model(self):
        """
        Free GPU memory used by YOLO.
        """
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None
            print(f"[YOLOModel] Unloaded model from {self.model_path}")

    def predict_image(self, image_path: str, save_path: str = None, show: bool = False):
        """
        Run inference on a single image.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        results = self.model.predict(source=image_path, device=self.device, save=bool(save_path))

        if save_path:
            for result in results:
                result.save(filename=save_path)

        if show:
            for result in results:
                result.show()

        return results

    def predict_folder(self, folder_path: str, save_path: str = "outputs"):
        """
        Run inference on all images in a folder.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        results = self.model.predict(source=folder_path, device=self.device, save=True, project=save_path)
        return results

    def predict_camera(self, camera_id: int = 0):
        """
        Run real-time inference on webcam.
        Press 'q' to quit.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        import cv2

        cap = cv2.VideoCapture(camera_id)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, device=self.device)
            annotated_frame = results[0].plot()

            cv2.imshow("YOLO Camera", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
