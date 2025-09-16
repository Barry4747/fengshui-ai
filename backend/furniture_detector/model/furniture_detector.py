from ultralytics import YOLO
import cv2
import os

class FurnitureDetector:
    def __init__(self, model_path="runs/detect/train/weights/best.pt", device=0):
        """
        Inicjalizacja modelu YOLOv8.
        :param model_path: ścieżka do pliku .pt (np. best.pt albo yolov8s.pt)
        :param device: 0 = GPU, "cpu" = CPU
        """
        self.model = YOLO(model_path)
        self.device = device

    def predict_image(self, image_path, save_path=None, show=False):
        """
        Wykrywanie obiektów na pojedynczym obrazie.
        """
        results = self.model.predict(source=image_path, device=self.device, save=bool(save_path))
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            for i, result in enumerate(results):
                result.save(filename=os.path.join(save_path, f"result_{i}.jpg"))
        
        if show:
            for result in results:
                cv2.imshow("Prediction", result.plot())
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        return results

    def predict_folder(self, folder_path, save_path="outputs"):
        """
        Wykrywanie obiektów na wszystkich obrazach w folderze.
        """
        results = self.model.predict(source=folder_path, device=self.device, save=True, project=save_path)
        return results

    def predict_camera(self, camera_id=0):
        """
        Wykrywanie obiektów w czasie rzeczywistym z kamery.
        """
        cap = cv2.VideoCapture(camera_id)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, device=self.device)
            annotated_frame = results[0].plot()

            cv2.imshow("Furniture Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
