import cv2
import numpy as np
import onnxruntime as ort


class VisionSystem:
    def __init__(self, model_path="best.onnx"):
        # Configuration parameters
        self.img_size = 640
        self.conf_thresh = 0.70
        self.nms_thresh = 0.45
        self.classes = ["damaged_apple", "ripe", "unripe"]
        self.colors = {0: (255, 255, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
        self.target_class_id = 1  # 'ripe'

        # Load ONNX Model
        try:
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            raise Exception(f"Failed to load ONNX model: {e}")

    def preprocess(self, image):
        img = cv2.resize(image, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, outputs, orig_h, orig_w):
        predictions = outputs[0][0]
        objectness = predictions[:, 4]

        mask = objectness > self.conf_thresh
        predictions = predictions[mask]
        objectness = objectness[mask]

        if len(predictions) == 0:
            return []

        class_probs = predictions[:, 5:]
        class_ids = np.argmax(class_probs, axis=1)
        final_scores = objectness * np.max(class_probs, axis=1)

        score_mask = final_scores > self.conf_thresh
        predictions = predictions[score_mask]
        class_ids = class_ids[score_mask]
        final_scores = final_scores[score_mask]

        if len(predictions) == 0:
            return []

        scale_x, scale_y = orig_w / self.img_size, orig_h / self.img_size
        boxes = predictions[:, :4]
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y
        w, h = boxes[:, 2] * scale_x, boxes[:, 3] * scale_y
        bboxes = np.column_stack([x1, y1, w, h]).astype(int)

        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(), final_scores.tolist(), self.conf_thresh, self.nms_thresh)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append((bboxes[i], final_scores[i], class_ids[i]))
        return results

    def detect(self, frame):
        """
        Processes an external frame and returns the frame with drawings,
        plus the target info if found.
        """
        h, w, _ = frame.shape
        target_info = None  # Will store (centroid_x, centroid_y, bbox)

        # Inference
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections = self.postprocess(outputs, h, w)

        # Draw and Extract Target
        for box, score, class_id in detections:
            x, y, bw, bh = box
            color = self.colors.get(class_id, (0, 255, 0))
            label = f"{self.classes[class_id]} {score:.2f}"

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Prioritize the "Ripe" class
            if class_id == self.target_class_id:
                cx, cy = int(x + (bw / 2)), int(y + (bh / 2))
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                # Save target info: (Center X, Center Y, BoundingBoxTuple)
                target_info = (cx, cy, (x, y, bw, bh))

        return frame, target_info
