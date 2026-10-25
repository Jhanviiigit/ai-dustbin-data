"""The student's interface for the detector.

    classifier = WasteClassifier(model_path="student/runs/gray/weights.pt")
    result = classifier.classify_image(crop_bgr)    # numpy BGR image of ONE item
    # {"label": "dry waste" | "wet waste", "confidence": 0.0-1.0,
    #  "dry_prob": ..., "wet_prob": ..., "inference_ms": ...}

confidence is the calibrated probability of the predicted label (temperature scaling).
The temperature is stored inside the weights file, so one file is all that is needed.
"""
import time

import numpy as np
import torch

from common import CLASSES, EVAL_TF, bgr_to_pil, load_checkpoint


class WasteClassifier:
    def __init__(self, model_path):
        self.model, ckpt = load_checkpoint(model_path)
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.variant = ckpt.get("variant")
        torch.set_grad_enabled(False)

    def classify_image(self, image):
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a HxWx3 BGR numpy array")
        start = time.perf_counter()
        x = EVAL_TF(bgr_to_pil(image)).unsqueeze(0)
        probs = torch.softmax(self.model(x)[0] / self.temperature, dim=0)
        idx = int(probs.argmax())
        return {
            "label": CLASSES[idx],
            "confidence": float(probs[idx]),
            "dry_prob": float(probs[0]),
            "wet_prob": float(probs[1]),
            "inference_ms": (time.perf_counter() - start) * 1000,
        }
