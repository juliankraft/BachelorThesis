import json

from megadetector.detection.run_detector import load_detector
from megadetector.detection.run_detector_batch import process_images

from os import PathLike


class MegaDetectorRunner:
    """
    A class to run the MegaDetector model on images. Designed to be used on a set of image sequences,
    only loading the model once and running it on all sequences.

    Parameters
    ----------
    model_path : str | PathLike
        Path to the MegaDetector model file. Or a string representing the model version available online.
    confidence : float
        Confidence threshold for the model. Default is 0.25.
    """
    def __init__(
            self, 
            model_path: str | PathLike, 
            confidence: float = 0.25
            ):
        
        self.model = load_detector(str(model_path))
        self.confidence = confidence

    def run_on_images(
            self,
            images: list[PathLike],
            output_file_path: PathLike | None = None,
            ):

        results = process_images(
            im_files=images,
            detector=self.model,
            confidence_threshold=self.confidence,
            quiet=True
        )

        all_confidences = []

        if type (results) is not list:
            raise ValueError("The results should be a list of dictionaries.")
        
        for r in results:
            r["file"] = r["file"].name

            r["detections"] = [
                det for det in r.get("detections", [])
                if det["category"] == "1"
            ]
        
            all_confidences.extend(det["conf"] for det in r["detections"])

        all_confidences.sort(reverse=True)
        
        if output_file_path is not None:
            with open(output_file_path, "w") as f:
                json.dump(results, f, indent=2)

        return all_confidences      