from typing import Tuple

def process_predictions(predictions) -> Tuple:
  """Extracts IDs, confidences, display names, and bounding boxes from Vertex AI predictions.

  Args:
    predictions: a list of Vertex AI ImageObjectPredictionResult objects
  
  Returns:
    Tuple(bounding boxes, confidences, IDs, display names)
  """
  bboxes = None

  for num_prediction, prediction_ in enumerate(predictions):
    # ids, confidences, displayNames, bboxes
    print(f"Prediction number: {num_prediction + 1}\n")
    ids = prediction_["ids"]
    confidences = prediction_["confidences"]
    display_names = prediction_["displayNames"]
    bboxes = prediction_["bboxes"]

    for count, id in enumerate(ids):
      print(f"ID: {id}")
      print(f"Display name: {display_names[count]}")
      print(f"Confidence: {confidences[count]}")
      print(f"Bounding boxes: {bboxes[count]}\n\n")
  
  return (bboxes, confidences, ids, display_names)