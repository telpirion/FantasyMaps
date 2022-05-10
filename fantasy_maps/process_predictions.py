from typing import Union, Tuple


def process_predictions(predictions, is_printed_to_out=False) -> Union[Tuple, None]:
    """Extracts IDs, confidences, display names, bounding boxes from first VertexAI prediction

    Args:
    predictions: a list of Vertex AI ImageObjectPredictionResult objects

    Returns:
    Tuple(bounding boxes, confidences, IDs, display names)
    """
    if len(predictions) == 0:
        return None

    prediction_ = predictions[0]

    # ids, confidences, displayNames, bboxes
    ids = prediction_["ids"]
    confidences = prediction_["confidences"]
    display_names = prediction_["displayNames"]
    bboxes = prediction_["bboxes"]

    if is_printed_to_out:
        for count, id in enumerate(ids):
            print(f"ID: {id}")
            print(f"Display name: {display_names[count]}")
            print(f"Confidence: {confidences[count]}")
            print(f"Bounding boxes: {bboxes[count]}\n\n")

    return (bboxes, confidences, ids, display_names)
