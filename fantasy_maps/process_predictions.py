def process_predictions(predictions):

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
  
  return bboxes