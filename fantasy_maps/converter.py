import math
from typing import Tuple, Union

def convert_fantasy_map_to_bounding_boxes(map_dict : dict) -> Tuple:
    """Translates fantasy map values from dictionary into bounding boxes.

    After conversion, the bounding boxes can then be used as input for Vertex AI
    training.

    The input dictionary must include the following keys:
    + imageHeight
    + imageWidth
    + cellWidth
    + cellheight
    + cellOffsetX
    + cellOffsetY
    + path

    Args:
        map_dict: the dictionary with the fantasy map values

    Returns:
        Tuple of bboxes, file_name, width, and height
    """
    height = map_dict["imageHeight"]
    width = map_dict["imageWidth"]
    cell_width = map_dict["cellWidth"]
    cell_height = map_dict["cellHeight"]

    num_columns = math.floor(width / cell_width) - 1
    num_rows = math.floor(height / cell_height) - 1

    bboxes = []

    for x in range(1, num_columns):
        for y in range(1, num_rows):
            x_min_tmp = map_dict["cellOffsetX"] + (cell_width * x) - 2
            x_max_tmp = x_min_tmp + cell_width + 4
            y_min_tmp = map_dict["cellOffsetY"] + (cell_height * y) - 2
            y_max_tmp = y_min_tmp + cell_height + 4

            x_min_train = x_min_tmp / width
            x_max_train = x_max_tmp / width
            y_min_train = y_min_tmp / height
            y_max_train = y_max_tmp / height

            bboxes.append(
                {
                    "xMin": x_min_train,
                    "yMin": y_min_train,
                    "yMax": y_max_train,
                    "xMax": x_max_train,
                }
            )

    file_name = map_dict["path"].split("/")[-1]

    return (bboxes, file_name, width, height)

def convert_batch_predictions_to_training_data(
    json_data: dict,
    minimum_confidence_value : float = 0.5
) -> Union[dict, None]:
    """Transforms IOD batch prediction results JSON into training data.

    Assumes input in the following format:
    ```
    {
        "instance": {
            "content": "[GCS_URI]",
            "mimeType": "[MIME_TYPE]"
        },
        "prediction": {
            "ids": [],
            "bboxes": [
                [
                    [X_MIN], [X_MAX], [Y_MIN], [Y_MAX]
                ]
            ],
            "confidences": [],
            "display_names": []
        }
    }
    ```

    The output needs to correspond to Vertex IOD training format:

    ```
    {
        "imageGcsUri": "[GCS_URI]",
        "boundingBoxAnnotations": [
            {
                "displayName": "cell",
                "xMin": [X_MIN],
                "yMin": [Y_MIN],
                "xMax": [X_MAX],
                "yMax": [Y_MAX]

            },
            ...
        ]
    }
    ```
    
    Args:
        json_data: the (batch) prediction data to transform
        minimum_confidence_value: the lowest allowable confidence value to
            allow in the resulting output. This acts as a filter for the 
            bounding boxes that are transformed into training data. Default
            value is 0.5.
    
    Returns:
        Vertex AI image object detection training data JSON (see remarks) or
        None if all predictions are below the minimum confidence value
    """
    try:
        image_gcs_uri = json_data["instance"]["content"]
        prediction = json_data["prediction"]
        bboxes = []

        for num, value in enumerate(prediction["confidences"]):
            if value > minimum_confidence_value:
                bbox = prediction["bboxes"][num]
                bboxes.append({
                        "displayName": "cell",  # "cell" is a constant
                        "xMin": bbox[0],
                        "yMin": bbox[2],
                        "xMax": bbox[1],
                        "yMax": bbox[3]

                    })

        if len(bboxes) == 0:
            return None
        else:
            return {
                "imageGcsUri": image_gcs_uri,
                "boundingBoxAnnotations": bboxes
            }

    except KeyError as key_error:
        print(f"Input has incorrect or missing key.\nFull error:\n{key_error}")