import math


def convert_fantasy_map_to_bounding_boxes(map_dict):
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
