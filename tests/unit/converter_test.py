import pytest

from fantasy_maps import converter

FAKED_FILE = "gs://fake-bucket/fake-prefix/fake-file.jpg"

@pytest.fixture
def setup():
    faked_input = {
        "path": FAKED_FILE,
        "imageWidth": 2000,
        "imageHeight": 2000,
        "cellOffsetX": 0,
        "cellOffsetY": 0,
        "cellWidth": 100,
        "cellHeight": 100
    }

    faked_prediction =     {
        "instance": FAKED_FILE,
        "prediction": {
            "ids": [123, 123],
            "bboxes": [
                [0.1, 0.2, 0.1, 0.2],
                [0.3, 0.4, 0.3, 0.4]
            ],
            "confidences": [0.75, 0.4],
            "display_names": ["cell", "cell"]
        }
    }
    yield (faked_input, faked_prediction)

def test_convert_fantasy_map_to_bounding_boxes(setup):
    faked_input = setup[0]
    actual_bboxes, actual_path, actual_width, actual_height = converter.convert_fantasy_map_to_bounding_boxes(faked_input)

    assert len(actual_bboxes) == 324
    assert actual_path.find("fake-file.jpg") > -1
    assert actual_width == 2000
    assert actual_height == 2000

def test_convert_batch_predictions_to_training_data(setup):
    faked_prediction = setup[1]
    actual_training_data = converter.convert_batch_predictions_to_training_data(faked_prediction)

    assert actual_training_data["imageGcsUri"].find(FAKED_FILE) != -1