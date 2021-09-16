import pytest

from fantasy_maps import converter

@pytest.fixture
def setup():
    faked_input = {
        "path": "gs://fake-bucket/fake-prefix/fake-file.jpg",
        "imageWidth": 2000,
        "imageHeight": 2000,
        "cellOffsetX": 0,
        "cellOffsetY": 0,
        "cellWidth": 100,
        "cellHeight": 100
    }
    yield faked_input

def test_convert_fantasy_map_to_bounding_boxes(setup):
    faked_input = setup
    actual_bboxes, actual_path, actual_width, actual_height = converter.convert_fantasy_map_to_bounding_boxes(faked_input)

    assert len(actual_bboxes) == 324
    assert actual_path.find("fake-file.jpg") > -1
    assert actual_width == 2000
    assert actual_height == 2000
    