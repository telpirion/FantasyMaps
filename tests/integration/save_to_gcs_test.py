import base64
import os
import pytest
from datetime import datetime
from fantasy_maps import processed_grid_image
from google.cloud import aiplatform as aip


@pytest.fixture
def setup():
    filename_1 = "resources/gridded-ruined-keep.jpg"
    filepath_1 = os.path.join(os.path.dirname(__file__), filename_1)
    filename_2 = "resources/small-cemetary.jpg"
    filepath_2 = os.path.join(os.path.dirname(__file__), filename_2)

    project = os.environ["GCP_PROJECT"]
    location = "us-central1"
    endpoint_id = "835681613667893248"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    yield project, location, endpoint_id, filepath_1, filepath_2, timestamp


def test_save_training_data_to_gcs_integration(setup):

    project, location, endpoint_id, filepath_1, filepath_2, timestamp = setup

    aip.init(project=project, location=location)

    # Get the saved endpoint
    endpoint_name = f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
    endpoint = aip.Endpoint(endpoint_name)

    # Run prediction on the first image
    with open(filepath_1, "rb") as f:
        file_content = f.read()

    encoded_content = base64.b64encode(file_content).decode("utf-8")

    response = endpoint.predict(
        instances=[{"content": encoded_content, "mimeType": "image/jpeg"}],
        parameters={"confidence_threshold": 0.5, "max_predictions": 5},
    )

    assert response is not None
    assert response.predictions is not None
    assert len(response.predictions) > 0

    processed_image = processed_grid_image.analyze_annotation_results(
        {"prediction": response.predictions[0]}, local_file_uri=filepath_1
    )
    processed_image_str = str(processed_image)

    assert processed_image is not None
    assert processed_image_str.find("ruined") > -1

    processed_image.store_image_as_dataset_row(
        "video-erschmid", f"fantasy-maps-tests/integration-test-{timestamp}"
    )

    # TODO(telpirion): Put some asserts here?
    # Run prediction on the second image
    with open(filepath_2, "rb") as f:
        file_content = f.read()

    encoded_content_2 = base64.b64encode(file_content).decode("utf-8")

    response2 = endpoint.predict(
        instances=[{"content": encoded_content_2, "mimeType": "image/jpeg"}],
        parameters={"confidence_threshold": 0.5, "max_predictions": 5},
    )

    assert response2 is not None
    assert response2.predictions is not None
    assert len(response2.predictions) > 0

    processed_image2 = processed_grid_image.analyze_annotation_results(
        {"prediction": response2.predictions[0]}, local_file_uri=filepath_2
    )
    processed_image2_str = str(processed_image2)

    assert processed_image2 is not None
    assert processed_image2_str.find("cemetary") > -1

    processed_image2.store_image_as_dataset_row(
        "video-erschmid", f"fantasy-maps-tests/integration-test-{timestamp}"
    )
