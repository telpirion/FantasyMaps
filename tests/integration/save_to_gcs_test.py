import base64
import os
import pytest
from fantasy_maps import processed_grid_image
from google.cloud import aiplatform as aip

@pytest.fixture
def setup():
    filename = 'resources/gridded-ruined-keep.jpg'
    filepath = os.path.join(os.path.dirname(__file__), filename)
    project = os.environ['GCP_PROJECT']
    location = 'us-central1'
    endpoint_id = '835681613667893248'
    yield project, location, endpoint_id, filepath

def test_save_training_data_to_gcs_integration(setup):

    project, location, endpoint_id, file_path = setup

    with open(file_path, "rb") as f:
        file_content = f.read()

    encoded_content = base64.b64encode(file_content).decode("utf-8")
    aip.init(project=project, location=location)

    # Get the saved endpoint
    endpoint_name = f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
    endpoint = aip.Endpoint(endpoint_name)

    response = endpoint.predict(instances=[{"content": encoded_content, "mimeType": "image/jpeg"}],
                            parameters={
                                "confidence_threshold": 0.5,
                                "max_predictions": 5
                            })

    assert response is not None
    assert response.predictions is not None
    assert len(response.predictions) > 0

    processed_image = processed_grid_image.analyze_annotation_results(
        {"prediction": response.predictions[0]},
        local_file_uri=file_path
    )
    processed_image_str = str(processed_image)

    assert processed_image is not None
    assert processed_image_str.find('ruined') > -1

    processed_image.store_image_as_dataset_row('video-erschmid', 'fantasy-maps-tests')
