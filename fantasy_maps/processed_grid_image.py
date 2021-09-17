import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import json
import math
import os
import PIL

from google.cloud import storage


class ProcessedGridImage:
    """A wrapper that combines image plotting and bounding boxes.

    This class combines image plotting features from imageio, imguag, and pillow
    with extraction techniques for Vertex AI prediction outputs.
    """

    CONFIDENCE_THRESHOLD = 0.7
    CELL_LABEL = "cell"

    def __init__(
        self,
        width,
        height,
        bboxes,
        confidences,
        *,
        drift_amount=5,
        local_file_uri="",
        gcs_file_uri="",
    ):
        """Instantiates the ProcessedGridImage class

        Args:
          width: the width of the image in pixels
          height: the height of the image in pixels
          bboxes: the bounding boxes of the objects detected on the image
          confidences: the confidence scores of the objects detected on the image
          drift_amount: a fudge factor for normalizing bounding boxes outside of
            the bounds of the actual gridlines
          local_file_uri: the filepath to the image file, local file
          gcs_file_uri: the Cloud Storage URI (gs://) to the image file

        """

        self.width = width
        self.height = height
        self.bboxes = bboxes
        self.confidences = confidences
        self.drift_amount = drift_amount
        self.local_file_uri = local_file_uri
        self.gcs_file_uri = gcs_file_uri
        self.bboxes_on_image = []
        self.normalized_bboxes = []
        self.cell_width = 0
        self.cell_height = 0
        self.cell_width_percent = 0
        self.cell_height_percent = 0
        self.cell_offset_x = 0  # Assuming a 0 offset map
        self.cell_offset_y = 0  # Assuming a 0 offset map

        self._compute_actual_bboxes()
        self._compute_normalized_bboxes()

        self.grid_cells_width = math.ceil(self.width / self.cell_width)
        self.grid_cells_height = math.ceil(self.height / self.cell_height)

    def __str__(self):
        return str(self.get_normalized_json_dnd())

    def show(self):
        """Shows the predicted bounding boxes overlaid on the original image."""
        if len(self.bboxes_on_image) == 0:
            self._compute_actual_bboxes()

        self._show(self.bboxes_on_image)

    def show_normalized_results(self, *, is_grid_based=False):
        """Shows normalized values for bounding boxes, based upon predictions.

        Args:
          is_grid_based: determines whether to show cells based upon inferred grid
            width and height
        """
        if not is_grid_based:
            if len(self.normalized_bboxes) == 0:
                self._compute_normalized_bboxes()

            self._show(
                self.normalized_bboxes,
            )

        else:
            grid_based_boxes = self._compute_grid_based_boxes()
            self._show(grid_based_boxes)

    def get_normalized_json_dnd(self):
        """Generates VTT-friendly version of this processed image as JSON.

        Returns:
          JSON representation of processed image
        """

        if len(self.normalized_bboxes) == 0:
            self._compute_normalized_bboxes()

        dnd_json_as_dict = {
            "imageWidth": self.width,
            "imageHeight": self.height,
            "cellOffsetX": self.cell_offset_x,
            "cellOffsetY": self.cell_offset_y,
            "cellWidth": self.cell_width,
            "cellHeight": self.cell_height,
            "path": self.local_file_uri,
        }

        return json.dumps(dnd_json_as_dict)

    def print_prediction_scores(self):
        """Prints the bounding boxes and confidence scores to console"""
        for count, id in enumerate(self.confidences):
            print(f"Display name: {self.confidences[count]}")
            print(f"Bounding boxes: {self.bboxes[count]}\n\n")

    def store_image_as_dataset_row(
        self,
        gcs_bucket,
        gcs_prefix,
        *,
        training_data_file=None,
        use_prediction_results=True,
    ):
        """Saves image and bounding boxes as Vertex AI training data row.

        Args:
            gcs_bucket: the Cloud Storage bucket to store the image, without 'gs://'
            gcs_prefix: the 'folder' in the bucket to store the image
            training_data_file: Optional. The Cloud Storage URI of the file to
                append this training data to. Must be in gcs_bucket provided in
                args
            use_prediction_results: Optional. If false, the normalized results
                for this image are stored as training data.
        """

        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket)

        # Step 1. Save the image to GCS
        if not self.local_file_uri and not self.gcs_file_uri:
            raise AttributeError("Neither local nor GCS URI set")

        elif not self.gcs_file_uri:
            self.upload_local_image_to_gcs(gcs_bucket, gcs_prefix)

        # Step 2. Determine whether the training manifest file exists already
        if training_data_file:
            # TODO(telpirion): Add code to download, verify training data file
            pass
        else:
            # If no training data file is provided, assume that the file name
            # is 'index.jsonl' and it is in the same prefix/folder as the image
            training_data_file = f"{gcs_prefix}/index.jsonl"

        training_data_in_bucket = bucket.blob(training_data_file)

        # Step 3. Get or create the training manifest file
        training_data = ""
        if training_data_in_bucket.exists():
            training_data = training_data_in_bucket.download_as_bytes()
            training_data = training_data.decode("utf-8")

            # Assume that we need to add a new line feed to the downloaded data
            training_data += "\n"

            # Delete the old blob now that it's no longer needed.
            bucket.delete_blob(training_data_file)

        # Step 4. Update the training manifest file
        if use_prediction_results:
            data_row = {
                "imageGcsUri": self.gcs_file_uri,
                "boundingBoxAnnotations": self.bboxes,
            }
        else:
            # TODO(telpirion): Add code to upload normalized training data
            pass

        # Prepare data for training
        data_row = self._prepare_data_for_training(data_row=data_row)
        training_data = training_data + json.dumps(data_row)

        # Step 5. Save the updated training manifest file back to the bucket
        updated_training_data_file = bucket.blob(training_data_file)
        updated_training_data_file.upload_from_string(training_data)

    def upload_local_image_to_gcs(self, gcs_bucket, gcs_prefix):
        """Saves a copy of this file to Google Cloud Storage.

        Args:
            gcs_bucket: the bucket to store the image to
            gcs_prefix: the folder in the bucket to save to.
        """
        storage_client = storage.Client()
        file_name = self.local_file_uri.split("/")[-1]
        bucket = storage_client.bucket(gcs_bucket)
        blob = bucket.blob(f"{gcs_prefix}/{file_name}")

        # Check whether this file is already uploaded.
        if not blob.exists():
            blob.upload_from_filename(self.local_file_uri)

        self.gcs_file_uri = f"gs://{gcs_bucket}/{gcs_prefix}/{file_name}"

    def download_gcs_image_to_local(self):
        """Save an image from Cloud Storage to the local environment"""
        storage_client = storage.Client()

        # Assume that the GCS URI was saved as 'gs://bucket/prefix/filename'
        image_bucket_uri = self.gcs_file_uri.split("/")[0]
        image_file_name = self.gcs_file_uri.split("/")[-1]
        image_bucket = storage_client.bucket(image_bucket_uri)

        # Ensure that the GCS image exists
        blob_names = image_bucket.list_blobs()
        filtered_blobs = filter(
            lambda blob: blob.name.find(image_file_name) > -1, blob_names
        )

        if len(filtered_blobs) == 0:
            raise NameError("Check image GCS URI")

        # Make a tmp directory
        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        # Download file to local
        blob = filtered_blobs[0]
        self.local_file_uri = f"tmp/{image_file_name}"
        blob.download_to_filename(self.local_file_uri)

    def _prepare_data_for_training(self, data_row):
        """Updates bounding box data for model training.

        This method adds a label to all bounding boxes (CELL_LABEL) and drops
        all of the training data below the confidence threshold

        Args:
            data_row: the row of data used for model training
        """
        training_bbox_data = []
        for count, bbox in enumerate(data_row["boundingBoxAnnotations"]):

            confidence = self.confidences[count]
            if confidence < self.CONFIDENCE_THRESHOLD:
                break

            training_bbox_data.append(
                {
                    "displayName": self.CELL_LABEL,
                    "xMin": bbox[0],
                    "yMin": bbox[1],
                    "xMax": bbox[2],
                    "yMax": bbox[3],
                }
            )

        data_row["boundingBoxAnnotations"] = training_bbox_data
        return data_row

    def _show(self, bboxes):
        """PRIVATE. Renders the image with bounding boxes overlaid on top.

        Args:
          bboxes: the bounding boxes to draw on top
        """
        ia.seed(1)

        image = imageio.imread(self.local_file_uri)

        bbs = BoundingBoxesOnImage(bboxes, shape=image.shape)
        ia.imshow(bbs.draw_on_image(image, size=2))

    def _compute_grid_based_boxes(self):
        """PRIVATE. Calculates new boundings boxes based upon predicted grid-cell width and height"""
        grid_based_boxes = []
        current_x = self.cell_offset_x
        current_y = self.cell_offset_y

        grid_cell_width = math.floor(self.width / self.grid_cells_width)
        grid_cell_height = math.floor(self.height / self.grid_cells_height)

        while current_x < self.width:
            while current_y < self.height:
                x_min = current_x
                x_max = current_x + grid_cell_width
                y_min = current_y
                y_max = current_y + grid_cell_height

                grid_based_boxes.append(
                    BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
                )

                current_y = current_y + grid_cell_height

            current_x = current_x + grid_cell_width
            current_y = self.cell_offset_y

        return grid_based_boxes

    def _compute_normalized_bboxes(self):
        """PRIVATE. Calculates new boundings boxes based upon averge size of predicted bounding boxes."""
        if self.bboxes_on_image is None:
            self._compute_actual_bboxes()

        # Assumption 1. The confidence scores are ordered High => Low
        # Assumption 2. The confidences and bboxes arrays are in sync
        # Assumption 3. The bboxes are perfect squares
        # Assumption 4. The bboxes have a tendency to be larger than the actual grid squares
        top_confidences = None
        if len(self.confidences) > 10:
            top_confidences = self.confidences[0:10]
        else:
            top_confidences = self.confidences

        widths = []
        heights = []
        for count, value in enumerate(top_confidences):
            current_bbox = self.bboxes[count]
            widths.append(current_bbox[1] - current_bbox[0])
            heights.append(current_bbox[3] - current_bbox[2])

        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)

        self.cell_width_percent = avg_width
        self.cell_height_percent = avg_height

        self.cell_width = math.ceil(self.cell_width_percent * self.width)
        self.cell_height = math.ceil(self.cell_height_percent * self.height)

        current_x = self.cell_offset_x
        current_y = self.cell_offset_y

        while current_x < self.width:
            while current_y < self.height:
                x_min = current_x
                x_max = current_x + self.cell_width - self.drift_amount
                y_min = current_y
                y_max = current_y + self.cell_height - self.drift_amount

                self.normalized_bboxes.append(
                    BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
                )

                current_y = current_y + self.cell_height - self.drift_amount

            current_x = current_x + self.cell_width - self.drift_amount
            current_y = self.cell_offset_y

    def _compute_actual_bboxes(self):
        """PRIVATE. Converts predicted bounding boxes as  pixel values"""
        self.bboxes_on_image = []

        for bbox in self.bboxes:
            x_min = bbox[0] * self.width
            x_max = bbox[1] * self.width
            y_min = bbox[2] * self.height
            y_max = bbox[3] * self.height

            self.bboxes_on_image.append(
                BoundingBox(x1=x_min, x2=x_max, y1=y_min, y2=y_max)
            )


def analyze_annotation_results(result, *, local_file_uri) -> ProcessedGridImage:
    """Converts a Vertex AI image object detection prediction to a ProcessedGridImage.

    Args:
      local_file_uri: filepath to a local copy of the image.

    Returns:
      ProcessedGridImage
    """
    bboxes = result["prediction"]["bboxes"]
    confidences = result["prediction"]["confidences"]

    # Use PIL to get the width and height of the image
    image = PIL.Image.open(local_file_uri)
    width, height = image.size
    image.close()

    return ProcessedGridImage(
        width=width,
        height=height,
        local_file_uri=local_file_uri,
        bboxes=bboxes,
        confidences=confidences,
    )
