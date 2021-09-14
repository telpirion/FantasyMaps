import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import json
import math
import PIL

class ProcessedGridImage():
  # A "fudge factor" -- the original training data laid bboxes outside of the bounds of the
  # actual gridlines in order to capture the grid lines.
  DRIFT_AMOUNT = 5

  def __init__(self, width, height, local_file_uri, bboxes, confidences):
    self.width = width
    self.height = height
    self.bboxes = bboxes
    self.confidences = confidences
    self.local_file_uri = local_file_uri
    self.bboxes_on_image = []
    self.normalized_bboxes = []
    self.cell_width = 0
    self.cell_height = 0
    self.cell_width_percent = 0
    self.cell_height_percent = 0
    self.cell_offset_x = 0 # Assuming a 0 offset map
    self.cell_offset_y = 0 # Assuming a 0 offset map
    
    self._compute_actual_bboxes()
    self._compute_normalized_bboxes()
    
    self.grid_cells_width = math.ceil(self.width / self.cell_width)
    self.grid_cells_height = math.ceil(self.height / self.cell_height)

  def __str__(self):
    return str(self.get_normalized_json_dnd())

  def show(self):
    # Begin plotting bounding box results
    ia.seed(1)
    %matplotlib inline

    image = imageio.imread(self.local_file_uri)

    if len(self.bboxes_on_image) == 0:
      self._compute_actual_bboxes()

    bbs = BoundingBoxesOnImage(self.bboxes_on_image, shape=image.shape)
    ia.imshow(bbs.draw_on_image(image, size=2))

  def show_normalized_results(self, *, is_grid_based=False):

    # Begin plotting bounding box results
    ia.seed(1)
    %matplotlib inline

    image = imageio.imread(self.local_file_uri)
    
    if not is_grid_based:
        if len(self.normalized_bboxes) == 0:
          self._compute_normalized_bboxes()

        bbs = BoundingBoxesOnImage(self.normalized_bboxes, shape=image.shape)
        ia.imshow(bbs.draw_on_image(image, size=2))
        
    else: 
        grid_based_boxes = self._compute_grid_based_boxes()
        
        bbs = BoundingBoxesOnImage(grid_based_boxes, shape=image.shape)
        ia.imshow(bbs.draw_on_image(image, size=2))

  def get_normalized_json_dnd(self):

    if len(self.normalized_bboxes) == 0:
      self._compute_normalized_bboxes()

    dnd_json_as_dict = {
      "imageWidth": self.width, 
      "imageHeight": self.height,
      "cellOffsetX": self.cell_offset_x,
      "cellOffsetY": self.cell_offset_y,
      "cellWidth": self.cell_width,
      "cellHeight": self.cell_height,
      "path": self.local_file_uri
    }

    return json.dumps(dnd_json_as_dict)


  def _compute_grid_based_boxes(self):
    
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
            
            grid_based_boxes.append(BoundingBox(x1=x_min,
                                        x2=x_max,
                                        y1=y_min, 
                                        y2=y_max))

            current_y = current_y + grid_cell_height
        
        current_x = current_x + grid_cell_width
        current_y = self.cell_offset_y
        
    return grid_based_boxes

  def _compute_normalized_bboxes(self):

    if self.bboxes_on_image is None:
      self._compute_actual_bboxes()

    # Assumption 1. The confidence scores are ordered High => Low
    # Assumption 2. The confidences and bboxes arrays are in sync
    # Assumption 3. The bboxes are perfect squares
    # Assumption 4. The bboxes have a tendency to be larger than the actual grid squares
    top_confidences = None
    if (len(self.confidences) > 10):
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
            x_max = current_x + self.cell_width - self.DRIFT_AMOUNT
            y_min = current_y
            y_max = current_y + self.cell_height - self.DRIFT_AMOUNT
            
            self.normalized_bboxes.append(BoundingBox(x1=x_min,
                                        x2=x_max,
                                        y1=y_min, 
                                        y2=y_max))

            current_y = current_y + self.cell_height - self.DRIFT_AMOUNT
        
        current_x = current_x + self.cell_width - self.DRIFT_AMOUNT
        current_y = self.cell_offset_y
            


  def _compute_actual_bboxes(self):
    self.bboxes_on_image = []

    for bbox in self.bboxes:
      x_min = bbox[0]*self.width
      x_max = bbox[1]*self.width
      y_min = bbox[2]*self.height
      y_max = bbox[3]*self.height

      self.bboxes_on_image.append(BoundingBox(x1=x_min,
                                        x2=x_max,
                                        y1=y_min, 
                                        y2=y_max))


def analyze_annotation_results(result, *, local_file_uri):

  bboxes = result['prediction']['bboxes']
  confidences = result['prediction']['confidences']

  # Use PIL to get the width and height of the image
  image = PIL.Image.open(local_file_uri)
  width, height = image.size
  image.close()

  return ProcessedGridImage(
      width=width,
      height=height,
      local_file_uri=local_file_uri,
      bboxes=bboxes,
      confidences=confidences)