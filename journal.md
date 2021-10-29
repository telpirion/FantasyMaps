# Journal

Log of attempts:

* **Try 1.** We used the top left-most cell for training the model. This resulted in predictions with a cluster of bounding boxes along the corners.
  + Dataset: `projects/147301782967/locations/us-central1/datasets/4591639722429775872`
  + Model: `projects/147301782967/locations/us-central1/models/6748089887754289152`
  + Endpoint: `projects/147301782967/locations/us-central1/endpoint/7556530001332404224`

* **Try 2.** We used the 1th cell (2nd from top, second from left) cell for training the model. This also resulted in predictions with a cluster of bounding boxes along the corners.
  + Dataset: `projects/147301782967/locations/us-central1/datasets/5740620577362673664`
  + Model: `projects/147301782967/locations/us-central1/models/8978216128232816640`
  + Endpoint: `projects/147301782967/locations/us-central1/endpoint/3245177783055286272`

* **Try 3.** _MVP_. We placed bounding boxes over ALL the cells in the training data.
  + Dataset: `projects/147301782967/locations/us-central1/datasets/6572379133542662144`
  + Model: `projects/147301782967/locations/us-central1/models/6227079705862864896`
  + Endpoint: `projects/147301782967/locations/us-central1/endpoint/6490584264529149952`

* **Try 4.** After creating an MVP online model, we tried to create an exportable, high-accuracy TF model. Unfortunately, our results, a seen in the model evaluations, were not very promising.
  + Dataset: Same as Try 3.
  + Model: `projects/147301782967/locations/us-central1/models/1526408012376309760`
  + Endpoint: `projects/147301782967/locations/us-central1/endpoint/1207721164135202816`

* **Try 5.** After the previous unsuccessful attempt to create an edge model (Try #4), we decided to try changing the training parameters a little bit: increasing the milli node hours to 100K; changing the test train split to 80/10/10 (was 70/20/10).
  + Dataset: Same as Try 3.
  + Failure. The model evaluations showed an average precision and recall of 0 (!). 

* **Try 6.** Created a pipeline that:
  + Trains two models, one online and the other Edge
  + Creates a new dataset based upon increased data
    - New dataset created by batch prediction using existing online model (MVP).
    - After batch prediction, the pipeline adds the new map & map coords data into the GCS bucket for the main pipeline flow.
  

Alternatives considered:

  + [Tensorflow image object detection](https://www.tensorflow.org/lite/examples/object_detection/overview)
  + [Hough transforms](https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549)
  + [Image convolutions](https://aishack.in/tutorials/image-convolution-examples/)

Process / work items:

  1. Get an exportable TF model.
  1. Generate C library (minimal) that calls into TF model
  1. Test C library with real files
  1. OSSPO process for library
  1. Release! 