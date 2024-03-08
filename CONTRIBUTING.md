# Contributing to Fantasy Maps

The current roadmap for FantasyMaps is to migrate our dataset and training application
over to PyTorch. This migration provides us with greater flexibility: we can
host our code on any cloud provider; we can train our model locally; we can produce
model artifacts locally.

## Migration plan

1. Select a object detection model that we want to use. We should look at 
   [HuggingFace](https://huggingface.co/) for good options.

   + Option: [DETR](https://huggingface.co/docs/transformers/en/tasks/object_detection)

2. Convert training data / manifest to format that is correct for object detection
   model.

3. Train model locally. Potentially switch over to Mojo for execution.

4. Evaluate model results and iterate.