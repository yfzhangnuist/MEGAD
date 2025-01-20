## Prepare

To run the code, you need prepare datasets and pretrain embeddings:

#### For Pre-Training

In ```./framework/pretrain/```, you need run the ```pretrain.py``` to generate pretrain embeddings.

## Run

For each dataset, create a folder in ```emb``` folder with its corresponding name to store node embeddings.

For training, run the ```main.py``` in the ```./framework``` folder, all parameter settings have default values, you can adjust them in ```main.py```.

## Test

In the training process, we evaluate the performance for each epoch.

