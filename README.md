# DMAMP: A deep-learning model for detecting antimicrobial peptides and multi-activities

This is a predictor for antimicrobial peptides and specific activities prediction simultaneously, called DMAMP (Deep-learning Model for AntiMicrobial Peptides).

![image](https://github.com/unimqz/DMAMP/blob/main/flowchart.png)

## Requirements

The prediction pipeline uses Python3 and requires the following modules:
* Numpy;
* Keras;
* Tensorflow;
* Sklearn;
* Matplotlib;

## Load the pre-trained model

Please execute the following code to load the model if you want to predict some antimicrobial peptides.
```
from keras.models import load_model
model_path = r'model/final_model_710_epoch20_lr0.0001_bc10_task11_2'
model = load_model(model_path, custom_objects={'weight_sample_loss': weight_sample_loss, 'f1_metric':f1_metric, 'recall_metric': recall_metric, 'precision_metric': precision_metric})
```
## Datasets

bench_878_amp.fa, bench_2405_nonamp.fa: The positive and negative datasets of the training datasets.

pos710.fa, neg710.fa: The positive and negative datasets of the test datasets.

'*.pickle' is the features files in the paper. We used the sixth feature group to realize the work. Thus, please call the sixth feature group in case the feature dimension is mismatching.
