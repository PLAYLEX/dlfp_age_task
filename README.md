# Pre training:Self-supervised learning with SimCLR and CoAtNet
Mostly based on: [Tutorial 6.1](https://github.com/hamkerlab/DL_for_practitioners/blob/main/06_1_SSL_SimCLR/06_1_SSL_SimCLR.ipynb)

## Data preparation
Dataset: [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new/data) <br>
Extract image paths and ages. <br>
Check class distribution -> imbalanced <br>
Split dataset into training (70%), validation (15%), and test sets (15%).   <br>

## Setup
Use functions **LinearLayer**, **ProjectionHead** from [Tutorial 6.1](https://github.com/hamkerlab/DL_for_practitioners/blob/main/06_1_SSL_SimCLR/06_1_SSL_SimCLR.ipynb)


Create **UTKFaceSimCLRDataset** similar to **DataSetAugment** for creating two different augmented views along with the label. <br>
Goal is to make the two augmented views of the same image different enough that the model has to learn robust features but not so different that they would appear to be different images.
- For **training** use a strong augmentation with random crops, horizontal flips, color jitter, grayscale, and blur <br>
- For **validation/testing** use a mild augmentation with less aggressive random crops, horizontal flips, grayscale<br>

Create **SimCLRCoAtNetModel** encoder

- use model **coatnet_0_224** similar to SimCLR Paper
- - not pre trained
- use **128** for size of output vector **z** similar to SimCLR Paper
- Use size of num_features from model for in_features and hidden_features
- use num_classes 0 ; which prevents include the final classification layer; 
we dont want to predict classes yet, we dont want to limit it.

Use BATCH_SIZE 32 <br>

Create **DataLoader** for training, validation, and test.
- Problem: on first try forgot "drop_last=True" which caused a crash on the last pre-training loop of the epoch
because total number of samples isnt always divisable by BATCH_SIZE

Use function **SimCLR_Loss** from [Tutorial 6.1](https://github.com/hamkerlab/DL_for_practitioners/blob/main/06_1_SSL_SimCLR/06_1_SSL_SimCLR.ipynb)
- Used TEMPERATURE 0.1 for the model to make it more sensitive towards images that are not similar but that the 
model confuses as similar.

Use function **LARS** from **optimizers** helper.<br>
- SimCLR Paper LR Scaling: Base = 0.3 for BatchSize 256
- - use LR 0,0375 == (0.3 * (BATCH_SIZE / 256.0)) 
- - with higher BATCH_SIZE you get a less noisy estimate of the gradient
- Use WEIGHT_DECAY 1e-6 == from notebook 

Use 50 Epochs and 10% of the epochs for warmup from [04_1_WarmUpScheduler](https://github.com/hamkerlab/DL_for_practitioners/blob/c80d72b77250a7dee47a9e79182af424faffedea/04_1_ViT/04_1_WarmUpScheduler.ipynb#L71)

Schedulers
- Use LambdaLR warmup scheduler from [Tutorial 6.1](https://github.com/hamkerlab/DL_for_practitioners/blob/main/06_1_SSL_SimCLR/06_1_SSL_SimCLR.ipynb)
- - linearly increases the LR multiplier from near 0 up to 1.0 over warmup period
- Use CosineAnnealingLR for main scheduler. based on [04_1_WarmUpScheduler](https://github.com/hamkerlab/DL_for_practitioners/blob/c80d72b77250a7dee47a9e79182af424faffedea/04_1_ViT/04_1_WarmUpScheduler.ipynb#L71)
- - minimum learning rate 1e-6 from [04_1_WarmUpScheduler](https://github.com/hamkerlab/DL_for_practitioners/blob/c80d72b77250a7dee47a9e79182af424faffedea/04_1_ViT/04_1_WarmUpScheduler.ipynb#L71)
- Combined schedulers with **SequentialLR** scheduler

## Training loop

Used training loop similar to [Tutorial 6.1](https://github.com/hamkerlab/DL_for_practitioners/blob/main/06_1_SSL_SimCLR/06_1_SSL_SimCLR.ipynb)

- modified warmup conditional checks since **SequentialLR** scheduler is used now
- Saved also encoder with best val loss for each epoch

## Visualization

Visualized t-SNE to show how data points group together

- Used function **visualize_embeddings_tsne** from [plotting.py](https://github.com/hamkerlab/DL_for_practitioners/blob/c80d72b77250a7dee47a9e79182af424faffedea/Utils/plotting.py)
- - changed parameter of TSNE call **n_iter** to **max_iter** since this parameter did not exist.

 

---

- Problem: in one server the GPU did not have enough VRam
- - Compared results between 2 servers with BATCH_SIZE 32 and BATCH_SIZE 16
- - Both used AdamW optimizer (LEARNING_RATE = 1e-3) and 5 Epochs for testing and 1 Warmup epoch
- - the validation loss stagnated in BATCH_SIZE 32 but it progressed fine in BATCH_SIZE 16
- - Maybe the causes are that there were too less Epochs, or smalled batches helped the model
getting stuck
- - Later switched to **LARS** and BATCH_SIZE 32 
