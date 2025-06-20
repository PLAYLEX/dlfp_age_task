# Deep Learning for Practitioners: Age Task
This repository contains the group work and partial solutions for the AI coding challenge "age task" discussed in "Deep Learning for Practitioners" at TU Chemnitz.

Each group member has his/her own branch with his/her own solution.

## Setup
1. Install python (e.g. 3.12)
2. Choose an favorite virtual environment and package manager: e.g. venv with pip
   1. `python -m venv .venv`
   2. `.\.venv\Scripts\Activate.ps1` --> Activate the virtual environment (after activation visible in terminal)
3. Install all necessary packages:
   1. `pip install ipywidgets -U`
   2. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -U` --> select an appropriate version for your hardware (GPU)

## Task Description
Create an AI Model, that is feasible to detect the **age** of a person (image).

The given Datasets:
- [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new/data)
- [AgeDB](https://www.kaggle.com/datasets/nitingandhi/agedb-database)

### Overview:

- We should create only **one** group solution.
- The task itself is again about the **age task** given the **UTKFace** dataset (and optional the **AgeDB** dataset).
  - The whole task is split in two parts: **Pre-training and Fine-tuning**
- But we **must** solve the task with a **Vision Transformer** (ViT) based architecture.
  - only our team **must** implement the **CoAtNet** architecture based on this paper: [CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/abs/2106.04803)
- in the end, we need a **presentation** describing our solution

### Task Description:

#### Preparation:
- implement **CoAtNet**
- prepare data (load data, transformation, augmentation, ...)
  - *maybe for each part different preparation needed*
- design our complete pipeline (from data preparation ... to final evaluation)

#### Part 1: Pre-training
- We should **pre-train** our CoAtNet with any **self-supervised learning** (SSL) approach (*see Chapter 6 on Opal*)
  - *this pre-trained CoAtNet should be saved, so it can be used in Part 2*
- for pre-training we should use AgeDB **or** UTKFace dataset
- **evaluate** this pre-trained CoAtNet with (at least):
  - **t-sne of latent space**

#### Part 2: Fine-tuning
- now we should **fine-tune** the pre-trained CoAtNet for the **age task** (*basically what we did in practical unit 1, so we could reuse our code*)
  - *we should be able to load the pre-trained CoAtNet and prepare for finetuning*
  - *this fine-tuned CoAtNet should be saved, so it can be used in evaluation*
- for fine-tuning we should use **only** the UTKFace dataset
- **evaluate** this fine-tuned CoAtNet with (at least):
  - **accuracy** of 4 classes (<18, 18-40, 41-60, >60)
  - example classification of **self-made images** of group members

### Presentation:

Finally, we should **present our work**:

#### Preparation:
1. What is CoAtNet and what is different to the other architectures?
2. How did we prepare the datasets? Are there differences for Pre-training and Fine-Tuning?
3. How does our complete pipeline look like?
   - (How did we organize the work?)

#### Pre-training:
4. How did we implement the CoAtNet?
5. Which SSL approach did we use? Why? Hyperparameters?
6. What are the evaluation results for pre-training?

#### Fine-tuning:
7. What was our training-setup?
8. What are the evaluation results for fine-tuning? Comparison to our results in practical unit 1.