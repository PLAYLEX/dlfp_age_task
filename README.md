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

The given Dataset: [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new/data)

1. Analyze the given Dataset and prepare the data.
2. Choose an appropriate network architecture and train the dataset on the network.
3. Evaluate the results with at least accuracy.