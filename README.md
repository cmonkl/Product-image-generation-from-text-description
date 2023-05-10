# Project description
![alt text](/imgs/example.png "Generated image and real image")

In this project we aim to work on image generation from textual descriptions. In particular, we want to generate a photo of the product (clothes) from online shop based on its textual description. The problem of creating an image for products in online shops arises from having only textual description of the product available, while it is desirable to have an image representation of the product to draw attention of buyers. Also, this project can have a great impact to the area of fashion, because generated images can be an inspiration for designers and manufacturers. 

Inspired by recent developments of diffusion models on text-to-image generation field, we use [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) as our main model. The data for traininig and evaluation is taken from fashion product [dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset). The fine-tuned model can be downloaded using link [Fine-Tuned model](https://drive.google.com/file/d/1h2AfpmAACvM3ShbuXHDP0MFKgI3hqQzF/view?usp=sharing). Training the model requires 16 GB of GPU memory (available on Google Colab or Kaggle Notebooks). Our model was trained for 15 epochs during 24 hours. 
 
# Project technology
The project is served on Docker. To improve performance of the model on CPU device, we use OpenVINO convertion of parameters. 

# Project components:
- data/: Contains description files for dataset items;
- code/: Contains .py files;
- notebooks/: Contains .ipynb files.

# Running training:
1. Fill in arguments for training in code/config.py
2. Run notebooks/dreambooth.ipynb

# Evaluation on Docker:
1. Run docker file
2. Write prompt