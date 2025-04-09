<div align="center">
  <img src="/Imgs/website.gif" width="90%"/>
</div>
</br>

Neural Style Transfer (NST) refers to a class of software algorithms that manipulate digital images, or videos, in order to adopt the appearance or visual style of another image. NST algorithms are characterized by their use of deep neural networks for the sake of image transformation. Popular use cases for NST are the creation of artificial artwork from photographs, for example by transferring the appearance of famous paintings to user-supplied photographs.

<br> <!-- line break -->

<div align="center">
<img src="/Images/nst.png"/>
</div>

<br> <!-- line break -->


## 🎯 Objective 
The main goal of this project is to explore Neural-style-transfer through implementation. We'll Implement a NST model using Tensorflow and keras, and at the end of the project we'll deploy it as a web app so that anyone can create stunning digital art which they could even sell as NFT's.

<br> <!----line break-->
## 📝 Summary of Neural Style Transfer

Style transfer is a computer vision technique that takes two images — a "content image" and "style image" — and blends them together so that the resulting output image retains the core elements of the content image, but appears to be “painted” in the style of the style reference image. Training a style transfer model requires two networks,which follow a encoder-decoder architecture : 
- A pre-trained feature extractor 
- A transfer network


<div align="center">
<img src="/Images/nst architecture.jpg" width="80%"/>
</div>

<br> <!-- line break -->



The ‘encoding nature’ of CNN’s is the key in Neural Style Transfer. Firstly, we initialize a noisy image, which is going to be our output image(G). We then calculate how similar is this image to the content and style image at a particular layer in the network(VGG network). Since we want that our output image(G) should have the content of the content image(C) and style of style image(S) we calculate the loss of generated image(G) w.r.t to the respective content(C) and style(S) image.



<div align="center">
<img src="/Images/final_oss.png" width="50%" />
</div>

<br> <!-- line break -->

## 👨‍💻 Implementation

Early implementations of Neural Style Transfer (NST) approached the task as an optimization problem. These methods required hundreds to thousands of iterations to stylize a single image, making real-time applications impractical. While effective, this technique was computationally intensive and time-consuming.

To address these limitations, researchers introduced Fast Neural Style Transfer (Fast-NST) [here](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2) — a breakthrough that significantly accelerates the process. Unlike the optimization-based methods, Fast-NST leverages a pre-trained feed-forward neural network that learns to stylize images in a single forward pass. This results in near-instant style transfer while maintaining high visual quality.

Modern state-of-the-art style transfer models have further evolved to support multi-style learning, allowing a single model to apply multiple artistic styles to any input image. This paves the way for highly flexible and creative image transformation pipelines.

## To run locally

1. Download the pre-trained TF model.

    - The 'model' directory already contains the pre-trained model,but you can also download the pre-trained model from [here](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2).

2. Import this repository using git command
```
git clone https://github.com/cyrus0001/Neural_Style_Transfer/.git
```
3. Install all the required dependencies inside a virtual environment
```
pip install -r requirements.txt
```
4. Copy the below code snippet and pass the required variable values
```python
import gradio as gr
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# Path of the downloaded pre-trained model or 'model' directory
model_path = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
print("Loading model from TensorFlow Hub...")
hub_module = hub.load(model_path)


content_image_path = r"C:\Users\Pictures\my_pic.jpg"
style_image_path = r"C:\Users\Desktop\images\my_style.jpg"

img = transfer_style_gradio(content_image_path,style_image_path,model_path)
# Saving the generated image
plt.imsave('stylized_image.jpeg',img)
plt.imshow(img)
plt.show()
```
<br> <!--line break-->
## Live Link
<div align="center">
        <a href="https://huggingface.co/spaces/cyrus007/neural-style-transfer" target="_blank">
            Click here to open Neural Style Transfer Demo
        </a>
</div>







