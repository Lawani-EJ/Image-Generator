<!-- ## First commit in setting up the enviroment and installing gradio
1. mkdir on cmd called AnimeCharImg-Gnrtr
2. Create the virtual environment and activate the virutal environment
3. Install the required packages pip install --upgrade gradio
4. Create a simple demo with gradio and launch 


# Output 😁⭐😊
### Screen Recording Output 
![Screenrecording](./assets/Gradio-Opera2024-11-2813-52-05-ezgif.com-video-to-gif-converter.gif)

### Screen Shot Output
![Screenshot](./assets/Gradio%20-%20Opera%2011_28_2024%201_52_31%20PM.png) -->

## Setting up the demo application 
This creates a Gradio interface using the gr.Interface class. It specifies the make_image function as the main function, defines the input sliders and their ranges, sets the title and CSS styling, and specifies the output as an image with the specified width and height. Finally, the .launch() method is called to launch the Gradio interface.

Creating the application The code sets up a Gradio interface for creating anime characters using a pre-trained model. Here is a brief explanation of each step

1. Import the necessary libraries for image transformation, PyTorch, and Gradio.
```python
import requests
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from PIL import Image  # For resizing images with high quality
import gradio as gr
```

2. Download the pretrained model and save it in the system.
```python
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0GD8EN/G_trained.pth"
response = requests.get(url)


with open("G_trained.pth", "wb") as f:
    f.write(response.content)
```

3. This defines the size of the latent vector, which is the input to the generator.
```python
# This defines the size of the latent vector, which is the input to the generator. 
latent_vector_size = 128
```


4. This block of code defines a custom generator class that inherits `fromnn.Modulein` PyTorch. It specifies the architecture of the generator network
```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_vector_size, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
```


5. This creates an instance of the Generator class, which represents the generator network.
```python
# Load the generator model
G = Generator()
```


6. These lines set the device to CPU and load the saved state dictionary of the generator model from theG_trained.pthfile.
```python
device = torch.device("cpu")
G.load_state_dict(torch.load("G_trained.pth", map_location=device))
```


7. This defines a function named make_image that takes input parameters a, b, value and generates an image using the generator network.
```python
def make_image(a, b, value):
```

8. This generates an image by passing the latent vector z through the generator network G. The resulting output is detached from the computational graph, and the dimensions are squeezed to remove any singleton dimensions.
```python
    z = a * torch.randn(1, latent_vector_size, 1, 1) + b
    Xhat = G(z)[0].detach().squeeze(0)
```


9. This performs min-max normalization on the image tensor Xhat, scaling its values between 0 and 1.
```python
    Xhat = (Xhat - Xhat.min()) / (Xhat.max() - Xhat.min())
```


10. These lines define the title and CSS styling for the Gradio interface.
```python
title = "Anime Character Generator"
css = ".output_image {height: 60rem !important; width: 100% !important;}"
```

## **Expected Outputs**

### **Screen Recording Output**
![Screen Recording](./assets/AnimeCharacterGenerator-Opera2024-12-0206-11-15-ezgif.com-video-to-gif-converter.gif)

### **Screenshot**
![Screenshot](./assets/Screenshot%20(175).png)

---

## **Features**

- **Interactive Sliders**: Adjust variation, bias, and latent variable value.
- **High-Quality Resizing**: Output images are resized to 512x512 using high-quality filtering.
- **Gradio Interface**: Simple and intuitive UI for generating anime characters.

---

## **How It Works**

1. **Model Initialization**: Loads a pre-trained generator model.
2. **Image Generation**:
   - A latent vector (`z`) is created using random noise scaled by the `Variation` slider.
   - The generator processes `z` to create a raw image tensor.
   - The image is normalized and converted to a high-quality PIL image.
3. **Gradio Interface**:
   - Users adjust sliders to control `Variation`, `Bias`, and `Fine Tune`.
   - The app generates and displays anime-style character images.

---
