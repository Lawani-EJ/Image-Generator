import requests
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from PIL import Image  # For resizing images with high quality
import gradio as gr


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0GD8EN/G_trained.pth"
response = requests.get(url)


with open("G_trained.pth", "wb") as f:
    f.write(response.content)

# This defines the size of the latent vector, which is the input to the generator. 
latent_vector_size = 128


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

# Load the generator model
G = Generator()
device = torch.device("cpu")
G.load_state_dict(torch.load("G_trained.pth", map_location=device))


def make_image(a, b, value):
    try:
        z = a * torch.randn(1, latent_vector_size, 1, 1) + b
        print(f"Latent vector shape: {z.shape}")  #Was Just Debugging


        Xhat = G(z)[0].detach().squeeze(0)
        print(f"Generated image shape: {Xhat.shape}")  #Was Just Debugging


        Xhat = (Xhat - Xhat.min()) / (Xhat.max() - Xhat.min())


        image = to_pil_image(Xhat)


        fixed_size = 512
        resized_image = image.resize((fixed_size, fixed_size), resample=Image.LANCZOS)

        resized_image.save("my_image.png", quality=95)
        return resized_image

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

title = "Anime Character Generator"
css = ".output_image {height: 60rem !important; width: 100% !important;}"

gr.Interface(
    fn=make_image,
    inputs=[
        gr.Slider(1, 10, label="Variation", value=1),
        gr.Slider(-5, 5, label="Bias", value=0),
        gr.Slider(-5, 5, label="Fine Tune: Latent Variable Value", value=0),
    ],
    title=title,
    css=css,
    outputs=gr.Image(type="pil", elem_id="output_image"),
).launch()
