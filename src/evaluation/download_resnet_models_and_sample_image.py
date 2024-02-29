import torch
import torchvision.models as models

workload_ids = [1, 7, 21, 31, 37, 50, 55, 56, 69, 71]

for i in workload_ids:
    resnet152 = models.resnet152(pretrained=True)
    torch.save(resnet152.state_dict(), f'resnet152_weights_{i}.pth')

import urllib.request

# URL of the sample image
image_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
# Path to save the image locally
image_path = "sample_image.jpg"

# Download the image
urllib.request.urlretrieve(image_url, image_path)