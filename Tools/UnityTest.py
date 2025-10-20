from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import time
import io
from io import BytesIO
import base64
import sys

b64_data = sys.stdin.read()
img_data = base64.b64decode(b64_data)

img = Image.open(BytesIO(img_data))

img.save("received_image.png")

buffered = BytesIO()
img.save(buffered, format="PNG")
result_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

print(result_b64)