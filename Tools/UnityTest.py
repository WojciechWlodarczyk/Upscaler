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

# Konwersja do obrazu
img = Image.open(BytesIO(img_data))

# Przykładowa operacja: zapis do pliku
img.save("received_image.png")

# Jeśli chcesz, możesz też zmodyfikować obraz i zwrócić go do Unity:
# Tutaj np. po prostu wracamy tym samym obrazem
buffered = BytesIO()
img.save(buffered, format="PNG")
result_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

# Wysyłamy wynik przez stdout
print(result_b64)




"""
img = Image.open("E:/pythonProject/UpscalerTest/Tools/fullyUpscaler_31_na25_smaller_batches.png").convert("RGB")

# konwersja do bajtów PNG
buffered = io.BytesIO()
img.save(buffered, format="PNG")
img_bytes = buffered.getvalue()

# konwersja na Base64
img_base64 = base64.b64encode(img_bytes).decode('utf-8')

print(img_base64)
"""