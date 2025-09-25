
import torch

if torch.cuda.is_available():
    print(f"GPU dostępne: {torch.cuda.get_device_name(0)}")
else:
    print("Używam tylko CPU")







print(torch.__version__)        # 2.0.1+cu117
print(torch.version.cuda)       # 11.7
print(torch.cuda.is_available()) # True