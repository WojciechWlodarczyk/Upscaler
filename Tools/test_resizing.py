import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms




input_img_14258_half_FHD = Image.open(f"D://mojeAI//MyUpscalerDataSet//cut25//half_FHD//piece_83.png").convert('RGB')  ######################################################################################  USUN!!!!!!
transform = transforms.ToTensor()  ######################################################################################  USUN!!!!!!
inputs_piece_14258 = transform(input_img_14258_half_FHD)  ######################################################################################  USUN!!!!!!

inputs_piece_14258 = inputs_piece_14258.unsqueeze(0)  # (1, C, H, W)
inputs_piece_14258 = F.interpolate(inputs_piece_14258, size=(216, 384), mode='bilinear', align_corners=False)
inputs_piece_14258 = inputs_piece_14258.squeeze(0)    # z powrotem (C, H, W)

to_pil = transforms.ToPILImage()
resized_img = to_pil(inputs_piece_14258)

# 6️⃣ Zapis do pliku
output_path = "D://mojeAI//MyUpscalerDataSet//cut25//result//piece_83_resized.png"
resized_img.save(output_path)