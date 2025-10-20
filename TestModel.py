import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os


def run_single_image(model, image_path, target_image, output_dir="D:/mojeAI/MyUpscalerDataSet/cut25/outputFHD", show=True):
    from PIL import Image
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
    #    transforms.Resize((540, 960)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0)

    img_target = Image.open(target_image).convert('RGB')
    transform = transforms.Compose([
        #    transforms.Resize((540, 960)),
        transforms.ToTensor()
    ])
    target_tensor = transform(img_target).unsqueeze(0)

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    target_img_show = target_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    input_img_show = input_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    output_img_show = output.squeeze(0).permute(1,2,0).cpu().numpy()

    if show:
        plt.figure(figsize=(12,6))
        plt.subplot(1, 2, 1)
        plt.title("target")
        plt.imshow(target_img_show)
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title("upscaled")
        plt.imshow(output_img_show)
        plt.axis('off')
        plt.show()

    from PIL import Image
    import numpy as np
    output_img = (output_img_show * 255).astype(np.uint8)
    output_pil = Image.fromarray(output_img)
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace("piece", "upscaled"))
    output_pil.save(output_path)


def SimpleTest(model, path):
    output_dir = path + "//output"

    image_path = 'D:/mojeAI/MyUpscalerDataSet/cut25/half_FHD/piece_83.png'
    image_target_path = 'D:/mojeAI/MyUpscalerDataSet/cut25/FHD/piece_83.png'
    #'D:/mojeAI/MyUpscalerDataSet/half_FHD/screenshot_1.png'

    run_single_image(model, image_path, image_target_path, output_dir)

    image_path = 'D:/mojeAI/MyUpscalerDataSet/cut25/half_FHD/piece_308.png'
    image_target_path = 'D:/mojeAI/MyUpscalerDataSet/cut25/FHD/piece_308.png'

    run_single_image(model, image_path, image_target_path, output_dir)

    image_path = 'D:/mojeAI/MyUpscalerDataSet/cut25/half_FHD/piece_313.png'
    image_target_path = 'D:/mojeAI/MyUpscalerDataSet/cut25/FHD/piece_313.png'

    run_single_image(model, image_path, image_target_path, output_dir)

    image_path = 'D:/mojeAI/MyUpscalerDataSet/cut25/half_FHD/piece_436.png'
    image_target_path = 'D:/mojeAI/MyUpscalerDataSet/cut25/FHD/piece_436.png'

    run_single_image(model, image_path, image_target_path, output_dir)

    image_path = 'D:/mojeAI/MyUpscalerDataSet/cut25/half_FHD/piece_743.png'
    image_target_path = 'D:/mojeAI/MyUpscalerDataSet/cut25/FHD/piece_743.png'

    run_single_image(model, image_path, image_target_path, output_dir)

def Upscale_83(model, path):
    output_dir = path + "//output//Test"

    image_path = 'D:/mojeAI/MyUpscalerDataSet/cut25/half_FHD/piece_83.png'
    image_target_path = 'D:/mojeAI/MyUpscalerDataSet/cut25/FHD/piece_83.png'
    # 'D:/mojeAI/MyUpscalerDataSet/half_FHD/screenshot_1.png'

    run_single_image(model, image_path, image_target_path, output_dir, False)
