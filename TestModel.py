import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os




def run_single_image(model, image_path, output_dir="D:/mojeAI/MyUpscalerDataSet/outputFHD"):
    from PIL import Image
    # Utworzenie folderu output2, jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)

    # Wczytanie i przygotowanie obrazu
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((540, 960)),  # dopasuj do wymagań wejściowych modelu
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0)

    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Przepuszczenie przez model
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    # Konwersja do obrazów do wyświetlenia i zapisu
    input_img_show = input_tensor.squeeze(0).permute(1,2,0).cpu().numpy() # Move back to CPU for plotting
    output_img_show = output.squeeze(0).permute(1,2,0).cpu().numpy() # Move back to CPU for plotting

    # Wyświetlenie obrazów
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Wejście")
    plt.imshow(input_img_show)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Wyjście")
    plt.imshow(output_img_show)
    plt.axis('off')
    plt.show()

    # Zapisanie wyjściowego obrazu
    from PIL import Image
    import numpy as np
    output_img = (output_img_show * 255).astype(np.uint8)
    output_pil = Image.fromarray(output_img)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    output_pil.save(output_path)


def SimpleTest(model, path):
    output_dir = path + "//output"

    image_path = 'D:/mojeAI/MyUpscalerDataSet/half_FHD/screenshot_1.png'

    run_single_image(model, image_path, output_dir)

    image_path = 'D:/mojeAI/MyUpscalerDataSet/half_FHD/screenshot_5.png'

    run_single_image(model, image_path, output_dir)

    image_path = 'D:/mojeAI/MyUpscalerDataSet/half_FHD/screenshot_15.png'

    run_single_image(model, image_path, output_dir)