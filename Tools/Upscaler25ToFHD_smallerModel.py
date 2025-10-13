from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import time
print(torch.__version__)
print("Wersja CUDA (w PyTorch):", torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = torch.jit.load("E:/pythonProject/UpscalerTest/Model_17/training_2025-10-11_10-47-04/Model.pt")


model.eval()

scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "model_scripted.pt")


def tensor_to_pil(tensor):
    # Upewnij się, że tensor jest na CPU i ma wartości 0-1
    tensor = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)

def process_image(input_path, output_path):
    times = []
    fps = []

    start_t1 = time.time()
    img = Image.open(input_path).convert("RGB")
    w, h = img.size

    tile_w, tile_h = 192, 108
    upscale_w, upscale_h = 384, 216

    tiles = []
    transform = transforms.ToTensor()
    for y in range(0, h, tile_h):
        row = []
        for x in range(0, w, tile_w):
            crop = img.crop((x, y, x + tile_w, y + tile_h))
            image_to_model = transform(crop).to(device)
            temp_t = time.time()
            output_tensor = model(image_to_model)
            t_end = time.time()
            times.append(t_end - temp_t)
            fps.append(1 / (t_end - temp_t))
            upscaled = tensor_to_pil(output_tensor)
            if upscaled.size != (upscale_w, upscale_h):
                raise ValueError(f"Upscaler zwrócił zły rozmiar: {upscaled.size}")
            row.append(upscaled)
        tiles.append(row)

    out_w, out_h = upscale_w * 5, upscale_h * 5
    output_img = Image.new("RGB", (out_w, out_h))

    for j, row in enumerate(tiles):
        for i, tile in enumerate(row):
            output_img.paste(tile, (i * upscale_w, j * upscale_h))

    output_img.save(output_path)

    total_time = time.time() - start_t1
    print(total_time)
    print(times)
    print(fps)

    print(f"Zapisano wynik do {output_path}")


if __name__ == "__main__":

    process_image("screenshot_31.png", "fullyUpscaler_31_na25_smaller.png")
