import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#model = torch.jit.load("E:/pythonProject/UpscalerTest/Model_17/training_2025-10-11_10-47-04/Model.pt")
model = torch.jit.load("E:/pythonProject/UpscalerTest/Model_20/training_2025-10-15_21-11-40/Model.pt")

#model = model.cpu()
model.eval()

dummy_input = torch.randn(1, 3, 108, 192)#, dtype=torch.float32, device="cpu")
dummy_input = dummy_input.to(device)
torch.onnx.export(
    model,
    dummy_input,
    "E:/pythonProject/UpscalerTest/Model_20/wojtekUpscaler_m20_5.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

pass


scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "model_scripted.pt")




def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    to_pil = transforms.ToPILImage()
    return [to_pil(t) for t in tensor]

def process_image(input_path, output_path, model, device, batch_size=8):
    start_t1 = time.time()

    img = Image.open(input_path).convert("RGB")
    w, h = img.size

    tile_w, tile_h = 192, 108
    upscale_w, upscale_h = 384, 216

    transform = transforms.ToTensor()

    tiles = []
    coords = []

    for y in range(0, h, tile_h):
        for x in range(0, w, tile_w):
            crop = img.crop((x, y, x + tile_w, y + tile_h))
            tiles.append(transform(crop))
            coords.append((x, y))

    model.eval()
    results = []
    times = []

    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch = torch.stack(tiles[i:i+batch_size]).to(device)

            torch.cuda.synchronize()
            t0 = time.time()
            output_batch = model(batch)
            torch.cuda.synchronize()
            t1 = time.time()

            times.append(t1 - t0)
            results.extend(tensor_to_pil(output_batch))

    out_w = (w // tile_w) * upscale_w
    out_h = (h // tile_h) * upscale_h
    output_img = Image.new("RGB", (out_w, out_h))

    for (x, y), tile in zip(coords, results):
        i = x // tile_w
        j = y // tile_h
        output_img.paste(tile, (i * upscale_w, j * upscale_h))

    output_img.save(output_path)
    total_time = time.time() - start_t1
    fps = len(tiles) / sum(times)

    print(f"Zapisano wynik do: {output_path}")
    print(f"Czas całkowity: {total_time:.2f}s")
    print(f"Kafelki: {len(tiles)}")
    print(f"Średni czas batchu: {np.mean(times):.3f}s")
    print(f"FPS (dla batchy): {fps:.2f}")


if __name__ == "__main__":

    process_image("screenshot_31.png", "fullyUpscaler_31_na25_smaller_batches.png", model, device, 25)
