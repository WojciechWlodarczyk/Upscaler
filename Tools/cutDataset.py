from PIL import Image
import os

path_out_fhd = 'D:\mojeAI\MyUpscalerDataSet\cut25\FHD_final_test'
path_out_half_fhd = 'D:\mojeAI\MyUpscalerDataSet\cut25\half_FHD_final_test'



def split_image(image_path, output_dir, rows=5, cols=5, count = 0):
    # Otwórz obraz
    img = Image.open(image_path)
    width, height = img.size

    # Oblicz szerokość i wysokość pojedynczego kawałka
    tile_width = width // cols
    tile_height = height // rows

    # Utwórz folder na wynik
    os.makedirs(output_dir, exist_ok=True)

    # Podziel na kawałki
    for row in range(rows):
        for col in range(cols):
            left = col * tile_width
            upper = row * tile_height
            right = left + tile_width
            lower = upper + tile_height

            # Wytnij fragment
            cropped = img.crop((left, upper, right, lower))

            # Zapisz kawałek
            cropped.save(os.path.join(output_dir, f"piece_{count}.png"))
            count += 1

    #print(f"Podzielono na {count} kawałków i zapisano w {output_dir}")

def split_both_images(index):
    fhd_input = 'D:\mojeAI\MyUpscalerDataSet\FHD_final_test\screenshot_' + str(index + 1) + '.png'
    half_fhd_input = 'D:\mojeAI\MyUpscalerDataSet\half_FHD_final_test\screenshot_' + str(index + 1) + '.png'

    count = index * 25 + 1
    split_image(fhd_input, path_out_fhd, 5, 5, count)
    split_image(half_fhd_input, path_out_half_fhd, 5, 5, count)

    print(f'Cut image: {index}')



for i in range(0, 250):
    split_both_images(i)
