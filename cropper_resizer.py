import os
from skimage import io
from skimage.transform import resize
import numpy as np

def crop_and_resize_dataset(
    input_dir,
    crop_size,
    resize_size,
    output_dir
):
    """
    Pre každý TIFF obrázok v input_dir:
      1. Oreže stredný štvorec crop_size x crop_size.
      2. Zmení veľkosť na resize_size x resize_size (antialiasing).
      3. Uloží do output_dir pod rovnakým názvom.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        # Spracuj len .tif alebo .tiff súbory (prípadne veľké/malé písmená)
        if not (fname.lower().endswith('.tif') or fname.lower().endswith('.tiff')):
            continue
        in_path = os.path.join(input_dir, fname)
        if not os.path.isfile(in_path):
            continue
        try:
            img = io.imread(in_path)
            h, w = img.shape[:2]
            # Výpočet súradníc pre stredný crop
            top = max((h - crop_size) // 2, 0)
            left = max((w - crop_size) // 2, 0)
            crop = img[top:top+crop_size, left:left+crop_size]
            # Ak crop presahuje okraj, doplní sa čiernou
            if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                crop_fixed = np.zeros((crop_size, crop_size) + crop.shape[2:], dtype=img.dtype)
                crop_fixed[:crop.shape[0], :crop.shape[1]] = crop
                crop = crop_fixed
            # Bezstratový resize s antialiasingom
            resized = resize(
                crop,
                (resize_size, resize_size) + crop.shape[2:],
                order=3,  # bicubická interpolácia
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            ).astype(img.dtype)
            out_path = os.path.join(output_dir, fname)
            io.imsave(out_path, resized)
            print(f"Spracované: {fname}")
        except Exception as e:
            print(f"Chyba pri spracovaní {fname}: {e}")

# Príklad použitia:
crop_and_resize_dataset("BR00109990/ch1", 1080, 512, "ch1_cropped_resized")