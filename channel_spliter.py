import os
import shutil

# Cesta k adresáru s obrázkami
src_dir = 'BR00109990'

for fname in os.listdir(src_dir):
    # nájdi pozíciu, kde sa v názve objaví "ch"
    idx = fname.find('ch')
    if idx != -1 and idx + 2 < len(fname):
        # znak hneď za "ch" by malo byť číslo 1–5
        ch = fname[idx + 2]
        if ch in '12345':
            # vytvor priečinok ch1, ch2, … ch5
            folder = f'ch{ch}'
            dst_folder = os.path.join(src_dir, folder)
            os.makedirs(dst_folder, exist_ok=True)
            # presuň súbor
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_folder, fname)
            shutil.move(src_path, dst_path)
            print(f'Moved {fname} → {folder}/')
