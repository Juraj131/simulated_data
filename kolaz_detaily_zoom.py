import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import numpy as np
import os

def create_comparison_with_zoom(
    image_path1, image_path2, 
    roi_coords, 
    output_filename="porovnanie_s_detailom.png", 
    title1="Obrázok 1 (CCHM Zabalený)", 
    title2="Obrázok 2 (Simulovaný Zabalený)",
    detail_title1="Detail Obrázka 1",
    detail_title2="Detail Obrázka 2"
    ):
    """
    Vytvorí koláž dvoch obrázkov s vyznačenou oblasťou záujmu (ROI),
    zväčšenými detailmi tejto ROI a spojovacími čiarami.
    ROI súradnice (x,y) sa očakávajú ako ľavý HORNÝ roh.
    """
    try:
        img1_original = io.imread(image_path1)
        img2_original = io.imread(image_path2)
    except FileNotFoundError as e:
        print(f"CHYBA: Jeden zo súborov nebol nájdený: {e}")
        return
    except Exception as e:
        print(f"CHYBA pri načítaní obrázkov: {e}")
        return

    def process_image(img, path):
        processed_img = img
        if img.ndim == 3:
            if img.shape[2] == 4: 
                from skimage.color import rgba2rgb, rgb2gray
                print(f"Konvertujem RGBA obrázok ({path}) na odtiene sivej.")
                processed_img = rgb2gray(rgba2rgb(img))
            elif img.shape[2] == 3: 
                from skimage.color import rgb2gray
                print(f"Konvertujem RGB obrázok ({path}) na odtiene sivej.")
                processed_img = rgb2gray(img)
            elif img.shape[2] == 1:
                 processed_img = np.squeeze(img, axis=2)
        elif img.ndim != 2:
            print(f"UPOZORNENIE: Obrázok ({path}) má neočakávaný počet dimenzií: {img.ndim}. Skúsim použiť tak, ako je.")
        
        # Normalizácia na float [0,1] ak sú to integer typy
        if processed_img.dtype != np.float32 and processed_img.dtype != np.float64:
            from skimage.util import img_as_float
            processed_img = img_as_float(processed_img)
        return processed_img
    
    img1 = process_image(img1_original, image_path1)
    img2 = process_image(img2_original, image_path2)

    roi_x, roi_y, roi_w, roi_h = map(int, roi_coords)

    slice_y_start = max(0, roi_y)
    slice_y_end = min(img1.shape[0], roi_y + roi_h)
    slice_x_start = max(0, roi_x)
    slice_x_end = min(img1.shape[1], roi_x + roi_w)

    if slice_y_start >= slice_y_end or slice_x_start >= slice_x_end:
        print(f"CHYBA: ROI súradnice ({roi_coords}) vedú k neplatnému výrezu pre obrázok 1 (tvar: {img1.shape}). Fallback na nulový detail.")
        detail1 = np.zeros((max(1, roi_h), max(1, roi_w)), dtype=img1.dtype)
    else:
        detail1 = img1[slice_y_start:slice_y_end, slice_x_start:slice_x_end]
    print(f"DEBUG: detail1 po výreze - tvar: {detail1.shape}, veľkosť: {detail1.size}")


    slice_y_end_img2 = min(img2.shape[0], roi_y + roi_h)
    slice_x_end_img2 = min(img2.shape[1], roi_x + roi_w)
    if slice_y_start >= slice_y_end_img2 or slice_x_start >= slice_x_end_img2:
        print(f"CHYBA: ROI súradnice ({roi_coords}) vedú k neplatnému výrezu pre obrázok 2 (tvar: {img2.shape}). Fallback na nulový detail.")
        detail2 = np.zeros((max(1, roi_h), max(1, roi_w)), dtype=img2.dtype)
    else:
        detail2 = img2[slice_y_start:slice_y_end_img2, slice_x_start:slice_x_end_img2]
    print(f"DEBUG: detail2 po výreze - tvar: {detail2.shape}, veľkosť: {detail2.size}")

    # Definovanie spoločných rozmerov pre ROI štvorčeky.
    # Tieto rozmery (roi_w, roi_h) sú odvodené z roi_coords a použijú sa pre oba štvorčeky,
    # čím sa zabezpečí, že ľavý štvorček bude mať rovnakú veľkosť ako pravý.
    rect_roi_width = roi_w
    rect_roi_height = roi_h

    fig = plt.figure(figsize=(10, 10)) 
    # Manuálne nastavenie medzier medzi subplotmi
    # wspace pre horizontálnu blízkosť (medzi stĺpcami)
    # hspace pre vertikálnu medzeru (medzi radmi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.15, wspace=0.05) 

    ax_main1 = fig.add_subplot(gs[0, 0])
    ax_main2 = fig.add_subplot(gs[0, 1])
    ax_detail1 = fig.add_subplot(gs[1, 0])
    ax_detail2 = fig.add_subplot(gs[1, 1])

    title_fontsize = 16 # Zmenšená veľkosť písma pre nadpisy

    ax_main1.imshow(img1, cmap='gray', origin='upper')
    ax_main1.set_title(title1, fontsize=title_fontsize)
    ax_main1.axis('off')

    ax_main2.imshow(img2, cmap='gray', origin='upper')
    ax_main2.set_title(title2, fontsize=title_fontsize)
    ax_main2.axis('off')

    roi_line_thickness = 2.5
    # Použitie spoločných rozmerov rect_roi_width a rect_roi_height pre oba štvorčeky
    rect1 = patches.Rectangle((roi_x, roi_y), rect_roi_width, rect_roi_height, 
                              linewidth=roi_line_thickness, edgecolor='r', facecolor='none', linestyle='--')
    ax_main1.add_patch(rect1)
    rect2 = patches.Rectangle((roi_x, roi_y), rect_roi_width, rect_roi_height, 
                              linewidth=roi_line_thickness, edgecolor='r', facecolor='none', linestyle='--')
    ax_main2.add_patch(rect2)

    if detail1.size > 0:
        ax_detail1.imshow(detail1, cmap='gray', origin='upper', aspect='equal')
    # ax_detail1.set_title(detail_title1, fontsize=title_fontsize) # Odstránený nadpis pre detail 1
    ax_detail1.axis('off')

    if detail2.size > 0:
        ax_detail2.imshow(detail2, cmap='gray', origin='upper', aspect='equal')
    # ax_detail2.set_title(detail_title2, fontsize=title_fontsize) # Odstránený nadpis pre detail 2
    ax_detail2.axis('off')

    # Definovanie bodov pre spojovacie čiary
    # Východiskové body z ROI v hlavných obrázkoch (súradnice dát hlavného obrázka)
    roi_main_bottom_left = (roi_x, roi_y + roi_h)             # Dolný ľavý roh ROI
    roi_main_bottom_right = (roi_x + roi_w, roi_y + roi_h)    # Dolný pravý roh ROI

    # Cieľové body v detailných obrázkoch, upravené pre extenty imshow
    # Horný ľavý roh extentu detailného obrázka
    detail_img_extent_top_left = (-0.5, -0.5) 

    # Horný pravý roh extentu detailného obrázka 1
    # detail1.shape[1] je počet stĺpcov (šírka) detailu
    detail_img_extent_top_right1 = (detail1.shape[1] - 0.5, -0.5)
    print(f"DEBUG: Cieľové body pre detail 1: ľavý_hore={detail_img_extent_top_left}, pravý_hore={detail_img_extent_top_right1}")

    if detail1.size > 0 : # Kontrola, či detail1 nie je prázdny
        # Prvá čiara (ľavé rameno) pre obrázok 1
        con1 = patches.ConnectionPatch(xyA=roi_main_bottom_left, xyB=detail_img_extent_top_left,
                                      coordsA='data', coordsB='data',
                                      axesA=ax_main1, axesB=ax_detail1,
                                      color="red", linestyle="--", linewidth=roi_line_thickness)
        fig.add_artist(con1)
        # Druhá čiara (pravé rameno) pre obrázok 1
        con2 = patches.ConnectionPatch(xyA=roi_main_bottom_right, xyB=detail_img_extent_top_right1,
                                      coordsA='data', coordsB='data',
                                      axesA=ax_main1, axesB=ax_detail1, 
                                      color="red", linestyle="--", linewidth=roi_line_thickness)
        fig.add_artist(con2)

    # Horný pravý roh extentu detailného obrázka 2
    detail_img_extent_top_right2 = (detail2.shape[1] - 0.5, -0.5)
    print(f"DEBUG: Cieľové body pre detail 2: ľavý_hore={detail_img_extent_top_left}, pravý_hore={detail_img_extent_top_right2}")

    if detail2.size > 0: # Kontrola, či detail2 nie je prázdny
        # Prvá čiara (ľavé rameno) pre obrázok 2
        con3 = patches.ConnectionPatch(xyA=roi_main_bottom_left, xyB=detail_img_extent_top_left,
                                      coordsA='data', coordsB='data',
                                      axesA=ax_main2, axesB=ax_detail2,
                                      color="red", linestyle="--", linewidth=roi_line_thickness)
        fig.add_artist(con3)
        # Druhá čiara (pravé rameno) pre obrázok 2
        con4 = patches.ConnectionPatch(xyA=roi_main_bottom_right, xyB=detail_img_extent_top_right2,
                                      coordsA='data', coordsB='data',
                                      axesA=ax_main2, axesB=ax_detail2,
                                      color="red", linestyle="--", linewidth=roi_line_thickness)
        fig.add_artist(con4)
    
    # plt.tight_layout(pad=5.0, h_pad=5.0, w_pad=5.0) # Odstránené volanie tight_layout

    # Manuálne nastavenie okrajov celej figúry
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    # Uloženie obrázka PRED zobrazením a zatvorením
    plt.savefig(output_filename, bbox_inches='tight', dpi=300) # Používame output_filename z parametrov funkcie
    print(f"Koláž bola uložená ako: {output_filename}")

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    # --- NASTAVTE TIETO CESTY A PARAMETRE ---
    # cesta_obrazok_cchm = r"C:\Users\juraj\Desktop\bc_prenos\split_dataset_tiff\train_dataset\images\wrappedbg_0_20.tiff"
    # # cesta_obrazok_simulovany = r"C:\Users\juraj\Desktop\simulated_data\SIMULATED_DATASETS\split_dataset_tiff_3\train_dataset\images\wrappedbg_0000.tiff"
    # cesta_obrazok_simulovany = r"C:\Users\juraj\Desktop\simulated_data\SIMULATED_DATASETS\split_dataset_tiff_3\train_dataset\images\wrappedbg_0000.tiff" # Použitá vaša zakomentovaná cesta
    cesta_obrazok_cchm = r"C:\Users\juraj\Desktop\bc_prenos\split_dataset_tiff\train_dataset\images\wrappedbg_0_8.tiff"
    
    cesta_obrazok_simulovany = r"C:\Users\juraj\Desktop\simulated_data\SIMULATED_DATASETS\new_dataset_float32_3\wrappedbg\r02c02f09p01-ch1sk1fk1fl1_f32.tiff" # Použitá vaša zakomentovaná cesta

    roi = (150, 100, 80, 80) 
    output_kolaz_filename = "final_kolaz_s_detailom.svg" # Zmena koncovky na .svg

    if not os.path.exists(cesta_obrazok_cchm):
        print(f"CHYBA: Súbor pre Obrázok 1 neexistuje: {cesta_obrazok_cchm}")
    elif not os.path.exists(cesta_obrazok_simulovany):
        print(f"CHYBA: Súbor pre Obrázok 2 neexistuje: {cesta_obrazok_simulovany}")
    else:
        print("Načítavam obrázky a generujem vizualizáciu...")
        create_comparison_with_zoom(
            cesta_obrazok_cchm,
            cesta_obrazok_simulovany,
            roi_coords=roi,
            output_filename=output_kolaz_filename, # Použitie definovaného názvu súboru
            title1="CCHM dáta", 
            title2="Simulované dáta"
            # detail_title1 a detail_title2 už nie sú potrebné, keďže nadpisy sú odstránené
        )
        print("Vizualizácia by sa mala zobraziť a uložiť.")