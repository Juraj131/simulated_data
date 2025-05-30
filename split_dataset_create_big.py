import os
import shutil
import random
import re
from collections import defaultdict

def create_multi_channel_dataset_splits(
    channel_input_sources,
    output_base_directory_name="split_dataset_tiff_multi",
    train_set_ratio=0.7,
    validation_set_ratio=0.15,
    random_operation_seed=42
):
    """
    Spracuje páry zabalených a rozbalených obrázkov z viacerých kanálov,
    premenuje ich s globálnym ID a rozdelí ich do tréningových, validačných
    a testovacích sád, pričom zachováva pomerné zastúpenie z každého kanála.
    Vstupné súbory sú iba kopírované.
    """
    random.seed(random_operation_seed)

    # 1. Definovanie a vytvorenie výstupnej adresárovej štruktúry
    dataset_split_names = {
        "train": "train_dataset",
        "validation": "valid_dataset",
        "test": "test_dataset"
    }
    image_data_subfolders = ["images", "labels"]

    if os.path.exists(output_base_directory_name):
        print(f"Upozornenie: Výstupný priečinok '{output_base_directory_name}' už existuje. Jeho obsah môže byť prepísaný alebo doplnený.")
    os.makedirs(output_base_directory_name, exist_ok=True)

    for split_dir_name in dataset_split_names.values():
        for subfolder_name in image_data_subfolders:
            path_to_create = os.path.join(output_base_directory_name, split_dir_name, subfolder_name)
            os.makedirs(path_to_create, exist_ok=True)

    all_file_ops_by_split = {
        "train": [],
        "validation": [],
        "test": []
    }

    current_file_id = 0
    
    print("Spracovávam kanály:")
    for channel_source in channel_input_sources:
        channel_name = channel_source['name']
        wrapped_images_input_dir = channel_source['wrapped_dir']
        unwrapped_images_input_dir = channel_source['unwrapped_dir']

        print(f"\n--- Kanál: {channel_name} ---")
        print(f"  Zabalené obrázky z: {wrapped_images_input_dir}")
        print(f"  Rozbalené obrázky z: {unwrapped_images_input_dir}")

        if not os.path.isdir(wrapped_images_input_dir):
            print(f"  Chyba: Vstupný priečinok pre zabalené obrázky kanála {channel_name} neexistuje: {wrapped_images_input_dir}")
            continue
        if not os.path.isdir(unwrapped_images_input_dir):
            print(f"  Chyba: Vstupný priečinok pre rozbalené obrázky kanála {channel_name} neexistuje: {unwrapped_images_input_dir}")
            continue

        try:
            wrapped_files_basenames = {os.path.splitext(f)[0] for f in os.listdir(wrapped_images_input_dir) if f.lower().endswith(('.tiff', '.tif'))}
            unwrapped_files_basenames = {os.path.splitext(f)[0] for f in os.listdir(unwrapped_images_input_dir) if f.lower().endswith(('.tiff', '.tif'))}
        except FileNotFoundError as e:
            print(f"  Chyba pri čítaní obsahu vstupných priečinkov pre kanál {channel_name}: {e}")
            continue

        common_image_basenames_channel = list(wrapped_files_basenames.intersection(unwrapped_files_basenames))

        if not common_image_basenames_channel:
            print(f"  Nenašli sa žiadne spoločné páry obrázkov pre kanál {channel_name}.")
            continue

        print(f"  Nájdených {len(common_image_basenames_channel)} spoločných párov obrázkov pre kanál {channel_name}.")
        random.shuffle(common_image_basenames_channel)

        total_pairs_channel = len(common_image_basenames_channel)
        train_count_channel = int(train_set_ratio * total_pairs_channel)
        validation_count_channel = int(validation_set_ratio * total_pairs_channel)
        test_count_channel = total_pairs_channel - train_count_channel - validation_count_channel


        print(f"  Rozdelenie pre kanál {channel_name}: Train={train_count_channel}, Valid={validation_count_channel}, Test={test_count_channel}")
        
        channel_pairs_processed_count = 0
        for i, basename in enumerate(common_image_basenames_channel):
            image_id_str = f"{current_file_id:04d}"

            original_wrapped_image_path = os.path.join(wrapped_images_input_dir, basename + ".tiff")
            if not os.path.exists(original_wrapped_image_path):
                 original_wrapped_image_path = os.path.join(wrapped_images_input_dir, basename + ".tif")
            
            original_unwrapped_image_path = os.path.join(unwrapped_images_input_dir, basename + ".tiff")
            if not os.path.exists(original_unwrapped_image_path):
                 original_unwrapped_image_path = os.path.join(unwrapped_images_input_dir, basename + ".tif")

            if not (os.path.exists(original_wrapped_image_path) and os.path.exists(original_unwrapped_image_path)):
                print(f"  Upozornenie: Jeden alebo oba súbory pre bázový názov '{basename}' v kanáli {channel_name} neexistujú. Pár bude preskočený.")
                continue

            new_wrapped_image_name = f"wrappedbg_{image_id_str}.tiff"
            new_unwrapped_image_name = f"unwrapped_{image_id_str}.tiff"

            if i < train_count_channel:
                current_split_key = "train"
            elif i < train_count_channel + validation_count_channel:
                current_split_key = "validation"
            else:
                current_split_key = "test"
            
            target_split_folder_name = dataset_split_names[current_split_key]
            destination_wrapped_image_path = os.path.join(output_base_directory_name, target_split_folder_name, "images", new_wrapped_image_name)
            destination_unwrapped_image_path = os.path.join(output_base_directory_name, target_split_folder_name, "labels", new_unwrapped_image_name)
            
            file_op_details = (original_wrapped_image_path, original_unwrapped_image_path, destination_wrapped_image_path, destination_unwrapped_image_path, channel_name)
            all_file_ops_by_split[current_split_key].append(file_op_details)
            
            current_file_id += 1
            channel_pairs_processed_count +=1
        print(f"  Skutočne pridaných párov z kanála {channel_name}: {channel_pairs_processed_count}")


    print(f"\nCelkovo pripravených párov naprieč kanálmi pred globálnym zamiešaním:")
    for split_key, ops_list in all_file_ops_by_split.items():
        print(f"  Pre {dataset_split_names[split_key]}: {len(ops_list)} párov")
        random.shuffle(ops_list) # Globálne zamiešanie operácií v rámci každej sady

    # 2. Kopírovanie súborov
    print("\nKopírujem súbory do cieľových priečinkov (vstupné súbory sa iba kopírujú)...")
    processed_counts = defaultdict(int) # Celkový počet skopírovaných párov pre každú sadu
    channel_split_counts = defaultdict(lambda: defaultdict(int)) # Počty podľa kanála a sady

    for split_key, file_ops_list in all_file_ops_by_split.items():
        for orig_w, orig_u, dest_w, dest_u, ch_name in file_ops_list:
            try:
                # Overenie, či cieľové súbory už neexistujú (ak by ID neboli unikátne, čo by nemalo nastať)
                if os.path.exists(dest_w) or os.path.exists(dest_u):
                    print(f"  Upozornenie: Cieľový súbor už existuje, preskakujem kopírovanie pre ID v {dest_w} / {dest_u}. Toto by sa nemalo stať pri správnom generovaní ID.")
                    continue

                shutil.copy2(orig_w, dest_w)
                shutil.copy2(orig_u, dest_u)
                processed_counts[split_key] += 1
                channel_split_counts[ch_name][split_key] += 1
            except Exception as e:
                print(f"Chyba pri kopírovaní páru ({os.path.basename(orig_w)}, {os.path.basename(orig_u)}) z kanála {ch_name} do {split_key}: {e}")

    print("\n--- FINÁLNE OVERENIE A ŠTATISTIKY ---")
    print("\n1. Celkové počty skopírovaných dvojíc v sadách:")
    total_processed_pairs = 0
    for split_key, count_val in processed_counts.items():
        print(f"  {dataset_split_names[split_key]}: {count_val} dvojíc")
        total_processed_pairs += count_val
    print(f"  Celkovo spracovaných dvojíc: {total_processed_pairs}")

    print("\n2. Kontrola duplicity identifikátorov a kompatibility párov v sadách:")
    all_sets_consistent = True
    for split_key, split_folder_name in dataset_split_names.items():
        print(f"  Kontrolujem sadu: {split_folder_name}")
        current_set_consistent = True
        images_dir = os.path.join(output_base_directory_name, split_folder_name, "images")
        labels_dir = os.path.join(output_base_directory_name, split_folder_name, "labels")

        image_ids = set()
        label_ids = set()
        duplicate_image_ids_found = False
        duplicate_label_ids_found = False

        if not os.path.isdir(images_dir):
            print(f"    Chyba: Priečinok {images_dir} neexistuje.")
            all_sets_consistent = False
            current_set_consistent = False
        else:
            for f_name in os.listdir(images_dir):
                match = re.search(r"wrappedbg_(\d{4})\.tiff", f_name)
                if match:
                    img_id = match.group(1)
                    if img_id in image_ids:
                        duplicate_image_ids_found = True
                    image_ids.add(img_id)
                elif f_name.lower().endswith(('.tiff', '.tif')): # Iné TIFF súbory
                    print(f"    Upozornenie: Neočakávaný TIFF súbor v {images_dir}: {f_name}")

        if not os.path.isdir(labels_dir):
            print(f"    Chyba: Priečinok {labels_dir} neexistuje.")
            all_sets_consistent = False
            current_set_consistent = False
        else:
            for f_name in os.listdir(labels_dir):
                match = re.search(r"unwrapped_(\d{4})\.tiff", f_name)
                if match:
                    lbl_id = match.group(1)
                    if lbl_id in label_ids:
                        duplicate_label_ids_found = True
                    label_ids.add(lbl_id)
                elif f_name.lower().endswith(('.tiff', '.tif')): # Iné TIFF súbory
                    print(f"    Upozornenie: Neočakávaný TIFF súbor v {labels_dir}: {f_name}")
        
        if duplicate_image_ids_found:
            print(f"    CHYBA: Duplicitné identifikátory nájdené v {images_dir}!")
            all_sets_consistent = False
            current_set_consistent = False
        if duplicate_label_ids_found:
            print(f"    CHYBA: Duplicitné identifikátory nájdené v {labels_dir}!")
            all_sets_consistent = False
            current_set_consistent = False

        if not (duplicate_image_ids_found or duplicate_label_ids_found): # Pokračuj len ak neboli duplikáty ID
            if image_ids == label_ids:
                if len(image_ids) == processed_counts[split_key]:
                    print(f"    OK: Sada {split_folder_name} obsahuje {len(image_ids)} unikátnych a kompatibilných dvojíc (ID sa zhodujú, počet sedí).")
                else:
                    print(f"    UPOZORNENIE: Počet unikátnych ID ({len(image_ids)}) v sade {split_folder_name} nezodpovedá očakávanému počtu skopírovaných dvojíc ({processed_counts[split_key]}).")
                    all_sets_consistent = False
                    current_set_consistent = False
            else:
                print(f"    CHYBA: Nesúlad identifikátorov medzi images a labels v sade {split_folder_name}!")
                if image_ids - label_ids:
                    print(f"      ID iba v images: {sorted(list(image_ids - label_ids))}")
                if label_ids - image_ids:
                    print(f"      ID iba v labels: {sorted(list(label_ids - image_ids))}")
                all_sets_consistent = False
                current_set_consistent = False
        
        if not current_set_consistent:
             print(f"    Problém zistený v sade {split_folder_name}.")


    if all_sets_consistent:
        print("  Všetky sady vyzerajú byť konzistentné a bez duplicitných ID v rámci párov.")
    else:
        print("  Boli zistené problémy s konzistenciou sád alebo duplicitnými ID. Skontrolujte výpisy vyššie.")


    print("\n3. Počty dvojíc z jednotlivých kanálov v sadách:")
    # Najprv zistíme všetky názvy kanálov, ktoré prispeli
    all_contributing_channels = sorted(list(channel_split_counts.keys()))
    
    header = "Kanál      |"
    for split_key in dataset_split_names.keys():
        header += f" {split_key.capitalize():<10} |" # Zarovnanie názvov sád
    print(header)
    print("-" * len(header))

    for ch_name in all_contributing_channels:
        row = f"{ch_name:<10} |" # Zarovnanie názvu kanála
        for split_key in dataset_split_names.keys():
            row += f" {channel_split_counts[ch_name].get(split_key, 0):<10} |"
        print(row)
    
    print("\n4. Percentuálne zloženie sád z celku:")
    if total_processed_pairs > 0:
        for split_key, count_val in processed_counts.items():
            percentage = (count_val / total_processed_pairs) * 100
            print(f"  {dataset_split_names[split_key]}: {count_val} dvojíc ({percentage:.2f}%)")
    else:
        print("  Neboli spracované žiadne dvojice, percentuálne zloženie nie je možné vypočítať.")

    print("\nSpracovanie a rozdelenie datasetu dokončené.")


if __name__ == "__main__":
    base_data_path = "." 

    channel_definitions = [
        {
            "name": "ch1",
            "wrapped_dir": os.path.join(base_data_path, "simulated_dataset_ch1", "wrappedbg"),
            "unwrapped_dir": os.path.join(base_data_path, "simulated_dataset_ch1", "unwrapped")
        },
        {
            "name": "ch2",
            "wrapped_dir": os.path.join(base_data_path, "simulated_dataset_ch2", "wrappedbg"),
            "unwrapped_dir": os.path.join(base_data_path, "simulated_dataset_ch2", "unwrapped")
        },
        {
            "name": "ch3",
            "wrapped_dir": os.path.join(base_data_path, "simulated_dataset_ch3", "wrappedbg"),
            "unwrapped_dir": os.path.join(base_data_path, "simulated_dataset_ch3", "unwrapped")
        },
        {
            "name": "ch4",
            "wrapped_dir": os.path.join(base_data_path, "simulated_dataset_ch4", "wrappedbg"),
            "unwrapped_dir": os.path.join(base_data_path, "simulated_dataset_ch4", "unwrapped")
        }
    ]
    
    output_main_directory = "split_dataset_multi_channel_tiff_verified"
    train_ratio = 0.7
    validation_ratio = 0.15
    seed = 42

    print(f"Spúšťam skript na vytvorenie a rozdelenie datasetu z viacerých kanálov...")
    print(f"Cieľový výstupný priečinok: {output_main_directory}")

    create_dummy_data = False 
    if create_dummy_data:
        print("\nUPOZORNENIE: Vytváram fiktívne dáta pre testovanie...")
        dummy_channel_defs = []
        for i in range(1, 5): 
            ch_name = f"ch{i}"
            base_dummy_path = f"dummy_simulated_dataset_{ch_name}"
            dummy_wrapped = os.path.join(base_dummy_path, "wrappedbg")
            dummy_unwrapped = os.path.join(base_dummy_path, "unwrapped")
            os.makedirs(dummy_wrapped, exist_ok=True)
            os.makedirs(dummy_unwrapped, exist_ok=True)
            
            num_files_per_channel = 20 + i*2 # Menší počet pre rýchlejší test
            for k in range(num_files_per_channel):
                # Použijeme .tiff pre konzistenciu
                # Vytvoríme prázdne súbory, obsah nie je pre test štruktúry dôležitý
                open(os.path.join(dummy_wrapped, f"image_pair_{ch_name}_{k:03d}.tiff"), "a").close()
                open(os.path.join(dummy_unwrapped, f"image_pair_{ch_name}_{k:03d}.tiff"), "a").close()
            dummy_channel_defs.append({"name": ch_name, "wrapped_dir": dummy_wrapped, "unwrapped_dir": dummy_unwrapped})
        channel_definitions = dummy_channel_defs 
        print("Fiktívne dáta vytvorené.")


    create_multi_channel_dataset_splits(
        channel_input_sources=channel_definitions,
        output_base_directory_name=output_main_directory,
        train_set_ratio=train_ratio,
        validation_set_ratio=validation_ratio,
        random_operation_seed=seed
    )

    if create_dummy_data:
        print("\nVymazávam fiktívne vstupné dáta...")
        for ch_def in channel_definitions: 
            base_dummy_to_remove = os.path.dirname(ch_def["wrapped_dir"])
            if os.path.exists(base_dummy_to_remove):
                shutil.rmtree(base_dummy_to_remove)
        print("Fiktívne vstupné dáta vymazané.")
        # Ak chcete vyčistiť aj výstupný priečinok po teste s fiktívnymi dátami:
        # print(f"Vymazávam fiktívny výstupný priečinok '{output_main_directory}'...")
        # if os.path.exists(output_main_directory):
        #     shutil.rmtree(output_main_directory)
        #     print(f"Fiktívny výstupný priečinok '{output_main_directory}' bol vymazaný.")

    print("\nSkript dokončil svoju prácu.")
