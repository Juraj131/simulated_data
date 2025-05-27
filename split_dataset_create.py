import os
import shutil
import random
import re

def create_image_dataset_splits(
    wrapped_images_input_dir,
    unwrapped_images_input_dir,
    output_base_directory_name="split_dataset_tiff",
    train_set_ratio=0.7,
    validation_set_ratio=0.15,
    random_operation_seed=42
):
    """
    Spracuje páry zabalených a rozbalených obrázkov, premenuje ich,
    priradí im ID a rozdelí ich do tréningových, validačných a testovacích sád.

    Args:
        wrapped_images_input_dir (str): Cesta k priečinku so zabalenými obrázkami.
        unwrapped_images_input_dir (str): Cesta k priečinku s rozbalenými obrázkami.
        output_base_directory_name (str): Názov hlavného výstupného priečinka.
        train_set_ratio (float): Pomer obrázkov pre tréningovú sadu.
        validation_set_ratio (float): Pomer obrázkov pre validačnú sadu.
        random_operation_seed (int): Seed pre náhodné operácie pre reprodukovateľnosť.
    """
    random.seed(random_operation_seed)

    # Kontrola vstupných priečinkov
    if not os.path.isdir(wrapped_images_input_dir):
        print(f"Chyba: Vstupný priečinok pre zabalené obrázky neexistuje: {wrapped_images_input_dir}")
        return
    if not os.path.isdir(unwrapped_images_input_dir):
        print(f"Chyba: Vstupný priečinok pre rozbalené obrázky neexistuje: {unwrapped_images_input_dir}")
        return

    # 1. Nájdenie spoločných bázových názvov obrázkov (predpokladáme .tiff príponu)
    try:
        wrapped_files_basenames = {os.path.splitext(f)[0] for f in os.listdir(wrapped_images_input_dir) if f.lower().endswith('.tiff')}
        unwrapped_files_basenames = {os.path.splitext(f)[0] for f in os.listdir(unwrapped_images_input_dir) if f.lower().endswith('.tiff')}
    except FileNotFoundError as e:
        print(f"Chyba pri čítaní obsahu vstupných priečinkov: {e}")
        return

    common_image_basenames = list(wrapped_files_basenames.intersection(unwrapped_files_basenames))

    if not common_image_basenames:
        print("Nenašli sa žiadne spoločné páry obrázkov v zadaných priečinkoch.")
        return

    print(f"Nájdených {len(common_image_basenames)} spoločných párov obrázkov.")

    # Náhodné zamiešanie pre rozdelenie do sád
    random.shuffle(common_image_basenames)

    # 2. Definovanie a vytvorenie výstupnej adresárovej štruktúry
    dataset_split_names = {
        "train": "train_dataset",
        "validation": "valid_dataset",
        "test": "test_dataset"
    }
    image_data_subfolders = ["images", "labels"]

    if os.path.exists(output_base_directory_name):
        print(f"Upozornenie: Výstupný priečinok '{output_base_directory_name}' už existuje. Jeho obsah môže byť prepísaný.")
        # Prípadne pridať možnosť vymazania: shutil.rmtree(output_base_directory_name)
    os.makedirs(output_base_directory_name, exist_ok=True)

    for split_dir_name in dataset_split_names.values():
        for subfolder_name in image_data_subfolders:
            path_to_create = os.path.join(output_base_directory_name, split_dir_name, subfolder_name)
            os.makedirs(path_to_create, exist_ok=True)

    # 3. Výpočet veľkostí jednotlivých sád
    total_pairs_count = len(common_image_basenames)
    train_count = int(train_set_ratio * total_pairs_count)
    validation_count = int(validation_set_ratio * total_pairs_count)
    test_count = total_pairs_count - train_count - validation_count

    print(f"\nRozdelenie párov:")
    print(f"  Celkový počet párov: {total_pairs_count}")
    print(f"  Tréningová sada: {train_count} párov")
    print(f"  Validačná sada: {validation_count} párov")
    print(f"  Testovacia sada: {test_count} párov")

    # 4. Priradenie párov do sád, premenovanie a kopírovanie
    current_file_id = 0
    processed_pairs_count = {"train": 0, "validation": 0, "test": 0}

    for i, basename in enumerate(common_image_basenames):
        image_id_str = f"{current_file_id:04d}"
        current_file_id += 1

        original_wrapped_image_path = os.path.join(wrapped_images_input_dir, basename + ".tiff")
        original_unwrapped_image_path = os.path.join(unwrapped_images_input_dir, basename + ".tiff")

        # Určenie, do ktorej sady pár patrí
        if i < train_count:
            current_split_key = "train"
        elif i < train_count + validation_count:
            current_split_key = "validation"
        else:
            current_split_key = "test"
        
        target_split_folder_name = dataset_split_names[current_split_key]

        # Definovanie nových názvov a cieľových ciest
        new_wrapped_image_name = f"wrappedbg_{image_id_str}.tiff"
        new_unwrapped_image_name = f"unwrapped_{image_id_str}.tiff"

        destination_wrapped_image_path = os.path.join(output_base_directory_name, target_split_folder_name, "images", new_wrapped_image_name)
        destination_unwrapped_image_path = os.path.join(output_base_directory_name, target_split_folder_name, "labels", new_unwrapped_image_name)

        # Kopírovanie a premenovanie
        try:
            if not os.path.exists(original_wrapped_image_path):
                print(f"Upozornenie: Zdrojový zabalený súbor neexistuje: {original_wrapped_image_path}")
                continue
            if not os.path.exists(original_unwrapped_image_path):
                print(f"Upozornenie: Zdrojový rozbalený súbor neexistuje: {original_unwrapped_image_path}")
                continue

            shutil.copy2(original_wrapped_image_path, destination_wrapped_image_path)
            shutil.copy2(original_unwrapped_image_path, destination_unwrapped_image_path)
            processed_pairs_count[current_split_key] += 1
        except Exception as e:
            print(f"Chyba pri spracovaní páru pre '{basename}' (ID: {image_id_str}): {e}")

    print("\nSpracovanie a rozdelenie datasetu dokončené.")
    print("Počet spracovaných párov v jednotlivých sadách:")
    for split_key, count_val in processed_pairs_count.items():
        print(f"  {dataset_split_names[split_key]}: {count_val} párov")

    # 5. Kontrolný výpis (Debug)
    print("\n--- Kontrolný výpis (Debug) ---")
    if processed_pairs_count["train"] > 0:
        train_images_dir_path = os.path.join(output_base_directory_name, dataset_split_names["train"], "images")
        train_labels_dir_path = os.path.join(output_base_directory_name, dataset_split_names["train"], "labels")
        try:
            example_train_image_files = os.listdir(train_images_dir_path)
            if example_train_image_files:
                # Zoberieme prvý nájdený súbor ako príklad
                example_wrapped_file_name = example_train_image_files[0]
                id_match = re.search(r"wrappedbg_(\d{4})\.tiff", example_wrapped_file_name)
                if id_match:
                    example_id = id_match.group(1)
                    expected_unwrapped_file_name = f"unwrapped_{example_id}.tiff"
                    
                    example_wrapped_full_path = os.path.join(train_images_dir_path, example_wrapped_file_name)
                    example_unwrapped_full_path = os.path.join(train_labels_dir_path, expected_unwrapped_file_name)

                    print(f"  Príklad kontroly páru z tréningovej sady (ID: {example_id}):")
                    print(f"    Obrázok (images): {example_wrapped_full_path} (Existuje: {os.path.exists(example_wrapped_full_path)})")
                    print(f"    Popisok (labels): {example_unwrapped_full_path} (Existuje: {os.path.exists(example_unwrapped_full_path)})")
                    if os.path.exists(example_wrapped_full_path) and os.path.exists(example_unwrapped_full_path):
                        print(f"    Párovanie pre ID {example_id} vyzerá byť v poriadku.")
                    else:
                        print(f"    CHYBA: Párovanie pre ID {example_id} zlyhalo alebo súbor chýba.")
                else:
                    print(f"  Nepodarilo sa extrahovať ID z názvu súboru: {example_wrapped_file_name}")
            else:
                print("  V tréningovom priečinku 'images' sa nenašli žiadne súbory pre kontrolu.")
        except Exception as e:
            print(f"  Chyba počas kontrolného výpisu: {e}")
    else:
        print("  Neboli spracované žiadne dáta do tréningovej sady pre kontrolu.")

    # Kontrola celkového počtu súborov
    total_copied_to_images = 0
    total_copied_to_labels = 0
    for split_dir_name_val in dataset_split_names.values():
        try:
            total_copied_to_images += len(os.listdir(os.path.join(output_base_directory_name, split_dir_name_val, "images")))
            total_copied_to_labels += len(os.listdir(os.path.join(output_base_directory_name, split_dir_name_val, "labels")))
        except FileNotFoundError:
            pass # Priečinok nemusí existovať, ak doň neboli skopírované žiadne súbory
    
    print(f"\nCelkový počet súborov skopírovaných do 'images' priečinkov: {total_copied_to_images}")
    print(f"Celkový počet súborov skopírovaných do 'labels' priečinkov: {total_copied_to_labels}")
    
    expected_total_processed_pairs = sum(processed_pairs_count.values())
    if total_copied_to_images == expected_total_processed_pairs and total_copied_to_labels == expected_total_processed_pairs:
        print("Celkový počet skopírovaných súborov zodpovedá počtu úspešne spracovaných párov.")
    else:
        print("Upozornenie: Celkový počet skopírovaných súborov nezodpovedá očakávanému počtu. Skontrolujte chybové hlášky.")
    print("--- Koniec kontrolného výpisu ---")

if __name__ == "__main__":
    # --- NASTAVTE CESTY K VAŠIM VSTUPNÝM PRIEČINKOM ---
    path_to_wrapped_dataset = "new_dataset_float32/wrappedbg"  # Napr. "data/wrapped_phase_images"
    path_to_unwrapped_dataset = "new_dataset_float32/unwrapped" # Napr. "data/unwrapped_phase_images"

    # --- NÁZOV VÝSTUPNÉHO PRIEČINKA ---
    output_directory = "split_dataset_tiff" # Môžete zmeniť podľa potreby

    print(f"Spúšťam skript na vytvorenie a rozdelenie datasetu...")
    print(f"Zdroj zabalených obrázkov: {path_to_wrapped_dataset}")
    print(f"Zdroj rozbalených obrázkov: {path_to_unwrapped_dataset}")
    print(f"Cieľový výstupný priečinok: {output_directory}")

    # Vytvorenie fiktívnych dát pre testovanie, ak neexistujú reálne cesty
    # Odkomentujte a upravte, ak chcete rýchlo otestovať funkčnosť skriptu
    # """
    if not (os.path.exists(path_to_wrapped_dataset) and os.path.exists(path_to_unwrapped_dataset)):
        print("\nUPOZORNENIE: Zadané vstupné cesty neexistujú. Vytváram fiktívne dáta pre testovanie...")
        path_to_wrapped_dataset = "dummy_wrapped_data"
        path_to_unwrapped_dataset = "dummy_unwrapped_data"
        os.makedirs(path_to_wrapped_dataset, exist_ok=True)
        os.makedirs(path_to_unwrapped_dataset, exist_ok=True)
        for k in range(20): # Vytvorí 20 párov
            with open(os.path.join(path_to_wrapped_dataset, f"image_pair_{k:03d}.tiff"), "w") as f_wr:
                f_wr.write(f"wrapped_content_{k}")
            with open(os.path.join(path_to_unwrapped_dataset, f"image_pair_{k:03d}.tiff"), "w") as f_un:
                f_un.write(f"unwrapped_content_{k}")
        # Pridanie niekoľkých nesúvisiacich súborov
        with open(os.path.join(path_to_wrapped_dataset, f"extra_image_A.tiff"), "w") as f_wr: f_wr.write("extra_A")
        with open(os.path.join(path_to_unwrapped_dataset, f"extra_image_B.tiff"), "w") as f_un: f_un.write("extra_B")
        print(f"Fiktívne dáta vytvorené v '{path_to_wrapped_dataset}' a '{path_to_unwrapped_dataset}'.")
    # """
    
    create_image_dataset_splits(
        wrapped_images_input_dir=path_to_wrapped_dataset,
        unwrapped_images_input_dir=path_to_unwrapped_dataset,
        output_base_directory_name=output_directory
    )

    # Prípadné vyčistenie fiktívnych dát po teste
    # if path_to_wrapped_dataset == "dummy_wrapped_data" and os.path.exists("dummy_wrapped_data"):
    #     shutil.rmtree("dummy_wrapped_data")
    # if path_to_unwrapped_dataset == "dummy_unwrapped_data" and os.path.exists("dummy_unwrapped_data"):
    #     shutil.rmtree("dummy_unwrapped_data")
    # Ak chcete vyčistiť aj výstupný priečinok po teste:
    # if os.path.exists(output_directory) and output_directory == "split_dataset_tiff": # Bezpečnostná kontrola
    #     shutil.rmtree(output_directory)
    #     print(f"Fiktívny výstupný priečinok '{output_directory}' bol vymazaný.")

    print("\nSkript dokončil svoju prácu.")
