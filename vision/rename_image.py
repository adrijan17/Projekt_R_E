import os
import shutil

def rename_images(IMAGES_PATH, ORDERED_IMAGES_PATH):
    os.makedirs(ORDERED_IMAGES_PATH, exist_ok=True)

    # Učitavanje path-ova originalnih slika
    images = []
    for file in os.listdir(IMAGES_PATH):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            full_path = os.path.join(IMAGES_PATH, file)
            images.append(full_path)
    
    # Preimenovanje path-ova slika u obliku: 
    # (1. slika -> 0001.png), (123. slika -> 0123.png)
    for index, image_path in enumerate(images, start=1):
        print(image_path)
        new_name = f"{index:04d}.png"
        new_image_path = os.path.join(ORDERED_IMAGES_PATH, new_name)
        
        shutil.copy(image_path, new_image_path)


def main():
    IMAGES_PATH = "/home/tomo/Faks/ProjektE/Code_and_Data/wire_detection_algorithm/images"
    ORDERED_IMAGES_PATH = "/home/tomo/Faks/ProjektE/Code_and_Data/wire_detection_algorithm/GUI_labeling/images_ordered"
    rename_images(IMAGES_PATH=IMAGES_PATH, ORDERED_IMAGES_PATH=ORDERED_IMAGES_PATH)

if __name__ == "__main__":
    main()