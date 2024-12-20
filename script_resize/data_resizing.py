
# C:\Users\godin\Downloads\animals\Training Data\Training Data\
from PIL import Image
import os
import re

# Set the path to the main directory containing all the folders
input_directories = ["L:/big_file_storage/4TCA_S1_TIP/animals_15_classes/original_archive/Training Data/Training Data",
                     "L:/big_file_storage/4TCA_S1_TIP/animals_15_classes/original_archive/Validation Data/Validation Data",
                     "L:/big_file_storage/4TCA_S1_TIP/animals_15_classes/original_archive/Train Augmented/Train Augmented",
                     "L:/big_file_storage/4TCA_S1_TIP/animals_15_classes/original_archive/Testing Data/Testing Data",
                     "L:/big_file_storage/4TCA_S1_TIP/animals_15_classes/original_archive/Interesting Data/Interesting Data"]

output_directory = ["L:/big_file_storage/4TCA_S1_TIP/animals_15_classes/resized_archive/Training Data/Training Data",
                     "L:/big_file_storage/4TCA_S1_TIP/animals_15_classes/resized_archive/Validation Data/Validation Data",
                     "L:/big_file_storage/4TCA_S1_TIP/animals_15_classes/resized_archive/Train Augmented/Train Augmented",
                     "L:/big_file_storage/4TCA_S1_TIP/animals_15_classes/resized_archive/Testing Data/Testing Data",
                     "L:/big_file_storage/4TCA_S1_TIP/animals_15_classes/resized_archive/Interesting Data/Interesting Data"]

for dir in range(len(input_directories)):

    # Loop through each folder in the current directory
    for folder_name in os.listdir(input_directories[dir]):
        folder_path = os.path.join(input_directories[dir], folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):

            print("Working on folder {}".format(folder_path))

            extracted_folder_name = os.path.basename(folder_path)
            output_folder_path = os.path.join(output_directory[dir], extracted_folder_name)
            print(f"Foolder is {output_folder_path}")
            os.makedirs(output_folder_path, exist_ok=True)

            custom_id = 1

            # Loop through each file in the folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                # Check if the file is an image (JPEG format)
                if file_name.lower().endswith(('.jpg', '.jpeg')):
                    # Supprimer le numéro final dans le nom de fichier avec une expression régulière et l'extansion avec os.path
                    cleaned_name = re.sub(r"\s*\(\d+\)$", f"{custom_id}", os.path.splitext(file_name)[0])

                    # Open the image
                    img = Image.open(file_path)

                    # Resize the image to 64x64
                    img_resized = img.resize((64, 64))

                    # Define the new file path with PNG extension
                    new_file_path = os.path.join(output_folder_path, f"{cleaned_name}.png")

                    # Save the resized image in PNG format
                    img_resized.save(new_file_path, format="PNG")

                elif file_name.lower().endswith('.png'):
                    # Supprimer le numéro final dans le nom de fichier avec une expression régulière
                    cleaned_name = re.sub(r"\s*\(\d+\)$", f"{custom_id}", file_name)

                    # Open the image
                    img = Image.open(file_path)

                    # Resize the image to 64x64
                    img_resized = img.resize((64, 64))

                    # Define the new file path with PNG extension
                    new_file_path = os.path.join(output_folder_path, cleaned_name)

                    # Save the resized image in PNG format
                    img_resized.save(new_file_path, format="PNG")
                custom_id += 1
            print(f"Les {custom_id-1} images on été traitée du dossier {folder_path}")


# # Open the image
# img = Image.open("C:/Users/godin/Downloads/animals/Training Data/Training Data/Tiger/Tiger (994).jpeg") #"test/Zebra (98).jpeg"
#
# # Set the desired width and height
# new_width = 64  # Desired width in pixels
# new_height = 64  # Desired height in pixels
#
# # Resize the image
# img_resized = img.resize((new_width, new_height))
#
# # Save the resized image
# img_resized.save("test/tiger994.png", format="PNG")