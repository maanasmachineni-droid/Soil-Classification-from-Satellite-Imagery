import os
import cv2
import numpy as np
import rasterio

# -----------------------------
# INPUT TIFF DATASET FOLDER
# -----------------------------
INPUT_FOLDER = "dataset"
# -----------------------------
# OUTPUT PNG DATASET FOLDER
# -----------------------------
OUTPUT_FOLDER = "dataset_png"

# -----------------------------
# IMAGE SIZE
# -----------------------------
IMG_SIZE = 256

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Process each category
for category in os.listdir(INPUT_FOLDER):

    category_path = os.path.join(
        INPUT_FOLDER,
        category
    )

    # Skip non-folders
    if not os.path.isdir(category_path):
        continue

    output_category = os.path.join(
        OUTPUT_FOLDER,
        category
    )

    os.makedirs(output_category, exist_ok=True)

    print(f"\nProcessing category: {category}")

    # Process TIFF files
    for file in os.listdir(category_path):

        if not file.lower().endswith(
            (".tif", ".tiff")
        ):
            continue

        try:

            file_path = os.path.join(
                category_path,
                file
            )

            print(f"Converting: {file}")

            # -----------------------------
            # OPEN TIFF FILE
            # -----------------------------
            with rasterio.open(file_path) as src:

                print("Band Count:", src.count)

                # ---------------------------------
                # SENTINEL-2 RGB BANDS
                # B2 = Blue
                # B3 = Green
                # B4 = Red
                # ---------------------------------

                blue = src.read(2)

                green = src.read(3)

                red = src.read(4)

                # Combine into RGB image
                image = np.dstack(
                    (red, green, blue)
                )

            # -----------------------------
            # REDUCE IMAGE SIZE
            # Prevent excessive RAM usage
            # -----------------------------

            # Crop huge satellite images
            image = image[:1024, :1024]

            # -----------------------------
            # NORMALIZE PIXELS
            # -----------------------------
            image = cv2.normalize(
                image,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            image = image.astype(np.uint8)

            # -----------------------------
            # RESIZE IMAGE
            # -----------------------------
            image = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE)
            )

            # -----------------------------
            # SAVE AS PNG
            # -----------------------------
            output_filename = file.replace(
                ".tif",
                ".png"
            ).replace(
                ".tiff",
                ".png"
            )

            save_path = os.path.join(
                output_category,
                output_filename
            )

            cv2.imwrite(save_path, image)

            print("Saved:", save_path)

        except Exception as e:

            print(f"Error processing {file}")

            print(e)

print("\nTIFF to PNG conversion completed!")