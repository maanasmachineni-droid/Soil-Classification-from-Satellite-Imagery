import os

DATASET_PATH = "dataset_png"

for category in os.listdir(DATASET_PATH):

    category_path = os.path.join(
        DATASET_PATH,
        category
    )

    if os.path.isdir(category_path):

        print(
            category,
            "→",
            len(os.listdir(category_path)),
            "images"
        )