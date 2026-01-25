import os  # file and folder operations
import shutil  # moves files and deletes folders
from kaggle.api.kaggle_api_extended import KaggleApi  # download from Kaggle
from requests.exceptions import HTTPError  # detect HTTP ergirors

# List of Kaggle datasets to download
DATASETS = [
    "hasyimabdillah/workoutfitness-video",
    "riccardoriccio/real-time-exercise-recognition-dataset",
    "dilanarvand/exercise-gif-dataset",
    "yinonhadad/exercise-skeletons",
    "philosopher0808/gym-workoutexercises-video",
]

TARGET_ROOT = "dataset"  # final merged dataset folder
VIDEO_EXTS = (".mp4", ".mov", ".MOV")  # video file extensions to consider
TEMP_ROOT = "kaggle_temp_downloads"  # delete after merging

# ----------------- Utility Functions -----------------


def normalize_name(name: str) -> str:
    """normalize exercise folder names: lowercase + underscores"""
    name = name.lower()
    name = name.replace("-", " ")
    return "_".join(name.split())


def process_folder(folder_path):
    """recursively walk folder_path, renames them, and move videos to TARGET_ROOT"""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(VIDEO_EXTS):  # only process video files
                exercise_name = normalize_name(os.path.basename(root))
                target_folder = os.path.join(TARGET_ROOT, exercise_name)
                os.makedirs(target_folder, exist_ok=True)

                src = os.path.join(root, file)
                base, _ = os.path.splitext(file)
                dst = os.path.join(target_folder, f"{exercise_name}_{base}.mp4")

                shutil.move(src, dst)


# ----------------- Main Workflow -----------------


def download_and_merge_safe():
    """Downloads and merges dataset into main Dataset folder"""
    os.makedirs(TARGET_ROOT, exist_ok=True)
    os.makedirs(TEMP_ROOT, exist_ok=True)  # ensures temp folder exists

    api = KaggleApi()
    api.authenticate()  # authenticate using kaggle.json

    for i, dataset_id in enumerate(DATASETS, start=1):
        temp_folder = os.path.join(TEMP_ROOT, f"dataset_{i}")
        os.makedirs(temp_folder, exist_ok=True)

        print(f"\n⬇ Attempting to download {dataset_id} into {temp_folder}")
        try:
            api.dataset_download_files(dataset_id, path=temp_folder, unzip=True)
        except (
            HTTPError
        ) as e:  # checks if there are permissions needed to access dataset on Kaggle
            if e.response.status_code == 403:
                print(
                    f"⚠ Skipping {dataset_id}: Forbidden (may require manual acceptance)"
                )
            else:
                print(f"⚠ Skipping {dataset_id}: HTTP Error {e.response.status_code}")
            continue
        except Exception as e:
            print(f"⚠ Skipping {dataset_id}: Unexpected error: {e}")
            continue

        print(f"✔ Downloaded {dataset_id}")
        print(f"🔄 Normalizing and moving videos from {dataset_id}")
        process_folder(temp_folder)
        print(f"✔ Finished processing {dataset_id}")

    # clean up and delete temporary folders
    shutil.rmtree(TEMP_ROOT)
    print(
        "\n✅ All accessible datasets downloaded, normalized, and merged into 'dataset/'"
    )


# ----------------- Run Script -----------------

if __name__ == "__main__":
    download_and_merge_safe()
