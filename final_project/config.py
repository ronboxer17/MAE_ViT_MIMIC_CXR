import os.path
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent
DATA_PATH = os.path.join(ROOT_PATH, r"assets/data")
LOGS_PATH = os.path.join(ROOT_PATH, "assets/logs")
MODELS_PATH = os.path.join(ROOT_PATH, "assets/models")
METRICS_PATH = os.path.join(ROOT_PATH, "assets/metrics")
MIMIC_PATH = os.path.join(DATA_PATH, "mimic-cxr")

METADATA_PATH = os.path.join(MIMIC_PATH, "mimic-cxr/mimic-cxr-2.0.0-metadata.csv")
SPLIT_PATH = os.path.join(MIMIC_PATH, "mimic-cxr/mimic-cxr-2.0.0-split.csv")
CHEXPERT_PATH = os.path.join(MIMIC_PATH, "mimic-cxr/mimic-cxr-2.0.0-chexpert.csv")
NEGIBOX_PATH = os.path.join(MIMIC_PATH, "mimic-cxr/mimic-cxr-2.0.0-negbio.csv")


IMG_PATH = os.path.join(r"D:/MIMIC CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files")
MIMIC_FILES_PATH = r"D:/MIMIC CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/"
NEW_IMG_PATH = os.path.join(r"D:/ron/MIMIC-MAE/data")
MIMIC_SAMPLE_ROOT = os.path.join(DATA_PATH, 'mimic-cxr', r'mimic_sample')

IDS_TO_IMAGES_PATHS = os.path.join(MIMIC_PATH, "meta-data/ids_to_images_paths.json")
IDS_WITH_LABELS_AND_SPLITS = os.path.join(
    MIMIC_PATH, r"meta-data/dicoms_with_labels_and_splits.csv"
)

DATASET_TYPES = ["train", "val", "test"]
