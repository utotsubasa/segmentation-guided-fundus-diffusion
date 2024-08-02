import pandas as pd
from pathlib import Path
from math import isnan
from random import Random
import dataclasses
from dataclasses import dataclass
import shutil
import numpy as np
from PIL import Image, ImageOps
from rich.progress import track
from cv2 import createCLAHE

# phases
@dataclass
class Phases:
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"
    
PHASES = [Phases.TRAIN, Phases.VAL, Phases.TEST]

@dataclass
class Columns:
    FUNDUS: str = "fundus"
    OPTIC_CUP: str = "fundus_oc_seg"
    OPTIC_DISC: str = "fundus_od_seg"

@dataclass
class Dirnames:
    FUNDUS: str = "full-fundus"
    OPTIC_CUP: str = "optic-cup"
    OPTIC_DISC: str = "optic-disc"

# experiment parameters
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1

# dataset dirs
RAW_DATASET_ROOT = "fundus_dataset"
DATASET_ROOT = "oc_od_segm"
METADATA_CSV_FILENAME = "metadata-standardized.csv"

csv_filepath = Path(RAW_DATASET_ROOT) / METADATA_CSV_FILENAME

def read_csv(csv_filepath: Path) -> pd.DataFrame:
    assert csv_filepath.suffix == ".csv"
    df = pd.read_csv(csv_filepath)
    return df

def filter_by_optic_disc(df: pd.DataFrame):
    def _od_filter(target):
        if isinstance(target, str):
            return True
        assert isnan(target)
        return False

    _df = df.query(f"{Columns.OPTIC_DISC}.apply(@{_od_filter.__name__})")
    _df.reset_index(drop=True, inplace=True)
    return _df

def _assert_absolute_path_from_data_root(target: str):
    assert target.startswith("/")
    pathlike = Path(target)
    parent = pathlike.parent
    segments = target.split("/")[1:]
    for segment in segments:
        assert segments
    
def _create_original_filepath(abspath_from_dataroot: str):
    ...

def _get_segments(target: str) -> tuple[str, str]:
    assert target.startswith("/")
    
    segments = target.split("/")[1:]
    assert len(segments) == 2
    
    datatype_dir, img_name = segments
    assert img_name.endswith(".png")
    
    return datatype_dir, img_name

def _get_orig_data_path(datatype_dir: str, img_name: str):
    raw_data_path = Path(RAW_DATASET_ROOT) / datatype_dir / datatype_dir / img_name
    return raw_data_path

def _get_new_data_path(data_folder: Path, phase: str, img_name: str):
    new_data_path = data_folder / phase / img_name
    return new_data_path

def _save_fundus_image(target: str, data_folder: Path, phase: str):
    datatype_dir, img_name = _get_segments(target)
    assert datatype_dir == Dirnames.FUNDUS
    
    raw_data_path = _get_orig_data_path(datatype_dir, img_name)
    new_data_path = _get_new_data_path(data_folder, phase, img_name)
    shutil.copy(raw_data_path, new_data_path)

def _create_optic_mask(od_raw_data_path: Path, oc_raw_data_path: Path | None):
    combined_mask = np.array(Image.open(od_raw_data_path).convert("L"))
    combined_mask[combined_mask > 0] = 1
    
    if oc_raw_data_path is not None:
        mask = np.array(Image.open(oc_raw_data_path).convert("L"))
        mask[mask > 0] = 2
        
        combined_mask = np.where(mask > 0, mask, combined_mask)
    
    return combined_mask

def _save_optic_mask(oc_target: str | float, od_target: str, mask_folder: Path, phase: str, mask_type: str = "all"):
    od_datatype_dir, img_name = _get_segments(od_target)
    assert od_datatype_dir == Dirnames.OPTIC_DISC
    
    od_raw_data_path = _get_orig_data_path(od_datatype_dir, img_name)
    
    if not isinstance(oc_target, str):
        assert isnan(oc_target)
        oc_raw_data_path = None
    elif oc_target == "Not Visible":
        oc_raw_data_path = None
    else:
        oc_datatype_dir, _ = _get_segments(oc_target)
        assert oc_datatype_dir == Dirnames.OPTIC_CUP
        
        oc_raw_data_path = _get_orig_data_path(oc_datatype_dir, img_name)
    
    combined_mask = _create_optic_mask(od_raw_data_path, oc_raw_data_path)
    save_path = _get_new_data_path(mask_folder / "all", phase, img_name)
    Image.fromarray(np.uint8(combined_mask)).save(save_path)
    
def prepare_dataset(random: Random):
    """
    1. optic discを持つrecordを取得する（optic discがあってもoptic cupがない場合もある）
    2. train, val, testに分割
    3. folder directory作成
    4. fundus画像をコピー
    5. optic discとcupのmaskを取得し、1つのmask画像を作成する
    """
    # 1.
    df = read_csv(csv_filepath)
    df = filter_by_optic_disc(df)
    # 2.
    indices: list[int] = list(df.index)
    random.shuffle(indices)
    num_train_indices = int(TRAIN_RATIO * len(indices))
    num_test_indices = int(TEST_RATIO * len(indices))
    phase_indices_dict = {
        Phases.TRAIN: indices[0:num_train_indices],
        Phases.VAL: indices[num_train_indices:num_train_indices+num_test_indices],
        Phases.TEST: indices[num_train_indices+num_test_indices:]
    }
    # 3.
    data_folder = Path(DATASET_ROOT) / "data"
    mask_folder = Path(DATASET_ROOT) / "mask"
    for phase in PHASES:
        (data_folder / phase).mkdir(parents=True, exist_ok=True)
        (mask_folder / "all" / phase).mkdir(parents=True, exist_ok=True)
    # 4., 5.
    for phase in PHASES:
        phase_indices = phase_indices_dict[phase]
        for index in track(phase_indices, description=f"saving {phase} data ..."):
            record = df.loc[index, :]
            fundus = record[Columns.FUNDUS]
            fundus_optic_cup = record[Columns.OPTIC_CUP]
            fundus_optic_disc = record[Columns.OPTIC_DISC]
            _save_fundus_image(fundus, data_folder, phase)
            _save_optic_mask(fundus_optic_cup, fundus_optic_disc, mask_folder, phase)
    
def prepare_grayscale_dataset():
    color_data_folder = Path(DATASET_ROOT) / "data"
    gray_data_folder = Path(DATASET_ROOT) / "gray_data"

    gray_data_folder.mkdir(parents=True, exist_ok=True)
    for phase in PHASES:
        gray_data_folder.joinpath(phase).mkdir(parents=True, exist_ok=True)

    for phase in PHASES:
        color_phase_folder = color_data_folder / phase
        gray_phase_folder = gray_data_folder / phase
        gray_phase_folder.mkdir(parents=True, exist_ok=True)

        color_image_paths = color_phase_folder.iterdir()
        for color_image_path in color_image_paths:
            gray_image = Image.open(color_image_path).convert("L")
            save_path = gray_phase_folder / color_image_path.name
            gray_image.save(save_path)
            
def prepare_equalized_dataset():
    color_data_folder = Path(DATASET_ROOT) / "data"
    equalized_data_folder = Path(DATASET_ROOT) / "equalized_data"

    equalized_data_folder.mkdir(parents=True, exist_ok=True)
    for phase in PHASES:
        equalized_data_folder.joinpath(phase).mkdir(parents=True, exist_ok=True)

    for phase in PHASES:
        color_phase_folder = color_data_folder / phase
        equalized_phase_folder = equalized_data_folder / phase
        equalized_phase_folder.mkdir(parents=True, exist_ok=True)

        color_image_paths = color_phase_folder.iterdir()
        for color_image_path in color_image_paths:
            gray_image = Image.open(color_image_path).convert("L")
            equalized_image = ImageOps.equalize(gray_image)
            save_path = equalized_phase_folder / color_image_path.name
            equalized_image.save(save_path)

def prepare_clahe_dataset():
    color_data_folder = Path(DATASET_ROOT) / "data"
    clahe_data_folder = Path(DATASET_ROOT) / "clahe_data"

    clahe_data_folder.mkdir(parents=True, exist_ok=True)
    for phase in PHASES:
        clahe_data_folder.joinpath(phase).mkdir(parents=True, exist_ok=True)
    
    clahe = createCLAHE(clipLimit=3, tileGridSize=(32, 32))

    for phase in PHASES:
        color_phase_folder = color_data_folder / phase
        clahe_phase_folder = clahe_data_folder / phase
        clahe_phase_folder.mkdir(parents=True, exist_ok=True)

        color_image_paths = color_phase_folder.iterdir()
        for color_image_path in color_image_paths:
            gray_image = Image.open(color_image_path).convert("L")
            clahe_image = np.array(gray_image)
            clahe_image = clahe.apply(clahe_image)
            clahe_image = Image.fromarray(clahe_image)
            save_path = clahe_phase_folder / color_image_path.name
            clahe_image.save(save_path)

if __name__ == "__main__":
    SEED = 42
    random = Random(SEED)
    prepare_dataset(random)
    prepare_grayscale_dataset()
    # prepare_equalized_dataset()
    # prepare_clahe_dataset()
