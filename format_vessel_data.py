from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from random import Random
from rich.progress import track
import numpy as np
from PIL import Image

@dataclass
class Phases:
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"

PHASES = [Phases.TRAIN, Phases.VAL, Phases.TEST]

@dataclass
class Columns:
    FUNDUS: str = "fundus"
    VESSEL: str = "bv_seg"

@dataclass
class Dirnames:
    FUNDUS: str = "full-fundus"
    VESSEL: str = "blood-vessel"

TRAIN_RATIO = 0.8
TEST_RATIO = 0.1

RAW_DATASET_ROOT = "fundus_dataset"
METADATA_CSV_FILENAME = "metadata-standardized.csv"
DATASET_ROOT = "vessel_segm"

csv_filepath = Path(RAW_DATASET_ROOT) / METADATA_CSV_FILENAME

def read_csv(csv_filepath: Path) -> pd.DataFrame:
    assert csv_filepath.suffix == ".csv"
    df = pd.read_csv(csv_filepath)
    return df

def filter_by_whether_vessel_segm_exists(df: pd.DataFrame) -> pd.DataFrame:
    def _bv_filter(target: str | None):
        return isinstance(target, str)
    
    _df = df.query(f"{Columns.VESSEL}.apply(@{_bv_filter.__name__})")
    _df.reset_index(drop=True, inplace=True)
    return _df

def _get_path_segments(target: str) -> tuple[str, str]:
    assert target.startswith("/")
    
    segments = target.split("/")[1:]
    assert len(segments) == 2
    
    datatype_dir, img_name = segments
    assert img_name.endswith(".png")
    
    return datatype_dir, img_name

def _get_orig_data_path(datatype_dir: str, img_name: str):
    raw_data_path = Path(RAW_DATASET_ROOT) / datatype_dir / datatype_dir / img_name
    return raw_data_path

def _get_new_data_path(data_folder: Path, phase: Path, img_name: str):
    new_data_path = data_folder / phase / img_name
    return new_data_path

def _save_fundus_image(target: str, data_folder: Path, phase: str):
    datatype_dir, img_name = _get_path_segments(target)
    assert datatype_dir == Dirnames.FUNDUS
    
    raw_data_path = _get_orig_data_path(datatype_dir, img_name)
    image = Image.open(raw_data_path).convert("L")
    new_data_path = _get_new_data_path(data_folder, phase, img_name)
    image.save(new_data_path)

def _create_vessel_mask(bv_raw_data_path: Path):
    mask = np.array(Image.open(bv_raw_data_path).convert("L"))
    mask[mask > 0] = 3
    
    return mask

def _save_vessel_mask(bv_target: str, mask_folder: Path, phase: str):
    bv_datatype_dir, img_name = _get_path_segments(bv_target)
    assert bv_datatype_dir == Dirnames.VESSEL
    
    bv_raw_data_path = _get_orig_data_path(bv_datatype_dir, img_name)
    mask = _create_vessel_mask(bv_raw_data_path)
    save_path = _get_new_data_path(mask_folder / "all", phase, img_name)
    Image.fromarray(np.uint8(mask)).save(save_path)

def prepare_dataset(random: Random):
    df = read_csv(csv_filepath)
    df = filter_by_whether_vessel_segm_exists(df)
    
    indices: list[int] = list(df.index)
    random.shuffle(indices)
    num_train_indices = int(TRAIN_RATIO * len(indices))
    num_test_indices = int(TEST_RATIO * len(indices))
    phase_indices_dict = {
        Phases.TRAIN: indices[0:num_train_indices],
        Phases.VAL: indices[num_train_indices:num_train_indices+num_test_indices],
        Phases.TEST: indices[num_train_indices+num_test_indices:]
    }
    
    data_folder = Path(DATASET_ROOT) / "data"
    mask_folder = Path(DATASET_ROOT) / "mask"
    for phase in PHASES:
        (data_folder / phase).mkdir(parents=True, exist_ok=True)
        (mask_folder / "all" / phase).mkdir(parents=True, exist_ok=True)

    for phase in PHASES:
        phase_indices = phase_indices_dict[phase]
        for index in track(phase_indices, description=f"saving {phase} data ..."):
            record = df.loc[index, :]
            fundus = record[Columns.FUNDUS]
            vessel = record[Columns.VESSEL]
            _save_fundus_image(fundus, data_folder, phase)
            _save_vessel_mask(vessel, mask_folder, phase)

if __name__ == "__main__":
    SEED = 42
    random = Random(SEED)
    prepare_dataset(random)
