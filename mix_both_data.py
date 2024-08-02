from dataclasses import dataclass
from pathlib import Path
import shutil
import random
from rich.progress import track

@dataclass
class Phases:
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"

PHASES = [Phases.TRAIN, Phases.VAL, Phases.TEST]

class Dirs:
    IMAGE_DIR: Path
    MASK_DIR: Path

@dataclass
class OpticDirs(Dirs):
    IMAGE_DIR: Path = Path("oc_od_segm/gray_data")
    MASK_DIR: Path = Path("oc_od_segm/mask")

@dataclass
class VesselDirs(Dirs):
    IMAGE_DIR: Path = Path("vessel_segm/data")
    MASK_DIR: Path = Path("vessel_segm/mask")

@dataclass
class MixedDirs(Dirs):
    IMAGE_DIR: Path = Path("mixed_segm/data")
    MASK_DIR: Path = Path("mixed_segm/mask")

def save_imgs(src_dir: Path, dst_dir: Path, prefix: str | None = None):
    data_paths = list(src_dir.iterdir())
    for src_data_path in data_paths:
        dst_data_name = src_data_path.name
        if prefix is not None:
            dst_data_name = prefix + "_" + dst_data_name
        dst_data_path = dst_dir / dst_data_name
        shutil.copy(src_data_path, dst_data_path)

def save_data(
    src_data_root: Path,
    src_img_dirname: str,
    src_mask_dirname: str,
    dst_data_root: Path,
    dst_img_dirname: str,
    dst_mask_dirname: str,
    phase: str,
    num_samples: int | None = None,
    prefix: str | None = None
):
    src_img_root = src_data_root / src_img_dirname / phase
    src_mask_root = src_data_root / src_mask_dirname / "all" / phase

    dst_img_root = dst_data_root / dst_img_dirname / phase
    dst_mask_root = dst_data_root / dst_mask_dirname / "all" / phase
    dst_img_root.mkdir(parents=True, exist_ok=True)
    dst_mask_root.mkdir(parents=True, exist_ok=True)
    
    src_img_paths = list(src_img_root.iterdir())
    if phase == "train":
        if num_samples is not None:
            random.shuffle(src_img_paths)
            src_img_paths = src_img_paths[:num_samples]

    for src_img_path in track(src_img_paths, description=f"phase: {phase}, data: {str(src_data_root)}"):
        src_data_name = src_img_path.name
        src_mask_path = src_mask_root / src_data_name

        dst_data_name = src_data_name if prefix is None else prefix + "_" + src_data_name
        dst_img_path = dst_img_root / dst_data_name
        dst_mask_path = dst_mask_root / dst_data_name
        
        shutil.copy(src=src_img_path, dst=dst_img_path)
        shutil.copy(src=src_mask_path, dst=dst_mask_path)
        
def main():
    for phase in PHASES:
        src_data_root = Path("oc_od_segm")
        src_img_dirname = "gray_data"
        src_mask_dirname = dst_mask_dirname = "mask"
        dst_data_root = Path("mixed_segm")
        dst_img_dirname = "data"
        num_samples = 500
        prefix = "optic"
        save_data(
            src_data_root=src_data_root,
            src_img_dirname=src_img_dirname,
            src_mask_dirname=src_mask_dirname,
            dst_data_root=dst_data_root,
            dst_img_dirname=dst_img_dirname,
            dst_mask_dirname=dst_mask_dirname,
            phase=phase,
            num_samples=num_samples,
            prefix=prefix,
        )

        src_data_root = Path("vessel_segm")
        src_img_dirname = dst_img_dirname = "data"
        src_mask_dirname = dst_mask_dirname = "mask"
        dst_data_root = Path("mixed_segm")
        prefix = "vessel"
        save_data(
            src_data_root=src_data_root,
            src_img_dirname=src_img_dirname,
            src_mask_dirname=src_mask_dirname,
            dst_data_root=dst_data_root,
            dst_img_dirname=dst_img_dirname,
            dst_mask_dirname=dst_mask_dirname,
            phase=phase,
            prefix=prefix,
        )
    # for phase in PHASES:
    #     for Dirs, prefix in [(OpticDirs, "optic"), (VesselDirs, "vessel")]:
    #         src_img_dir: Path = Dirs.IMAGE_DIR / phase
    #         dst_img_dir: Path = MixedDirs.IMAGE_DIR / phase
    #         dst_img_dir.mkdir(parents=True, exist_ok=True)
    #         print(f"saving image (phase: {phase}, mask: {prefix})")
    #         save_imgs(src_img_dir, dst_img_dir, prefix=prefix)
            
    #         src_mask_dir: Path = Dirs.MASK_DIR / "all" / phase
    #         dst_mask_dir: Path = MixedDirs.MASK_DIR / "all" / phase
    #         dst_mask_dir.mkdir(parents=True, exist_ok=True)
    #         print(f"saving mask (phase: {phase}, mask: {prefix})")
    #         save_imgs(src_mask_dir, dst_mask_dir, prefix=prefix)

if __name__ == "__main__":
    main()
