
# SPDX-License-Identifier: AGPL-3.0-only

from pathlib import Path

from ultralytics import YOLO, settings


settings.update({"mlflow": False})

DATA_PATH = Path(__file__).resolve().parent / "data.yaml"
MODEL = "yolo11s.pt"


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_PATH}. "
            "Create data.yaml using the format described in README.md."
        )

    model = YOLO(MODEL)

    model.train(
        data=str(DATA_PATH),
        epochs=50,
        batch=-1,
        patience=10,
        fraction=1,
        device="cuda",
        workers=4,
        imgsz=960,

        # --- Color / lighting (different skin tones, cameras, indoor/outdoor) ---
        hsv_h=0.05,      # slight hue shift
        hsv_s=0.7,        # saturation variation
        hsv_v=0.9,        # brightness variation (low light / harsh light)

        # --- Geometry ---
        degrees=0.0,     # small rotations (tilted heads); keep modest
        translate=0.1,    # shift position
        scale=0.9,        # big size variation (close-up vs. far-away faces)
        shear=0.0,        # slight shear
        perspective=0.0,  # mild perspective warp (off-angle cameras)
        fliplr=0.5,       # horizontal flip: faces are roughly symmetric
        flipud=0.0,       # NO vertical flips: upside-down faces are unrealistic

        # --- Composition (small faces, crowds, partial views) ---
        mosaic=0.5,       # combines 4 images; great for small/crowded faces
        close_mosaic=10,  # turn mosaic off for the final 10 epochs
        mixup=0.0,        # light blending for regularization
        erasing=0.1,      # random erasing simulates occlusion (masks, hands, hair)
    )

if __name__ == "__main__":
    main()