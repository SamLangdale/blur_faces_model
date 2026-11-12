# Blur Faces Model

This folder contains two programs that use a YOLO model you provide to detect faces:

- `blur_video.py` blurs faces in a video and saves a new MP4.
- `blocker.py` displays a live, click-through blur overlay over detected faces on the Windows desktop.
- `training.py` trains a model using your own dataset configuration and data.

Model weights and datasets are not included. Before running either inference program, provide a compatible model checkpoint and set its path in `config.json`. 

Face detection is not guaranteed to find every face. Review the output before sharing it if complete anonymization is important.

## Requirements

- Windows
- Python 3.10 or newer
- A working Python environment with the packages in `requirements.txt`

From this folder, create and activate a virtual environment, then install the dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```



## Configuration

Both inference programs read `config.json` from this folder. It is included with paths set to `weights/best.pt`. Put a compatible checkpoint at that location, or change both `model_path` values to point to a checkpoint you are authorized to use. The checkpoint must contain a face-detection model; the desktop overlay additionally filters class names to those containing `face`.

Relative model and tracker paths are resolved from this folder. Adjust the confidence, image size, annotation, and padding settings in `config.json` as needed.

## Blur a video

Run from this folder:

```powershell
python blur_video.py "C:\path\to\input.mp4"
```

By default, the output is written next to the input video as `input_blurred_raw.mp4` when `show_annotations` is `false`, or `input_blurred_annotated.mp4` when it is `true`. Choose a specific output path with the optional second argument:

```powershell
python blur_video.py "C:\path\to\input.mp4" "C:\path\to\output.mp4"
```

You can override the configured model or select an inference device:

```powershell
python blur_video.py "C:\path\to\input.mp4" --model "C:\path\to\model.pt" --device cpu
```

`--device` also accepts a GPU index such as `0`; if omitted, Ultralytics selects the device. The script processes frames, blurs detections, and combines the video with the original audio when present. FFmpeg is used for this step; `imageio-ffmpeg` is included as a fallback dependency.

## Blur faces on the desktop

Start the live overlay from this folder:

```powershell
python blocker.py
```

The overlay stays on top and is click-through. Press **Esc** to close it. This script currently requests inference on GPU device `0`; a compatible CUDA-capable setup is needed.

## Settings

The `blur_video` settings in `config.json` control the video model path, ByteTrack tracker configuration, confidence threshold, inference image size, annotation visibility, and blur padding. The `blocker` settings control the live overlay's model, confidence threshold, image size, transparent color, and update interval. Lower confidence thresholds may detect more faces but can also increase false detections.

## Training

Training requires a dataset that you have permission to use and a `data.yaml` file in this folder. For example, for a one-class face dataset:

```yaml
path: C:/path/to/your/dataset
train: images/train
val: images/val
names:
  0: face
```

Update the paths and class names to match your dataset. Keep your local `data.yaml` and dataset out of the published package. Run training from this folder:

```powershell
python training.py
```

The script starts from Ultralytics' `yolo11s.pt` checkpoint, which Ultralytics may download when first needed, and requires a CUDA-capable GPU by default (`device="cuda"` in `training.py`). Training outputs are written by Ultralytics and are excluded from version control.


