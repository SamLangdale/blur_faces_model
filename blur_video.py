# SPDX-License-Identifier: AGPL-3.0-only

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
from ultralytics import YOLO

CONFIG = json.loads(
	(Path(__file__).resolve().parent / "config.json").read_text(encoding="utf-8")
)
VIDEO_CONFIG = CONFIG["blur_video"]

#model path
MODEL_PATH = Path(__file__).resolve().parent / VIDEO_CONFIG["model_path"]
# path to config file for model
TRACKER_PATH = Path(__file__).resolve().parent / VIDEO_CONFIG["tracker_path"]
# confidence threshold for the model detecting a face. ajust according to model precision and recall
CONFIDENCE = VIDEO_CONFIG["confidence"]
# image size of input frames
IMAGE_SIZE = VIDEO_CONFIG["image_size"]
# false : raw blurred face. True: shows ID and tracking box
SHOW_ANNOTATIONS = VIDEO_CONFIG["show_annotations"]
# makes the blurring box bigger
PADDING = VIDEO_CONFIG["padding"]


#where the actual model looks at the frame and tracks the faces
def blur_faces(frame, model, device):
	predict_options = {
		"imgsz": IMAGE_SIZE,
		"conf": CONFIDENCE,
		"verbose": False,
	}
	if device:
		predict_options["device"] = device

	result = model.track(
		frame,
		persist=True,
		tracker=str(TRACKER_PATH),
		**predict_options,
	)[0]
	for box in result.boxes or ():
		class_name = str(model.names[int(box.cls[0])])

		# option if you only want to blur faces 
		#if "face" not in class_name.casefold():
		#	continue

		left, top, right, bottom = map(int, box.xyxy[0].tolist())
		confidence = float(box.conf[0])
		track_id = int(box.id[0]) if box.id is not None else None
		padding_x = round((right - left) * PADDING)
		padding_y = round((bottom - top) * PADDING)
		blur_left = max(0, left - padding_x)
		blur_top = max(0, top - padding_y)
		blur_right = min(frame.shape[1], right + padding_x)
		blur_bottom = min(frame.shape[0], bottom + padding_y)
		if blur_right <= blur_left or blur_bottom <= blur_top:
			continue

		# marks face and blurs
		face = frame[blur_top:blur_bottom, blur_left:blur_right]
		sigma = max(12, min(face.shape[0], face.shape[1]) / 5)
		frame[blur_top:blur_bottom, blur_left:blur_right] = cv2.GaussianBlur(
			face, (0, 0), sigmaX=sigma
		)
		if not SHOW_ANNOTATIONS:
			continue

		# adds additional annotations if option is choosen
		blue = (255, 0, 0)
		cv2.rectangle(frame, (left, top), (right, bottom), blue, 1)
		id_label = f"ID {track_id if track_id is not None else '--'}"
		confidence_label = f"{confidence:.2f}"
		font = cv2.FONT_HERSHEY_SIMPLEX
		(id_width, id_height), id_baseline = cv2.getTextSize(
			id_label, font, 0.6, 2
		)
		(confidence_width, confidence_height), confidence_baseline = cv2.getTextSize(
			confidence_label, font, 0.4, 1
		)
		label_height = max(id_height, confidence_height)
		baseline = max(id_baseline, confidence_baseline)
		label_width = id_width + 6 + confidence_width
		label_top = max(0, top - label_height - baseline - 6)
		cv2.rectangle(
			frame,
			(left, label_top),
			(left + label_width + 8, label_top + label_height + baseline + 6),
			blue,
			cv2.FILLED,
		)
		cv2.putText(
			frame,
			id_label,
			(left + 4, label_top + label_height + 2),
			font,
			0.6,
			(0, 0, 0),
			2,
			cv2.LINE_AA,
		)
		cv2.putText(
			frame,
			confidence_label,
			(left + 4 + id_width + 6, label_top + label_height + 2),
			font,
			0.4,
			(0, 0, 0),
			1,
			cv2.LINE_AA,
		)
	return frame

# where the frames are processed
def process_video(input_path, output_path, model_path, device):
	ffmpeg = shutil.which("ffmpeg")
	if ffmpeg is None:
		try:
			import imageio_ffmpeg
		except ImportError as error:
			raise RuntimeError(
				"FFmpeg is required to preserve the video's audio. Install it on PATH "
				"or run `python -m pip install imageio-ffmpeg` in your virtual environment."
			) from error
		ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
	if not model_path.is_file():
		raise FileNotFoundError(f"Trained face model not found: {model_path}")
	if not TRACKER_PATH.is_file():
		raise FileNotFoundError(f"ByteTrack tracker config not found: {TRACKER_PATH}")
	if input_path.resolve() == output_path.resolve():
		raise ValueError("Input and output paths must be different.")

	capture = cv2.VideoCapture(str(input_path))
	if not capture.isOpened():
		raise RuntimeError(f"Could not open input video: {input_path}")

	fps = capture.get(cv2.CAP_PROP_FPS)
	if fps <= 0:
		capture.release()
		raise RuntimeError(f"Could not determine the input video's frame rate: {input_path}")

	ok, first_frame = capture.read()
	if not ok:
		capture.release()
		raise RuntimeError(f"Input video contains no readable frames: {input_path}")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	temp_file = tempfile.NamedTemporaryFile(
		suffix=".mp4", prefix="face_blur_", dir=output_path.parent, delete=False
	)
	temp_video_path = Path(temp_file.name)
	temp_file.close()

	writer = None
	try:
		height, width = first_frame.shape[:2]
		writer = cv2.VideoWriter(
			str(temp_video_path),
			cv2.VideoWriter_fourcc(*"mp4v"),
			fps,
			(width, height),
		)
		if not writer.isOpened():
			raise RuntimeError("Could not create the temporary video file.")

		model = YOLO(str(model_path))
		frame_number = 0
		frame = first_frame
		while True:
			writer.write(blur_faces(frame, model, device))
			frame_number += 1
			if frame_number % 100 == 0:
				print(f"Processed {frame_number} frames...")

			ok, frame = capture.read()
			if not ok:
				break

		writer.release()
		writer = None
		capture.release()

		print(f"Processed {frame_number} frames. Combining video and audio...")
		command = [
			ffmpeg,
			"-y",
			"-i",
			str(temp_video_path),
			"-i",
			str(input_path),
			"-map",
			"0:v:0",
			"-map",
			"1:a?",
			"-c:v",
			"copy",
			"-c:a",
			"aac",
			"-b:a",
			"192k",
			"-movflags",
			"+faststart",
			str(output_path),
		]
		result = subprocess.run(command, capture_output=True, text=True)
		if result.returncode:
			details = result.stderr.strip() or "FFmpeg failed without an error message."
			raise RuntimeError(f"Could not create output video:\n{details}")
		print(f"Saved blurred video to: {output_path}")
	finally:
		capture.release()
		if writer is not None:
			writer.release()
		if temp_video_path.exists():
			temp_video_path.unlink()


def main():
	parser = argparse.ArgumentParser(
		description="Blur detected faces in a video"
	)
	parser.add_argument("input", type=Path, help="Path to the input video")
	parser.add_argument(
		"output",
		type=Path,
		nargs="?",
		help="Output MP4 path (defaults to <input>_blurred.mp4)",
	)
	parser.add_argument(
		"--model",
		type=Path,
		default=MODEL_PATH,
		help=f"Face-detection model (default: {MODEL_PATH})",
	)
	parser.add_argument(
		"--device",
		default=None,
		help="Inference device, such as 0 or cpu (default: automatic)",
	)
	args = parser.parse_args()

	output_path = args.output or args.input.with_name(
        f"{args.input.stem}_blurred" + ("_raw" if not SHOW_ANNOTATIONS else "_annotated") + ".mp4"
	)

	process_video(args.input, output_path, args.model, args.device)


if __name__ == "__main__":
	main()
