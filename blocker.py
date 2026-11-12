# SPDX-License-Identifier: AGPL-3.0-only

import ctypes
import json
from ctypes import wintypes
import threading
from pathlib import Path
from typing import Generic, TypeVar
import dxcam

import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from ultralytics.engine.results import Results

CONFIG = json.loads(
	(Path(__file__).resolve().parent / "config.json").read_text(encoding="utf-8")
)
BLOCKER_CONFIG = CONFIG["blocker"]

MODEL_PATH = Path(__file__).resolve().parent / BLOCKER_CONFIG["model_path"]
#IMAGE_PATH = Path(__file__).resolve().parent / "censor.avif"
TRANSPARENT_COLOR = BLOCKER_CONFIG["transparent_color"]
CONFIDENCE = BLOCKER_CONFIG["confidence"]
IMAGE_SIZE = BLOCKER_CONFIG["image_size"]
UPDATE_INTERVAL_MS = BLOCKER_CONFIG["update_interval_ms"]

T = TypeVar("T")


class LatestMailbox(Generic[T]):
	def __init__(self):
		self._lock = threading.Lock()
		self._item: T | None = None

	def put(self, item: T) -> None:
		with self._lock:
			self._item = item

	def get(self) -> T | None:
		with self._lock:
			item = self._item
			self._item = None
			return item


def main():
	if not MODEL_PATH.is_file():
		raise FileNotFoundError(f"Trained face model not found: {MODEL_PATH}")

	# Must happen before Tk() and any GetSystemMetrics calls
	ctypes.windll.shcore.SetProcessDpiAwareness(2)

	model = YOLO(str(MODEL_PATH))
	user32 = ctypes.windll.user32

	user32.GetParent.argtypes = [wintypes.HWND]
	user32.GetParent.restype = wintypes.HWND
	user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
	user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
	user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
	user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
	user32.SetWindowPos.argtypes = [
		wintypes.HWND, wintypes.HWND,
		ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
		wintypes.UINT,
	]
	user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]

	screen_left = user32.GetSystemMetrics(76)
	screen_top = user32.GetSystemMetrics(77)
	screen_width = user32.GetSystemMetrics(78)
	screen_height = user32.GetSystemMetrics(79)

	root = tk.Tk()
	root.overrideredirect(True)
	root.attributes("-topmost", True)
	root.attributes("-transparentcolor", TRANSPARENT_COLOR)
	root.geometry(f"{screen_width}x{screen_height}{screen_left:+d}{screen_top:+d}")
	canvas = tk.Canvas(
		root,
		background=TRANSPARENT_COLOR,
		borderwidth=0,
		highlightthickness=0,
	)
	canvas.pack(fill="both", expand=True)
	root.update()  # make sure the real window exists

	# Use the top-level window, not Tk's inner child window
	hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()

	GWL_EXSTYLE = -20
	WS_EX_TRANSPARENT = 0x00000020  # click-through
	WS_EX_TOOLWINDOW = 0x00000080   # hide from alt-tab
	style = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
	# Tk already made it layered via -transparentcolor; don't re-add/overwrite that
	user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW)
	user32.SetWindowPos(hwnd, wintypes.HWND(-1), 0, 0, 0, 0, 0x0002 | 0x0001 | 0x0010 | 0x0040)

	# Exclude overlay from screenshots so the model doesn't see its own boxes
	user32.SetWindowDisplayAffinity(hwnd, 0x00000011)  # WDA_EXCLUDEFROMCAPTURE


	mailbox: LatestMailbox[list[tuple[int, int, Image.Image]] | Exception] = LatestMailbox()
	stop_worker = threading.Event()
	cam = dxcam.create(output_color="BGR")

	def process_screenshots():
		try:
			while not stop_worker.is_set():
				screen_bgr = cam.grab()
				if screen_bgr is None:
					if stop_worker.wait(0.002):
						break
					continue
				predictions = model.predict(
					source=screen_bgr,
					imgsz=IMAGE_SIZE,
					conf=CONFIDENCE,
					verbose=False,
					device=0,
					)
				result = next(iter(predictions), None)

				boxes = result.boxes if isinstance(result, Results) else None
				blurred_faces: list[tuple[int, int, Image.Image]] = []
				for box in boxes or ():
					class_name = str(model.names[int(box.cls[0])])
					if "face" not in class_name.casefold():
						continue
					left, top, right, bottom = map(int, box.xyxy[0].tolist())
					padding_x = round((right - left) * 0.1)
					padding_y = round((bottom - top) * 0.1)
					left = max(0, left - padding_x)
					top = max(0, top - padding_y)
					right = min(screen_bgr.shape[1], right + padding_x)
					bottom = min(screen_bgr.shape[0], bottom + padding_y)
					if right <= left or bottom <= top:
						continue

					face = screen_bgr[top:bottom, left:right]
					sigma = max(12, min(face.shape[0], face.shape[1]) / 5)
					blurred = cv2.GaussianBlur(face, (0, 0), sigmaX=sigma)
					blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
					blurred_faces.append((left, top, Image.fromarray(blurred_rgb)))
				mailbox.put(blurred_faces)
				if stop_worker.wait(UPDATE_INTERVAL_MS / 1000):
					break
		except Exception as error:
			mailbox.put(error)

	def close_overlay():
		stop_worker.set()
		root.destroy()

	root.protocol("WM_DELETE_WINDOW", close_overlay)

	def update_overlay():
		if user32.GetAsyncKeyState(0x1B) & 0x8000:
			close_overlay()
			return

		item = mailbox.get()
		if isinstance(item, Exception):
			messagebox.showerror("Blocker error", str(item), parent=root)
			close_overlay()
			return
		if item is not None:
			canvas.delete("all")
			face_images = []
			for left, top, blurred_face in item:
				face_image = ImageTk.PhotoImage(blurred_face)
				face_images.append(face_image)
				canvas.create_image(left, top, image=face_image, anchor="nw")
			canvas.face_images = face_images

		root.after(UPDATE_INTERVAL_MS, update_overlay)

	worker = threading.Thread(target=process_screenshots, daemon=True)
	worker.start()
	root.after(0, update_overlay)
	try:
		root.mainloop()
	except KeyboardInterrupt:
		close_overlay()


if __name__ == "__main__":
	main()
