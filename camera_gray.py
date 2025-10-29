# -*- coding: utf-8 -*-
"""灰度形状检测版海康相机识别软件。

该版本针对黑白相机，只检测长方形与正方形物料。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon

from camera import (
    APP_ICON,
    APP_TITLE,
    MIN_AREA,
    MAX_AREA,
    UI_PAINT_FPS,
    HikGrabber,
    UsbGrabber,
    rounded_qpixmap,
    resource_path,
    _is_right_angle_quad,
    _quad_aspect_ratio,
)


APP_TITLE_GRAY = f"{APP_TITLE} - 灰度识别"

SHAPE_COLORS = {
    "正方形": (72, 201, 111),  # BGR
    "长方形": (0, 191, 255),
}


@dataclass
class GrayDetection:
    label: str
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]

    @property
    def text_position(self) -> Tuple[int, int]:
        x, y, w, _ = self.bbox
        return int(x), max(24, int(y) - 10)


def _iter_thresholds(gray: np.ndarray) -> Iterable[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    yield cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    yield cv2.morphologyEx(cv2.bitwise_not(otsu), cv2.MORPH_CLOSE, kernel, iterations=2)

    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        5,
    )
    yield cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)
    yield cv2.morphologyEx(cv2.bitwise_not(adaptive), cv2.MORPH_CLOSE, kernel, iterations=1)

    edges = cv2.Canny(blur, 40, 120)
    yield cv2.dilate(edges, kernel, iterations=1)


def detect_gray_shapes(frame_bgr: np.ndarray) -> List[GrayDetection]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detections: List[GrayDetection] = []
    centers: List[Tuple[float, float]] = []

    for mask in _iter_thresholds(gray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_AREA < area < MAX_AREA):
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            if not _is_right_angle_quad(approx, tolerance=0.25):
                continue
            aspect = _quad_aspect_ratio(approx)
            if aspect is None:
                continue
            label = "正方形" if aspect <= 1.15 else "长方形"
            pts = approx.reshape(-1, 2).astype(float)
            center = tuple(pts.mean(axis=0))
            if any(np.linalg.norm(np.array(center) - np.array(prev)) < 12.0 for prev in centers):
                continue
            centers.append(center)
            bbox = cv2.boundingRect(approx)
            detections.append(GrayDetection(label=label, contour=approx, bbox=bbox))

    return detections


class GrayMainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(None, QtCore.Qt.Window)
        self.setWindowTitle(APP_TITLE_GRAY)
        self.resize(1100, 680)

        self.last_frame_bgr: np.ndarray | None = None
        self._last_paint_ts = 0.0
        self.last_detections: List[GrayDetection] = []

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.ctrl_panel = QtWidgets.QFrame()
        self.ctrl_panel.setFixedWidth(300)
        layout.addWidget(self.ctrl_panel)

        right_panel = QtWidgets.QVBoxLayout()
        header = QtWidgets.QHBoxLayout()
        self.lbl_cam = QtWidgets.QLabel("Camera —")
        self.lbl_fps = QtWidgets.QLabel("FPS —")
        header.addWidget(self.lbl_cam, 1, QtCore.Qt.AlignLeft)
        header.addWidget(self.lbl_fps, 0, QtCore.Qt.AlignRight)
        right_panel.addLayout(header)

        self.video_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_lbl.setMinimumSize(640, 480)
        right_panel.addWidget(self.video_lbl, 1)

        self.msg_frame = QtWidgets.QFrame()
        self.msg_frame.setObjectName("msgBar")
        self.msg_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.msg_frame.setFixedHeight(48)
        msg_layout = QtWidgets.QVBoxLayout(self.msg_frame)
        msg_layout.setContentsMargins(14, 8, 14, 8)
        self.msg_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        msg_layout.addWidget(self.msg_label)

        self.msg_frame.setStyleSheet(
            """
            #msgBar {
                background: rgba(255, 255, 255, 160);
                border-radius: 12px;
            }
            #msgBar QLabel {
                color: #1a1a1a;
                font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC";
                font-size: 16px;
                font-weight: 600;
            }
            """
        )

        shadow = QtWidgets.QGraphicsDropShadowEffect(self.msg_frame)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 4)
        shadow.setColor(QtGui.QColor(0, 0, 0, 140))
        self.msg_frame.setGraphicsEffect(shadow)

        right_panel.addWidget(self.msg_frame)
        layout.addLayout(right_panel, 1)

        self.msg_timer = QtCore.QTimer(self)
        self.msg_timer.setSingleShot(True)
        self.msg_timer.timeout.connect(lambda: self.msg_label.setText(""))

        self._init_controls()

        self.grabber: QtCore.QThread | None = None
        self.start_camera()

    def _init_controls(self) -> None:
        vbox = QtWidgets.QVBoxLayout(self.ctrl_panel)
        vbox.setAlignment(QtCore.Qt.AlignTop)

        g_cam = QtWidgets.QGroupBox("相机")
        cam_lay = QtWidgets.QHBoxLayout(g_cam)
        self.btn_reopen = QtWidgets.QPushButton("重连")
        self.btn_stop = QtWidgets.QPushButton("停止")
        self.btn_reopen.clicked.connect(self.reopen_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        cam_lay.addWidget(self.btn_reopen)
        cam_lay.addWidget(self.btn_stop)
        vbox.addWidget(g_cam)

        self.recognize_btn = QtWidgets.QPushButton("手动识别")
        self.recognize_btn.clicked.connect(self.recognize_once)
        vbox.addWidget(self.recognize_btn)

        self.status_box = QtWidgets.QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setFixedHeight(160)
        vbox.addWidget(self.status_box)
        vbox.addStretch(1)

    def start_camera(self, prefer_hik: bool = True) -> None:
        self.stop_camera()
        if prefer_hik:
            try:
                grabber = HikGrabber(self)
            except Exception as exc:  # pragma: no cover - 依赖外部硬件
                self._append_status(f"[HIK] {exc}")
            else:
                self.grabber = grabber
                grabber.frameSignal.connect(self.on_frame_from_camera)
                grabber.infoSignal.connect(self._on_info)
                grabber.errorSignal.connect(self._on_camera_error)
                grabber.start()
                return

        grabber = UsbGrabber(self)
        self.grabber = grabber
        grabber.frameSignal.connect(self.on_frame_from_camera)
        grabber.infoSignal.connect(self._on_info)
        grabber.start()

    def stop_camera(self) -> None:
        if self.grabber and self.grabber.isRunning():
            self.grabber.stop()
            self.grabber.wait(1000)
        self.grabber = None

    def reopen_camera(self) -> None:
        self.start_camera(prefer_hik=True)

    @QtCore.pyqtSlot(np.ndarray)
    def on_frame_from_camera(self, frame_bgr: np.ndarray) -> None:
        now = time.time()
        if (now - self._last_paint_ts) < (1.0 / UI_PAINT_FPS):
            return
        self._last_paint_ts = now

        self.last_frame_bgr = frame_bgr
        frame_draw = frame_bgr.copy()
        detections = detect_gray_shapes(frame_draw)
        self.last_detections = detections

        for det in detections:
            color = SHAPE_COLORS.get(det.label, (0, 255, 0))
            cv2.polylines(frame_draw, [det.contour], True, color, 2)

        rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)

        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setFont(QtGui.QFont("微软雅黑", 18, QtGui.QFont.Bold))
        for det in detections:
            color = SHAPE_COLORS.get(det.label, (0, 255, 0))
            qcolor = QtGui.QColor(color[2], color[1], color[0])
            painter.setPen(qcolor)
            tx, ty = det.text_position
            painter.drawText(tx, ty, det.label)
        painter.end()

        scaled = pix.scaled(
            self.video_lbl.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_lbl.setPixmap(rounded_qpixmap(scaled, 18))

        if not self.msg_timer.isActive():
            self.msg_label.setText(self._format_summary(detections))

    def recognize_once(self) -> None:
        if self.last_frame_bgr is None:
            self.msg_label.setText("未检测到目标")
            self.msg_timer.start(2000)
            return
        img = self.last_frame_bgr.copy()
        detections = detect_gray_shapes(img)
        self.msg_label.setText(self._format_summary(detections))
        self.msg_timer.start(2000)

    def _format_summary(self, detections: Sequence[GrayDetection]) -> str:
        if not detections:
            return "未检测到目标"
        counts: dict[str, int] = {}
        for det in detections:
            counts[det.label] = counts.get(det.label, 0) + 1
        parts = [f"{label} x{count}" for label, count in counts.items()]
        return "，".join(parts)

    def _on_info(self, message: str) -> None:
        if message.startswith("[INFO]"):
            text = message.replace("[INFO]", "").strip()
            self.lbl_cam.setText(text or "Camera —")
        elif message.startswith("[FPS]"):
            text = message.replace("[FPS]", "FPS").strip()
            self.lbl_fps.setText(text)
        else:
            self._append_status(message)

    def _on_camera_error(self, message: str) -> None:
        self._append_status(f"[ERROR] {message}")
        self.msg_label.setText(message)
        self.msg_timer.start(4000)

    def _append_status(self, text: str) -> None:
        self.status_box.append(text)
        self.status_box.moveCursor(QtGui.QTextCursor.End)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI 事件
        self.stop_camera()
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    try:
        app.setWindowIcon(QIcon(resource_path(APP_ICON)))
    except Exception:
        pass
    win = GrayMainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
