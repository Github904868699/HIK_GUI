# -*- coding: utf-8 -*-
"""灰度形状检测版海康相机识别软件。

该版本针对黑白相机，只检测长方形与正方形物料，同时保留 Modbus 通讯
能力，并使用 ``config.json`` 对识别及通讯参数进行配置。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon

from camera import (
    APP_ICON,
    APP_TITLE,
    CONFIG_PATH,
    MAX_AREA,
    MIN_AREA,
    RESULT_BASE_ADDR,
    UI_PAINT_FPS,
    HikGrabber,
    ModbusRegisterModel,
    UsbGrabber,
    _is_right_angle_quad,
    _quad_aspect_ratio,
    list_local_ipv4_addresses,
    load_config,
    result_codes_from_cmd_map,
    resource_path,
    rounded_qpixmap,
    start_modbus_server,
)


APP_TITLE_GRAY = f"{APP_TITLE} - 灰度识别"

SHAPE_COLORS = {
    "正方形": (72, 201, 111),  # BGR
    "长方形": (0, 191, 255),
}


def _ensure_odd(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    if value % 2 == 0:
        value += 1
    return value


@dataclass
class GrayDetectionSettings:
    """可在 ``config.json`` 中配置的灰度检测参数。"""

    min_area: int = MIN_AREA
    max_area: int = MAX_AREA
    approx_epsilon: float = 0.03
    right_angle_tolerance: float = 0.25
    aspect_square_max: float = 1.15
    aspect_rect_min: float = 1.2
    aspect_rect_max: float = 5.0
    merge_distance: float = 12.0
    gaussian_kernel: int = 5
    morph_iterations: int = 2
    adaptive_block_size: int = 21
    adaptive_c: float = 5.0
    canny_threshold1: int = 40
    canny_threshold2: int = 120
    edge_dilate_iterations: int = 1
    use_otsu: bool = True
    use_invert: bool = True
    use_adaptive: bool = True
    use_adaptive_invert: bool = True
    use_edges: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "GrayDetectionSettings":
        data = dict(data or {})
        data.setdefault("min_area", MIN_AREA)
        data.setdefault("max_area", MAX_AREA)
        data.setdefault("approx_epsilon", 0.03)
        data.setdefault("right_angle_tolerance", 0.25)
        data.setdefault("aspect_square_max", 1.15)
        data.setdefault("aspect_rect_min", 1.2)
        data.setdefault("aspect_rect_max", 5.0)
        data.setdefault("merge_distance", 12.0)
        data.setdefault("gaussian_kernel", 5)
        data.setdefault("morph_iterations", 2)
        data.setdefault("adaptive_block_size", 21)
        data.setdefault("adaptive_c", 5.0)
        data.setdefault("canny_threshold1", 40)
        data.setdefault("canny_threshold2", 120)
        data.setdefault("edge_dilate_iterations", 1)
        data.setdefault("use_otsu", True)
        data.setdefault("use_invert", True)
        data.setdefault("use_adaptive", True)
        data.setdefault("use_adaptive_invert", True)
        data.setdefault("use_edges", True)
        data["gaussian_kernel"] = _ensure_odd(int(data.get("gaussian_kernel", 5)))
        data["adaptive_block_size"] = _ensure_odd(int(data.get("adaptive_block_size", 21)))
        return cls(**data)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        out["gaussian_kernel"] = _ensure_odd(int(self.gaussian_kernel))
        out["adaptive_block_size"] = _ensure_odd(int(self.adaptive_block_size))
        return out


@dataclass
class GrayDetection:
    label: str
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]

    @property
    def text_position(self) -> Tuple[int, int]:
        x, y, w, _ = self.bbox
        return int(x), max(24, int(y) - 10)


def _iter_thresholds(gray: np.ndarray, settings: GrayDetectionSettings) -> Iterable[np.ndarray]:
    kernel = np.ones((3, 3), np.uint8)
    blur = cv2.GaussianBlur(
        gray,
        (_ensure_odd(settings.gaussian_kernel), _ensure_odd(settings.gaussian_kernel)),
        0,
    )

    if settings.use_otsu:
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield cv2.morphologyEx(
            otsu,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=max(1, int(settings.morph_iterations)),
        )
    if settings.use_invert:
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(otsu)
        yield cv2.morphologyEx(
            inv,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=max(1, int(settings.morph_iterations)),
        )

    block_size = _ensure_odd(settings.adaptive_block_size)
    if settings.use_adaptive:
        adaptive = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            float(settings.adaptive_c),
        )
        yield cv2.morphologyEx(
            adaptive,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=max(1, int(settings.morph_iterations)),
        )
    if settings.use_adaptive_invert:
        adaptive = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            float(settings.adaptive_c),
        )
        inv = cv2.bitwise_not(adaptive)
        yield cv2.morphologyEx(
            inv,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=max(1, int(settings.morph_iterations)),
        )

    if settings.use_edges:
        edges = cv2.Canny(
            blur,
            int(settings.canny_threshold1),
            int(settings.canny_threshold2),
        )
        yield cv2.dilate(
            edges,
            kernel,
            iterations=max(1, int(settings.edge_dilate_iterations)),
        )


def detect_gray_shapes(
    frame_bgr: np.ndarray,
    settings: GrayDetectionSettings,
) -> List[GrayDetection]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detections: List[GrayDetection] = []
    centers: List[Tuple[float, float]] = []

    for mask in _iter_thresholds(gray, settings):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (settings.min_area < area < settings.max_area):
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(cnt, float(settings.approx_epsilon) * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            if not _is_right_angle_quad(approx, tolerance=float(settings.right_angle_tolerance)):
                continue
            aspect = _quad_aspect_ratio(approx)
            if aspect is None:
                continue
            if aspect <= float(settings.aspect_square_max):
                label = "正方形"
            elif float(settings.aspect_rect_min) <= aspect <= float(settings.aspect_rect_max):
                label = "长方形"
            else:
                continue
            pts = approx.reshape(-1, 2).astype(float)
            center = tuple(pts.mean(axis=0))
            if any(
                np.linalg.norm(np.array(center) - np.array(prev)) < float(settings.merge_distance)
                for prev in centers
            ):
                continue
            centers.append(center)
            bbox = cv2.boundingRect(approx)
            detections.append(GrayDetection(label=label, contour=approx, bbox=bbox))

    return detections


class GrayMainWindow(QtWidgets.QWidget):
    modbus_trigger_sig = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__(None, QtCore.Qt.Window)
        self.setWindowTitle(APP_TITLE_GRAY)
        self.resize(1140, 700)

        self.config: Dict[str, object] = load_config(CONFIG_PATH)
        if not isinstance(self.config.get("ui"), dict):
            self.config["ui"] = {}
        detection_cfg = self.config.get("detection", {})
        self.detection_settings = GrayDetectionSettings.from_dict(detection_cfg)
        self.config["detection"] = self.detection_settings.to_dict()

        self.result_codes = result_codes_from_cmd_map(self.config.get("cmd_map", {}))

        server_cfg = dict(self.config.get("server", {}) or {})
        self.modbus_host: str = str(server_cfg.get("host", "0.0.0.0"))
        self.modbus_port: int = int(server_cfg.get("port", 502))
        self.modbus_model = ModbusRegisterModel(size=16)
        self.modbus_model.set_register(0, 0)
        self.modbus_server = None
        self.modbus_error: Optional[str] = None
        self._modbus_shutdown_event: Optional[threading.Event] = None
        self._last_result_count = 0

        self.last_frame_bgr: Optional[np.ndarray] = None
        self.last_detections: List[GrayDetection] = []
        self._last_paint_ts = 0.0

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
        self.video_lbl.setMinimumSize(720, 520)
        right_panel.addWidget(self.video_lbl, 1)

        self.msg_frame = QtWidgets.QFrame()
        self.msg_frame.setObjectName("msgBar")
        self.msg_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.msg_frame.setFixedHeight(54)
        msg_layout = QtWidgets.QVBoxLayout(self.msg_frame)
        msg_layout.setContentsMargins(16, 8, 16, 8)
        self.msg_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        msg_layout.addWidget(self.msg_label)

        self.msg_frame.setStyleSheet(
            """
            #msgBar {
                background: rgba(255, 255, 255, 168);
                border-radius: 12px;
            }
            #msgBar QLabel {
                color: #1a1a1a;
                font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC";
                font-size: 17px;
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

        self.modbus_trigger_sig.connect(self._on_modbus_trigger)
        self._refresh_modbus_ip()
        self._start_modbus_server()
        self._publish_modbus_result(0)

        self.grabber: Optional[QtCore.QThread] = None
        self.start_camera()

    def _init_controls(self) -> None:
        vbox = QtWidgets.QVBoxLayout(self.ctrl_panel)
        vbox.setAlignment(QtCore.Qt.AlignTop)
        vbox.setSpacing(10)

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
        self.recognize_btn.clicked.connect(lambda: self._handle_recognition_request(manual=True))
        vbox.addWidget(self.recognize_btn)

        self._init_detection_group(vbox)
        self._init_modbus_group(vbox)

        self.status_box: Optional[QtWidgets.QTextEdit] = None
        vbox.addStretch(1)

    def _init_detection_group(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("识别参数")
        group.setCheckable(True)
        group.setChecked(False)
        group.setFlat(True)

        container = QtWidgets.QWidget()
        container.setVisible(False)
        form = QtWidgets.QFormLayout(container)
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        form.setContentsMargins(6, 6, 6, 6)

        group_lay = QtWidgets.QVBoxLayout()
        group_lay.setContentsMargins(0, 0, 0, 0)
        group_lay.addWidget(container)
        group.setLayout(group_lay)

        group.toggled.connect(container.setVisible)
        group.toggled.connect(self._on_detect_group_toggled)

        self._preset_combos: List[Tuple[QtWidgets.QComboBox, List[Tuple[str, Dict[str, object]]]]] = []

        def add_combo(label: str, presets: List[Tuple[str, Dict[str, object]]]) -> None:
            combo = QtWidgets.QComboBox()
            for name, values in presets:
                combo.addItem(name, values)
            combo.currentIndexChanged.connect(self._on_preset_changed)
            form.addRow(label, combo)
            self._preset_combos.append((combo, presets))

        add_combo(
            "面积范围",
            [
                ("标准物料", {"min_area": 500, "max_area": 60000}),
                ("小尺寸", {"min_area": 250, "max_area": 30000}),
                ("大尺寸", {"min_area": 900, "max_area": 120000}),
            ],
        )
        add_combo(
            "识别精度",
            [
                ("标准", {"approx_epsilon": 0.03, "right_angle_tolerance": 0.25}),
                ("严格", {"approx_epsilon": 0.02, "right_angle_tolerance": 0.18}),
                ("宽松", {"approx_epsilon": 0.05, "right_angle_tolerance": 0.32}),
            ],
        )
        add_combo(
            "形状判定",
            [
                ("标准", {"aspect_square_max": 1.12, "aspect_rect_min": 1.2, "aspect_rect_max": 4.0}),
                ("严格", {"aspect_square_max": 1.08, "aspect_rect_min": 1.25, "aspect_rect_max": 3.0}),
                ("宽松", {"aspect_square_max": 1.18, "aspect_rect_min": 1.15, "aspect_rect_max": 5.0}),
            ],
        )
        add_combo(
            "阈值策略",
            [
                (
                    "自动",
                    {
                        "use_otsu": True,
                        "use_invert": True,
                        "use_adaptive": True,
                        "use_adaptive_invert": False,
                        "use_edges": True,
                    },
                ),
                (
                    "自适应优先",
                    {
                        "use_otsu": False,
                        "use_invert": False,
                        "use_adaptive": True,
                        "use_adaptive_invert": True,
                        "use_edges": True,
                    },
                ),
                (
                    "纯阈值",
                    {
                        "use_otsu": True,
                        "use_invert": False,
                        "use_adaptive": False,
                        "use_adaptive_invert": False,
                        "use_edges": False,
                    },
                ),
            ],
        )
        add_combo(
            "滤波增强",
            [
                (
                    "标准",
                    {
                        "gaussian_kernel": 5,
                        "morph_iterations": 2,
                        "merge_distance": 12.0,
                        "adaptive_block_size": 21,
                        "adaptive_c": 5.0,
                        "canny_threshold1": 40,
                        "canny_threshold2": 120,
                        "edge_dilate_iterations": 1,
                    },
                ),
                (
                    "平滑",
                    {
                        "gaussian_kernel": 7,
                        "morph_iterations": 1,
                        "merge_distance": 14.0,
                        "adaptive_block_size": 25,
                        "adaptive_c": 7.0,
                        "canny_threshold1": 35,
                        "canny_threshold2": 100,
                        "edge_dilate_iterations": 1,
                    },
                ),
                (
                    "锐利",
                    {
                        "gaussian_kernel": 3,
                        "morph_iterations": 3,
                        "merge_distance": 10.0,
                        "adaptive_block_size": 17,
                        "adaptive_c": 3.0,
                        "canny_threshold1": 45,
                        "canny_threshold2": 140,
                        "edge_dilate_iterations": 2,
                    },
                ),
            ],
        )

        btn_reset = QtWidgets.QPushButton("恢复默认参数")
        btn_reset.clicked.connect(self._reset_detection_defaults)
        form.addRow(btn_reset)

        parent_layout.addWidget(group)

        self.detect_group = group
        self._detect_container = container
        self._load_presets_to_ui()
        self._apply_presets_from_ui()

    def _init_modbus_group(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Modbus 通讯")
        lay = QtWidgets.QGridLayout(group)

        host_label = QtWidgets.QLabel("监听地址")
        self.modbus_ip_combo = QtWidgets.QComboBox()
        self.modbus_ip_combo.currentIndexChanged.connect(self._on_modbus_ip_changed)
        lay.addWidget(host_label, 0, 0)
        lay.addWidget(self.modbus_ip_combo, 0, 1, 1, 2)

        port_label = QtWidgets.QLabel("端口")
        self.modbus_port_spin = QtWidgets.QSpinBox()
        self.modbus_port_spin.setRange(1, 65535)
        self.modbus_port_spin.setValue(self.modbus_port)
        self.modbus_port_spin.valueChanged.connect(self._on_modbus_port_changed)
        lay.addWidget(port_label, 1, 0)
        lay.addWidget(self.modbus_port_spin, 1, 1, 1, 2)

        self.btn_modbus_restart = QtWidgets.QPushButton("重启服务器")
        self.btn_modbus_restart.clicked.connect(lambda: self._start_modbus_server())
        lay.addWidget(self.btn_modbus_restart, 2, 0, 1, 1)

        self.btn_modbus_stop = QtWidgets.QPushButton("停止")
        self.btn_modbus_stop.clicked.connect(self._on_modbus_stop_clicked)
        lay.addWidget(self.btn_modbus_stop, 2, 1, 1, 1)

        self.lbl_modbus_status = QtWidgets.QLabel()
        self.lbl_modbus_status.setWordWrap(True)
        lay.addWidget(self.lbl_modbus_status, 3, 0, 1, 3)

        parent_layout.addWidget(group)

    def _load_presets_to_ui(self) -> None:
        settings = self.detection_settings
        for combo, presets in self._preset_combos:
            idx = self._find_preset_index(presets, settings)
            blocker = QtCore.QSignalBlocker(combo)
            combo.setCurrentIndex(idx)
            del blocker

        ui_cfg = self.config.get("ui")
        visible = False
        if isinstance(ui_cfg, dict):
            visible = bool(ui_cfg.get("show_detection_panel", False))
        blocker = QtCore.QSignalBlocker(self.detect_group)
        self.detect_group.setChecked(visible)
        del blocker
        self._detect_container.setVisible(visible)

    @staticmethod
    def _value_close(current: object, target: object) -> bool:
        if isinstance(target, bool):
            return bool(current) is bool(target)
        try:
            cur = float(current)
            tgt = float(target)
        except (TypeError, ValueError):
            return False
        if abs(tgt) < 1:
            tol = 0.01
        else:
            tol = max(0.05, abs(tgt) * 0.01)
        return abs(cur - tgt) <= tol

    def _find_preset_index(
        self,
        presets: Sequence[Tuple[str, Dict[str, object]]],
        settings: GrayDetectionSettings,
    ) -> int:
        for idx, (_name, values) in enumerate(presets):
            matched = True
            for key, target in values.items():
                current = getattr(settings, key, None)
                if not self._value_close(current, target):
                    matched = False
                    break
            if matched:
                return idx
        return 0

    def _on_preset_changed(self) -> None:
        self._apply_presets_from_ui()

    def _apply_presets_from_ui(self) -> None:
        merged: Dict[str, object] = dict(self.detection_settings.to_dict())
        for combo, _ in self._preset_combos:
            data = combo.currentData()
            if isinstance(data, dict):
                merged.update(data)
        new_settings = GrayDetectionSettings.from_dict(merged)
        changed = new_settings.to_dict() != self.detection_settings.to_dict()
        self.detection_settings = new_settings
        self.config["detection"] = self.detection_settings.to_dict()
        if changed:
            self._save_config()

    def _reset_detection_defaults(self) -> None:
        self.detection_settings = GrayDetectionSettings()
        self.config["detection"] = self.detection_settings.to_dict()
        self._load_presets_to_ui()
        self._save_config()

    def _on_detect_group_toggled(self, checked: bool) -> None:
        ui_cfg = self.config.setdefault("ui", {})
        if isinstance(ui_cfg, dict):
            ui_cfg["show_detection_panel"] = bool(checked)
        self._save_config()

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
        detections = detect_gray_shapes(frame_draw, self.detection_settings)
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

    def _format_summary(self, detections: Sequence[GrayDetection]) -> str:
        if not detections:
            return "未检测到目标"
        counts: Dict[str, int] = {}
        for det in detections:
            counts[det.label] = counts.get(det.label, 0) + 1
        parts = [f"{label} x{count}" for label, count in counts.items()]
        return "，".join(parts)

    def _handle_recognition_request(self, manual: bool) -> None:
        model = self.modbus_model
        if model:
            if manual:
                model.set_register(0, 1)
            self._publish_modbus_result([])

        if self.last_frame_bgr is None:
            self._append_status("[识别] 当前没有画面")
            self.msg_label.setText("未检测到目标")
            self.msg_timer.start(2000)
            self._publish_modbus_result(0xFF)
            if model:
                model.set_register(0, 0)
            return

        img = self.last_frame_bgr.copy()
        detections = detect_gray_shapes(img, self.detection_settings)
        self.msg_label.setText(self._format_summary(detections))
        self.msg_timer.start(2000)

        result_values: List[int] = []
        if detections:
            summary = self._format_summary(detections)
            self._append_status(f"[识别] {summary}")
            for det in detections:
                code = self.result_codes.get(det.label)
                if code is not None:
                    result_values.append(int(code))
            if result_values:
                self._publish_modbus_result(result_values)
                codes_text = ", ".join(f"0x{code:04X}" for code in result_values)
                self._append_status(f"[MODBUS] 写入结果: {codes_text}")
            else:
                self._publish_modbus_result(0)
                self._append_status("[识别] 未找到匹配的结果编码，写入 0")
        else:
            self._append_status("[识别] 未检测到目标")
            self._publish_modbus_result(0xFF)

        if model:
            model.set_register(0, 0)

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
        if isinstance(self.grabber, HikGrabber):
            self.start_camera(prefer_hik=False)

    def _append_status(self, text: str) -> None:
        if self.status_box:
            self.status_box.append(text)
            self.status_box.moveCursor(QtGui.QTextCursor.End)
        else:
            print(text)

    def _refresh_modbus_ip(self) -> None:
        entries = [("0.0.0.0", "全部网口")] + list_local_ipv4_addresses()
        current = self.modbus_host or "0.0.0.0"
        ips = [ip for ip, _ in entries]
        if current not in ips:
            entries.append((current, "当前"))

        blocker = QtCore.QSignalBlocker(self.modbus_ip_combo)
        self.modbus_ip_combo.clear()
        for ip, iface in entries:
            if iface:
                text = f"{ip} ({iface})"
            else:
                text = ip
            self.modbus_ip_combo.addItem(text, ip)
        idx = self.modbus_ip_combo.findData(current)
        if idx < 0:
            idx = 0
        self.modbus_ip_combo.setCurrentIndex(idx)
        del blocker

    def _on_modbus_ip_changed(self, index: int) -> None:
        if index < 0:
            return
        data = self.modbus_ip_combo.itemData(index)
        host = str(data or "0.0.0.0")
        if host != self.modbus_host:
            self.modbus_host = host
            self._start_modbus_server()

    def _on_modbus_port_changed(self, value: int) -> None:
        if value != self.modbus_port:
            self.modbus_port = int(value)
            self._start_modbus_server()

    def _on_modbus_stop_clicked(self) -> None:
        self._stop_modbus_server()
        self.modbus_error = "手动停止"
        self._update_modbus_status()
        self._save_config()

    def _stop_modbus_server(self) -> Optional[threading.Event]:
        server = self.modbus_server
        if not server:
            self._modbus_shutdown_event = None
            return None

        self.modbus_server = None
        done = threading.Event()

        def do_shutdown() -> None:
            try:
                server.shutdown()
            except Exception as exc:
                self._append_status(f"[MODBUS] 停止异常: {exc}")
            finally:
                try:
                    server.server_close()
                except Exception as exc:
                    self._append_status(f"[MODBUS] 关闭异常: {exc}")
                done.set()

        threading.Thread(target=do_shutdown, daemon=True).start()
        self._modbus_shutdown_event = done
        return done

    def _start_modbus_server(self) -> None:
        shutdown_event = self._stop_modbus_server()
        if isinstance(shutdown_event, threading.Event):
            if not shutdown_event.wait(timeout=1.0):
                self._append_status("[MODBUS] 等待旧连接关闭超时，继续启动新服务器")
            self._modbus_shutdown_event = None

        self.modbus_error = None
        try:
            self.modbus_server = start_modbus_server(
                self.modbus_host, self.modbus_port, self.modbus_model, on_write=self._on_modbus_write
            )
        except Exception as exc:
            self.modbus_server = None
            self.modbus_error = str(exc)
            self._append_status(f"[MODBUS] 启动失败: {exc}")

        self.config.setdefault("server", {})
        server_cfg = self.config["server"]
        if isinstance(server_cfg, dict):
            server_cfg["host"] = self.modbus_host
            server_cfg["port"] = self.modbus_port
        self._save_config()
        self._update_modbus_status()

    def _update_modbus_status(self) -> None:
        if self.modbus_server and not self.modbus_error:
            text = f"运行中：{self.modbus_host}:{self.modbus_port}"
        else:
            text = f"已停止：{self.modbus_error or '未启动'}"
        self.lbl_modbus_status.setText(text)

    def _on_modbus_write(self, addr: int, value: int) -> None:
        if addr == 0 and value == 1:
            self._append_status("[MODBUS] 收到拍照请求")
            self.modbus_trigger_sig.emit()

    @QtCore.pyqtSlot()
    def _on_modbus_trigger(self) -> None:
        self._handle_recognition_request(manual=False)

    def _publish_modbus_result(self, values) -> None:
        model = self.modbus_model
        if not model:
            return
        if isinstance(values, int):
            sanitized = [values & 0xFFFF]
        else:
            items = list(values)
            if not items:
                items = [0]
            sanitized = [int(v) & 0xFFFF for v in items]
        if self._last_result_count > len(sanitized):
            sanitized.extend([0] * (self._last_result_count - len(sanitized)))
        model.write(RESULT_BASE_ADDR, sanitized)
        self._last_result_count = len(sanitized)

    def _save_config(self) -> None:
        data = dict(self.config)
        data["detection"] = self.detection_settings.to_dict()
        server_cfg = data.setdefault("server", {})
        if isinstance(server_cfg, dict):
            server_cfg["host"] = self.modbus_host
            server_cfg["port"] = self.modbus_port
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._append_status(f"[CONFIG] 保存失败: {exc}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI 事件
        self._stop_modbus_server()
        if isinstance(self._modbus_shutdown_event, threading.Event):
            self._modbus_shutdown_event.wait(timeout=1.0)
        self.stop_camera()
        self._save_config()
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
