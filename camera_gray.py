# -*- coding: utf-8 -*-
"""灰度形状检测版海康相机识别软件。

该版本针对黑白相机，只检测长方形与正方形物料。
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon

from camera import (
    APP_ICON,
    APP_TITLE,
    UI_PAINT_FPS,
    HikGrabber,
    MaskWindow,
    ModbusRegisterModel,
    RESULT_BASE_ADDR,
    UsbGrabber,
    list_local_ipv4_addresses,
    load_config,
    result_codes_from_cmd_map,
    rounded_qpixmap,
    resource_path,
    start_modbus_server,
    _is_right_angle_quad,
    _quad_aspect_ratio,
)


APP_TITLE_GRAY = f"{APP_TITLE} - 灰度识别"

SHAPE_COLORS = {
    "正方形": (72, 201, 111),  # BGR
    "长方形": (0, 191, 255),
}


class ParamSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(float)

    def __init__(
        self,
        text: str,
        minimum: float,
        maximum: float,
        value: float,
        *,
        step: float = 1.0,
        decimals: int = 0,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(step)
        self._decimals = int(decimals)
        slider_steps = max(1, int(round((self._max - self._min) / self._step)))
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.label = QtWidgets.QLabel(text)
        layout.addWidget(self.label)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, slider_steps)
        layout.addWidget(self.slider, 1)

        self.value_label = QtWidgets.QLabel("0")
        self.value_label.setFixedWidth(72)
        layout.addWidget(self.value_label, 0)

        self.slider.valueChanged.connect(self._on_value_changed)
        self.setValue(value)

    def _format_value(self, value: float) -> str:
        return f"{value:.{self._decimals}f}" if self._decimals else f"{int(round(value))}"

    def _on_value_changed(self, _: int) -> None:
        val = self.value()
        self.value_label.setText(self._format_value(val))
        self.valueChanged.emit(val)

    def value(self) -> float:
        pos = self.slider.value()
        return self._min + pos * self._step

    def setValue(self, value: float) -> None:
        clamped = min(max(value, self._min), self._max)
        pos = int(round((clamped - self._min) / self._step))
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self.value_label.setText(self._format_value(self.value()))


@dataclass
class GrayShapeConfig:
    key: str
    label: str
    enabled: bool
    min_area: int
    max_area: int
    min_aspect: Optional[float] = None
    max_aspect: Optional[float] = None
    sliders: Dict[str, ParamSlider] = field(default_factory=dict)
    checkbox: Optional[QtWidgets.QCheckBox] = None
    panel: Optional[QtWidgets.QWidget] = None

    def accepts(self, area: float, aspect: float) -> bool:
        if area < self.min_area or area > self.max_area:
            return False
        if self.min_aspect is not None and aspect < self.min_aspect:
            return False
        if self.max_aspect is not None and aspect > self.max_aspect:
            return False
        return True

    def sync_from_sliders(self) -> None:
        if "min_area" in self.sliders:
            self.min_area = int(self.sliders["min_area"].value())
        if "max_area" in self.sliders:
            self.max_area = int(self.sliders["max_area"].value())
        if "min_aspect" in self.sliders:
            self.min_aspect = float(self.sliders["min_aspect"].value())
        if "max_aspect" in self.sliders:
            self.max_aspect = float(self.sliders["max_aspect"].value())



@dataclass
class GrayDetection:
    label: str
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]

    @property
    def text_position(self) -> Tuple[int, int]:
        x, y, w, _ = self.bbox
        return int(x), max(24, int(y) - 10)


def _prepare_gray(
    gray: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    clahe_clip: float = 0.0,
    clahe_grid: int = 8,
) -> np.ndarray:
    processed = gray
    if clahe_clip > 0:
        clip = max(0.1, clahe_clip)
        grid = max(1, int(clahe_grid))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        processed = clahe.apply(processed)
    if alpha != 1.0 or beta != 0.0:
        processed = cv2.convertScaleAbs(processed, alpha=max(0.1, alpha), beta=beta)
    return processed


def _iter_thresholds(
    gray: np.ndarray,
    *,
    canny_low: float,
    canny_high: float,
    gradient_thresh: float,
    use_kmeans: bool,
    kmeans_downscale: int,
    kmeans_blur: int,
    kmeans_morph: int,
) -> Iterable[np.ndarray]:
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

    low = max(1, int(round(min(canny_low, canny_high))))
    high = max(low + 1, int(round(max(canny_low, canny_high))))
    edges = cv2.Canny(blur, low, high)
    yield cv2.dilate(edges, kernel, iterations=1)

    if gradient_thresh > 0:
        grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
        _, grad_mask = cv2.threshold(
            grad,
            max(1, float(gradient_thresh)),
            255,
            cv2.THRESH_BINARY,
        )
        grad_mask = cv2.dilate(grad_mask, kernel, iterations=1)
        grad_mask = cv2.morphologyEx(grad_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        yield grad_mask

    if use_kmeans:
        yield from _kmeans_masks(
            gray,
            downscale=kmeans_downscale,
            blur_kernel=kmeans_blur,
            morph_iterations=kmeans_morph,
        )


def _kmeans_masks(
    gray: np.ndarray,
    *,
    downscale: int,
    blur_kernel: int,
    morph_iterations: int,
) -> Iterable[np.ndarray]:
    height, width = gray.shape
    scale = max(1, int(round(downscale)))
    if scale > 1:
        small_w = max(1, int(round(width / scale)))
        small_h = max(1, int(round(height / scale)))
        small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        small = gray

    kernel_size = int(round(blur_kernel))
    if kernel_size % 2 == 0:
        kernel_size = max(1, kernel_size - 1)
    if kernel_size >= 3:
        small = cv2.GaussianBlur(small, (kernel_size, kernel_size), 0)

    data = small.reshape((-1, 1)).astype(np.float32)
    if data.size < 2:
        return

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    try:
        _compact, labels, centers = cv2.kmeans(
            data,
            2,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
    except cv2.error:
        return

    labels = labels.reshape(small.shape)
    centers = centers.flatten()
    order = np.argsort(centers)
    base_kernel = np.ones((3, 3), np.uint8)

    for idx in order:
        mask_small = np.zeros_like(small, dtype=np.uint8)
        mask_small[labels == idx] = 255
        if morph_iterations > 0:
            mask_small = cv2.morphologyEx(
                mask_small,
                cv2.MORPH_CLOSE,
                base_kernel,
                iterations=int(morph_iterations),
            )
        mask = cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_NEAREST)
        if cv2.countNonZero(mask) == 0:
            continue
        yield mask


def detect_gray_shapes(
    frame_bgr: np.ndarray,
    shape_cfgs: Sequence[GrayShapeConfig],
    *,
    merge_distance: float,
    angle_tolerance: float,
    approx_epsilon: float,
    contrast_alpha: float = 1.0,
    contrast_beta: float = 0.0,
    clahe_clip_limit: float = 0.0,
    canny_low: float = 20.0,
    canny_high: float = 160.0,
    gradient_thresh: float = 10.0,
    kmeans_enabled: bool = False,
    kmeans_downscale: int = 1,
    kmeans_blur: int = 3,
    kmeans_morph: int = 1,
) -> Tuple[List[GrayDetection], Dict[str, np.ndarray]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = _prepare_gray(
        gray,
        alpha=contrast_alpha,
        beta=contrast_beta,
        clahe_clip=clahe_clip_limit,
    )
    detections: List[GrayDetection] = []
    centers: List[np.ndarray] = []
    masks: Dict[str, np.ndarray] = {}
    height, width = gray.shape
    for cfg in shape_cfgs:
        if cfg.enabled:
            masks[cfg.label] = np.zeros((height, width), dtype=np.uint8)

    for mask in _iter_thresholds(
        gray,
        canny_low=canny_low,
        canny_high=canny_high,
        gradient_thresh=gradient_thresh,
        use_kmeans=kmeans_enabled,
        kmeans_downscale=kmeans_downscale,
        kmeans_blur=kmeans_blur,
        kmeans_morph=kmeans_morph,
    ):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(cnt, max(1.0, approx_epsilon * perimeter), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            if not _is_right_angle_quad(approx, tolerance=angle_tolerance):
                continue
            aspect = _quad_aspect_ratio(approx)
            if aspect is None:
                continue
            target_cfg: GrayShapeConfig | None = None
            for cfg in shape_cfgs:
                if not cfg.enabled:
                    continue
                if cfg.accepts(area, aspect):
                    target_cfg = cfg
                    break
            if target_cfg is None:
                continue
            pts = approx.reshape(-1, 2).astype(float)
            center = pts.mean(axis=0)
            if any(np.linalg.norm(center - prev) < merge_distance for prev in centers):
                continue
            centers.append(center)
            bbox = cv2.boundingRect(approx)
            detections.append(GrayDetection(label=target_cfg.label, contour=approx, bbox=bbox))
            if target_cfg.label in masks:
                cv2.drawContours(masks[target_cfg.label], [approx], -1, 255, thickness=-1)

    return detections, masks


class GrayMainWindow(QtWidgets.QWidget):
    modbus_trigger_sig = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__(None, QtCore.Qt.Window)
        self.setWindowTitle(APP_TITLE_GRAY)
        self.resize(1100, 680)

        self.config = load_config()
        gray_section = self.config.get("gray_shapes", {})
        shapes_conf: Mapping[str, dict] = gray_section.get("shapes", {})
        self.shape_cfgs: Dict[str, GrayShapeConfig] = self._create_shape_configs(shapes_conf)
        self.shape_cfg_list: List[GrayShapeConfig] = [
            self.shape_cfgs[key]
            for key in ("square", "rectangle")
            if key in self.shape_cfgs
        ]

        self.merge_distance = float(gray_section.get("merge_distance", 12.0))
        self.angle_tolerance = float(gray_section.get("angle_tolerance", 0.25))
        self.approx_epsilon = float(gray_section.get("approx_epsilon", 0.03))
        self.contrast_alpha = float(gray_section.get("contrast_alpha", 1.0))
        self.contrast_beta = float(gray_section.get("contrast_beta", 0.0))
        self.clahe_clip = float(gray_section.get("clahe_clip", 0.0))
        self.canny_low = float(gray_section.get("canny_low", 20.0))
        self.canny_high = float(gray_section.get("canny_high", 160.0))
        self.gradient_thresh = float(gray_section.get("gradient_thresh", 10.0))
        if self.canny_high <= self.canny_low:
            self.canny_high = self.canny_low + 20.0
        self.gradient_thresh = max(0.0, self.gradient_thresh)
        self.kmeans_enabled = bool(gray_section.get("kmeans_enabled", True))
        self.kmeans_downscale = max(1, int(round(gray_section.get("kmeans_downscale", 2))))
        self.kmeans_blur = max(1, int(round(gray_section.get("kmeans_blur", 5))))
        if self.kmeans_blur % 2 == 0:
            self.kmeans_blur += 1
        self.kmeans_morph = max(0, int(round(gray_section.get("kmeans_morph", 2))))

        self.result_codes = result_codes_from_cmd_map(self.config.get("cmd_map", {}))

        self.mask_windows: Dict[str, MaskWindow] = {}
        self.last_masks: Dict[str, np.ndarray] = {}
        self.last_frame_bgr: np.ndarray | None = None
        self._last_paint_ts = 0.0
        self.last_detections: List[GrayDetection] = []
        self._status_messages: List[str] = []

        server_cfg = self.config.get("server", {})
        self.modbus_host = str(server_cfg.get("host", "0.0.0.0") or "0.0.0.0")
        self.modbus_port = int(server_cfg.get("port", 502) or 502)
        self.modbus_model = ModbusRegisterModel(size=16)
        self.modbus_server = None
        self.modbus_error: Optional[str] = None
        self._modbus_shutdown_event: Optional[threading.Event] = None
        self._last_result_count = 0

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.ctrl_panel = QtWidgets.QFrame()
        self.ctrl_panel.setFixedWidth(320)
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

        self.modbus_trigger_sig.connect(self._on_modbus_trigger)
        self._start_modbus_server(self.modbus_host)
        self._refresh_modbus_ip()
        self.ip_refresh_timer = QtCore.QTimer(self)
        self.ip_refresh_timer.setInterval(5000)
        self.ip_refresh_timer.timeout.connect(self._refresh_modbus_ip)
        self.ip_refresh_timer.start()

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

        self._create_shape_controls(vbox)

        g_modbus = QtWidgets.QGroupBox("Modbus")
        form = QtWidgets.QFormLayout(g_modbus)
        self.modbus_ip_combo = QtWidgets.QComboBox()
        self.modbus_ip_combo.currentIndexChanged.connect(self._on_modbus_ip_changed)
        self.modbus_port_label = QtWidgets.QLabel(str(self.modbus_port))
        self.modbus_status_lbl = QtWidgets.QLabel("")
        form.addRow("服务器IP:", self.modbus_ip_combo)
        form.addRow("端口:", self.modbus_port_label)
        form.addRow("状态:", self.modbus_status_lbl)
        vbox.addWidget(g_modbus)

        vbox.addStretch(1)

    def _create_shape_configs(self, shapes_conf: Mapping[str, dict]) -> Dict[str, GrayShapeConfig]:
        defaults = {
            "square": ("正方形", 1200, 120000, 1.0, 1.2),
            "rectangle": ("长方形", 1500, 180000, 1.2, 3.5),
        }
        shapes: Dict[str, GrayShapeConfig] = {}
        for key, (label, min_area, max_area, min_aspect, max_aspect) in defaults.items():
            data = shapes_conf.get(key, {}) or {}
            cfg = GrayShapeConfig(
                key=key,
                label=label,
                enabled=bool(data.get("enabled", True)),
                min_area=int(data.get("min_area", min_area)),
                max_area=int(data.get("max_area", max_area)),
                min_aspect=float(data.get("min_aspect", min_aspect)) if data.get("min_aspect", min_aspect) is not None else None,
                max_aspect=float(data.get("max_aspect", max_aspect)) if data.get("max_aspect", max_aspect) is not None else None,
            )
            shapes[key] = cfg
        return shapes

    def _create_shape_controls(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        if not self.shape_cfg_list:
            return
        group = QtWidgets.QGroupBox("形状检测")
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setSpacing(10)
        for cfg in self.shape_cfg_list:
            container = QtWidgets.QFrame()
            container_layout = QtWidgets.QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(4)

            header = QtWidgets.QHBoxLayout()
            cb = QtWidgets.QCheckBox(cfg.label)
            cb.setChecked(cfg.enabled)
            cb.toggled.connect(lambda state, key=cfg.key: self._on_shape_enabled(key, state))
            header.addWidget(cb)
            header.addStretch(1)
            mask_btn = QtWidgets.QPushButton("掩膜")
            mask_btn.clicked.connect(lambda _=0, label=cfg.label: self.toggle_mask(label))
            header.addWidget(mask_btn)
            container_layout.addLayout(header)

            panel = QtWidgets.QFrame()
            panel_layout = QtWidgets.QVBoxLayout(panel)
            panel_layout.setContentsMargins(8, 4, 8, 4)
            panel_layout.setSpacing(6)
            for slider in self._build_shape_sliders(cfg):
                panel_layout.addWidget(slider)
            container_layout.addWidget(panel)

            cfg.checkbox = cb
            cfg.panel = panel
            panel.setVisible(cfg.enabled)

            group_layout.addWidget(container)

        parent_layout.addWidget(group)

    def _build_shape_sliders(self, cfg: GrayShapeConfig) -> List[ParamSlider]:
        sliders: List[ParamSlider] = []

        def clamp(value: float, mn: float, mx: float) -> float:
            return max(mn, min(value, mx))

        if cfg.key == "square":
            specs = [
                ("min_area", "最小面积", 600.0, 200000.0, float(cfg.min_area), 100.0, 0),
                ("max_area", "最大面积", 2000.0, 400000.0, float(cfg.max_area), 100.0, 0),
                ("max_aspect", "最大长宽比", 1.02, 1.5, float(cfg.max_aspect or 1.2), 0.01, 2),
            ]
        else:
            specs = [
                ("min_area", "最小面积", 800.0, 250000.0, float(cfg.min_area), 100.0, 0),
                ("max_area", "最大面积", 4000.0, 500000.0, float(cfg.max_area), 100.0, 0),
                ("min_aspect", "最小长宽比", 1.10, 3.0, float(cfg.min_aspect or 1.2), 0.01, 2),
                ("max_aspect", "最大长宽比", 1.20, 5.0, float(cfg.max_aspect or 3.5), 0.01, 2),
            ]

        for name, text, mn, mx, val, step, decimals in specs:
            slider = ParamSlider(text, mn, mx, clamp(val, mn, mx), step=step, decimals=decimals)
            slider.valueChanged.connect(lambda _value, key=cfg.key: self._on_shape_slider_changed(key))
            cfg.sliders[name] = slider
            sliders.append(slider)

        cfg.sync_from_sliders()
        if self._ensure_shape_constraints(cfg):
            cfg.sync_from_sliders()
        return sliders

    def _on_shape_enabled(self, key: str, enabled: bool) -> None:
        cfg = self.shape_cfgs.get(key)
        if not cfg:
            return
        cfg.enabled = bool(enabled)
        if cfg.panel:
            cfg.panel.setVisible(cfg.enabled)

    def _on_shape_slider_changed(self, key: str) -> None:
        cfg = self.shape_cfgs.get(key)
        if not cfg:
            return
        cfg.sync_from_sliders()
        if self._ensure_shape_constraints(cfg):
            cfg.sync_from_sliders()

    def _ensure_shape_constraints(self, cfg: GrayShapeConfig) -> bool:
        updated = False
        if cfg.max_area < cfg.min_area:
            cfg.max_area = cfg.min_area
            slider = cfg.sliders.get("max_area")
            if slider:
                slider.setValue(float(cfg.max_area))
            updated = True
        if cfg.min_aspect is not None:
            min_allowed = 1.0 if cfg.key == "square" else 1.05
            if cfg.min_aspect < min_allowed:
                cfg.min_aspect = min_allowed
                slider = cfg.sliders.get("min_aspect")
                if slider:
                    slider.setValue(float(cfg.min_aspect))
                updated = True
        if cfg.min_aspect is not None and cfg.max_aspect is not None:
            min_gap = 0.02 if cfg.key == "square" else 0.05
            if cfg.max_aspect < cfg.min_aspect + min_gap:
                cfg.max_aspect = cfg.min_aspect + min_gap
                slider = cfg.sliders.get("max_aspect")
                if slider:
                    slider.setValue(float(cfg.max_aspect))
                updated = True
        return updated

    def _sync_all_shape_cfgs(self) -> None:
        for cfg in self.shape_cfg_list:
            cfg.sync_from_sliders()
            if self._ensure_shape_constraints(cfg):
                cfg.sync_from_sliders()

    def _update_mask_windows(self, masks: Dict[str, np.ndarray]) -> None:
        self.last_masks = masks
        for label, window in list(self.mask_windows.items()):
            if window.isVisible() and label in masks:
                try:
                    window.update_mask(masks[label])
                except Exception as exc:
                    self._append_status(f"[掩膜] 更新失败 {label}: {exc}")

    def toggle_mask(self, label: str) -> None:
        window = self.mask_windows.get(label)
        if window and window.isVisible():
            window.close()
            return
        if window is None:
            window = MaskWindow(f"{label} 掩膜", self)
            self.mask_windows[label] = window
        mask = self.last_masks.get(label)
        if mask is not None and mask.size:
            try:
                window.update_mask(mask)
            except Exception as exc:
                self._append_status(f"[掩膜] 更新失败 {label}: {exc}")
        window.show()
        window.raise_()
        window.activateWindow()

    def _run_detection(self) -> List[GrayDetection]:
        if self.last_frame_bgr is None:
            return []
        frame = self.last_frame_bgr.copy()
        self._sync_all_shape_cfgs()
        detections, masks = detect_gray_shapes(
            frame,
            self.shape_cfg_list,
            merge_distance=self.merge_distance,
            angle_tolerance=self.angle_tolerance,
            approx_epsilon=self.approx_epsilon,
            contrast_alpha=self.contrast_alpha,
            contrast_beta=self.contrast_beta,
            clahe_clip_limit=self.clahe_clip,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            gradient_thresh=self.gradient_thresh,
            kmeans_enabled=self.kmeans_enabled,
            kmeans_downscale=self.kmeans_downscale,
            kmeans_blur=self.kmeans_blur,
            kmeans_morph=self.kmeans_morph,
        )
        self.last_detections = detections
        self._update_mask_windows(masks)
        return detections

    def _handle_recognition_request(self, manual: bool = False) -> None:
        model = getattr(self, "modbus_model", None)
        if model:
            if manual:
                model.set_register(0, 1)
            self._publish_modbus_result([])

        if self.last_frame_bgr is None:
            self.msg_label.setText("未检测到目标")
            self.msg_timer.start(2000)
            self._publish_modbus_result(0xFF)
            if model:
                model.set_register(0, 0)
            return

        detections = self._run_detection()
        if detections:
            summary = self._format_summary(detections)
            self.msg_label.setText(summary)
            self.msg_timer.start(2000)
            codes = [self.result_codes.get(det.label) for det in detections]
            codes = [code for code in codes if code is not None]
            if codes:
                self._publish_modbus_result(codes)
            else:
                self._publish_modbus_result(0)
        else:
            self.msg_label.setText("未检测到目标")
            self.msg_timer.start(2000)
            self._publish_modbus_result(0xFF)

        if model:
            model.set_register(0, 0)

    def _publish_modbus_result(self, values) -> None:
        model = getattr(self, "modbus_model", None)
        if not model:
            return
        if isinstance(values, int):
            values_list = [values]
        else:
            values_list = list(values)
        if not values_list:
            values_list = [0]
        sanitized = [int(v) & 0xFFFF for v in values_list]
        if self._last_result_count > len(sanitized):
            sanitized.extend([0] * (self._last_result_count - len(sanitized)))
        model.write(RESULT_BASE_ADDR, sanitized)
        self._last_result_count = len(sanitized)

    def _update_modbus_status(self) -> None:
        if self.modbus_server:
            status = f"运行中: {self.modbus_host}:{self.modbus_port}"
        elif self.modbus_error:
            status = f"未启动: {self.modbus_error}"
        else:
            status = "未启动"
        if hasattr(self, "modbus_status_lbl"):
            self.modbus_status_lbl.setText(status)

    def _refresh_modbus_ip(self) -> None:
        if not hasattr(self, "modbus_ip_combo"):
            return
        entries = list(list_local_ipv4_addresses())
        ips = [ip for ip, _ in entries]
        if self.modbus_host not in ips:
            entries.append((self.modbus_host, "当前"))
        blocker = QtCore.QSignalBlocker(self.modbus_ip_combo)
        self.modbus_ip_combo.clear()
        for ip, name in entries:
            text = f"{ip} ({name})" if name else ip
            self.modbus_ip_combo.addItem(text, ip)
        idx = self.modbus_ip_combo.findData(self.modbus_host)
        if idx < 0 and self.modbus_ip_combo.count():
            idx = 0
        if idx >= 0:
            self.modbus_ip_combo.setCurrentIndex(idx)
        self._update_modbus_status()

    def _on_modbus_ip_changed(self, index: int) -> None:
        if index < 0 or not hasattr(self, "modbus_ip_combo"):
            return
        data = self.modbus_ip_combo.itemData(index)
        host = str(data or self.modbus_ip_combo.itemText(index))
        if host and host != self.modbus_host:
            self._start_modbus_server(host)

    def _stop_modbus_server(self) -> Optional[threading.Event]:
        server = getattr(self, "modbus_server", None)
        if not server:
            return None

        done = threading.Event()

        def _shutdown():
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
            finally:
                done.set()

        threading.Thread(target=_shutdown, daemon=True).start()
        self.modbus_server = None
        return done

    def _start_modbus_server(self, host: Optional[str] = None) -> None:
        if host:
            self.modbus_host = host
        shutdown_event = self._stop_modbus_server()
        if shutdown_event:
            shutdown_event.wait(1.0)
        self.modbus_error = None
        try:
            self.modbus_server = start_modbus_server(
                self.modbus_host,
                self.modbus_port,
                self.modbus_model,
                on_write=self._on_modbus_write,
            )
        except Exception as exc:
            self.modbus_server = None
            self.modbus_error = str(exc)
        self.config.setdefault("server", {})["host"] = self.modbus_host
        self.config["server"]["port"] = self.modbus_port
        self._update_modbus_status()
        if hasattr(self, "modbus_ip_combo"):
            self._refresh_modbus_ip()

    def _on_modbus_write(self, addr: int, value: int) -> None:
        if addr == 0 and value == 1:
            self.modbus_trigger_sig.emit()

    @QtCore.pyqtSlot()
    def _on_modbus_trigger(self) -> None:
        self._handle_recognition_request(manual=False)

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
        self._sync_all_shape_cfgs()
        detections, masks = detect_gray_shapes(
            frame_draw,
            self.shape_cfg_list,
            merge_distance=self.merge_distance,
            angle_tolerance=self.angle_tolerance,
            approx_epsilon=self.approx_epsilon,
            contrast_alpha=self.contrast_alpha,
            contrast_beta=self.contrast_beta,
            clahe_clip_limit=self.clahe_clip,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            gradient_thresh=self.gradient_thresh,
            kmeans_enabled=self.kmeans_enabled,
            kmeans_downscale=self.kmeans_downscale,
            kmeans_blur=self.kmeans_blur,
            kmeans_morph=self.kmeans_morph,
        )
        self.last_detections = detections
        self._update_mask_windows(masks)

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
        self._handle_recognition_request(manual=True)

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
        if isinstance(self.grabber, HikGrabber):
            self.stop_camera()
            self.start_camera(prefer_hik=False)

    def _append_status(self, text: str) -> None:
        self._status_messages.append(text)
        print(text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI 事件
        self.stop_camera()
        if hasattr(self, "ip_refresh_timer"):
            try:
                self.ip_refresh_timer.stop()
            except Exception:
                pass
        self._stop_modbus_server()
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
