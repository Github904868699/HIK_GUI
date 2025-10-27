# -*- coding: utf-8 -*-
import os, sys
from pathlib import Path
import struct
import ctypes

from ctypes import POINTER, byref, cast, c_ubyte
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
import cv2
import json, socket, socketserver, threading, time

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - 运行环境可能缺少 psutil
    psutil = None  # type: ignore

CONFIG_PATH = "config.json"
TARGET_DISPLAY_WIDTH = 1280
UI_TARGET_FPS = 15.0
UI_PAINT_FPS = 12.0
RESULT_BASE_ADDR = 1  # 对应保持寄存器 40002

APP_TITLE = "HIK MVS"
APP_ICON  = "Camera.ico"
CHS = {"circle": "圆形", "triangle": "三角形", "rect": "正方形"}
MIN_AREA, MAX_AREA = 500, 300_000
FPS_CALC_INTERVAL  = 30

try:
    from MvCameraControl_class import MvCamera
    from CameraParams_header import (
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_FRAME_OUT_INFO_EX,
        MVCC_INTVALUE,
        MV_CC_PIXEL_CONVERT_PARAM,
    )
    from CameraParams_const import MV_GIGE_DEVICE, MV_ACCESS_Exclusive
    from PixelType_header import (
        PixelType_Gvsp_BGR8_Packed,
        PixelType_Gvsp_RGB8_Packed,
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_BayerRG8,
        PixelType_Gvsp_BayerBG8,
        PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_YUV422_Packed,
        PixelType_Gvsp_YUV422_YUYV_Packed,
    )
    from MvErrorDefine_const import MV_OK
    HIK_SDK_AVAILABLE = True
    HIK_SDK_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - 平台可能缺少 SDK
    MvCamera = None  # type: ignore[assignment]
    MV_CC_DEVICE_INFO = None  # type: ignore[assignment]
    MV_CC_DEVICE_INFO_LIST = None  # type: ignore[assignment]
    MV_FRAME_OUT_INFO_EX = None  # type: ignore[assignment]
    MVCC_INTVALUE = None  # type: ignore[assignment]
    MV_CC_PIXEL_CONVERT_PARAM = None  # type: ignore[assignment]
    MV_GIGE_DEVICE = 0  # type: ignore[assignment]
    MV_ACCESS_Exclusive = 1  # type: ignore[assignment]
    PixelType_Gvsp_BGR8_Packed = 0  # type: ignore[assignment]
    PixelType_Gvsp_RGB8_Packed = 0  # type: ignore[assignment]
    PixelType_Gvsp_Mono8 = 0  # type: ignore[assignment]
    PixelType_Gvsp_BayerRG8 = 0  # type: ignore[assignment]
    PixelType_Gvsp_BayerBG8 = 0  # type: ignore[assignment]
    PixelType_Gvsp_BayerGB8 = 0  # type: ignore[assignment]
    PixelType_Gvsp_BayerGR8 = 0  # type: ignore[assignment]
    PixelType_Gvsp_YUV422_Packed = 0  # type: ignore[assignment]
    PixelType_Gvsp_YUV422_YUYV_Packed = 0  # type: ignore[assignment]
    MV_OK = 0  # type: ignore[assignment]
    HIK_SDK_AVAILABLE = False
    HIK_SDK_IMPORT_ERROR = exc

def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", Path(__file__).parent)
    return str(Path(base, rel))

def safe_load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def rounded_qpixmap(pix: QtGui.QPixmap, radius: int = 18) -> QtGui.QPixmap:
    if pix.isNull():
        return pix
    w, h = pix.width(), pix.height()
    out = QtGui.QPixmap(w, h)
    out.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(out)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    path = QtGui.QPainterPath()
    path.addRoundedRect(QtCore.QRectF(0, 0, w, h), radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pix)
    painter.end()
    return out

def load_config(path: str = CONFIG_PATH) -> dict:
    cfg = safe_load_json(path, default=None)
    if not cfg:
        cfg = {
            "server": {"host": "0.0.0.0", "port": 502},
            "cmd_map": {
            },
            "colors": [

            ]
        }
    if isinstance(cfg, list):
        cfg = {"server": {"host":"0.0.0.0","port":502}, "cmd_map": {}, "colors": cfg}
    cfg.setdefault("server", {"host": "0.0.0.0", "port": 502})
    cfg.setdefault("cmd_map", {})
    cfg.setdefault("colors", [])
    return cfg


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def list_local_ipv4_addresses() -> List[Tuple[str, str]]:
    addresses: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    if psutil is not None:
        try:
            for iface, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        ip = addr.address
                        if ip and not ip.startswith("127.") and ip not in seen:
                            addresses.append((ip, iface))
                            seen.add(ip)
        except Exception:
            pass

    hostname = socket.gethostname()
    try:
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if ip and not ip.startswith("127.") and ip not in seen:
                addresses.append((ip, hostname))
                seen.add(ip)
    except Exception:
        pass

    if not addresses:
        ip = get_local_ip()
        addresses.append((ip, ""))

    return addresses

@dataclass
class ColorCfg:
    name: str
    bgr: Tuple[int, int, int]
    lower: np.ndarray
    upper: np.ndarray
    sliders: Dict[str, "HSVSlider"] = field(default_factory=dict)
    shape_checks: Dict[str, QtWidgets.QCheckBox] = field(default_factory=dict)
    shapes_init: Dict[str, bool] = field(default_factory=dict)
    @property
    def group_title(self) -> str:
        return f"HSV {self.name}"
    @property
    def mask_button_title(self) -> str:
        return f"{self.name} 掩膜"

def colors_from_config(cfg: dict) -> List[ColorCfg]:
    data = cfg.get("colors", [])
    colors: List[ColorCfg] = []
    if not data:
        return colors
    for item in data:
        shapes = item.get("shapes", {})
        colors.append(ColorCfg(
            item["name"],
            tuple(item["bgr"]),
            np.array(item["lower"], dtype=np.uint8),
            np.array(item["upper"], dtype=np.uint8),
            shapes_init={
                "circle": bool(shapes.get("circle", True)),
                "triangle": bool(shapes.get("triangle", True)),
                "rect": bool(shapes.get("rect", True)),
            }
        ))
    return colors

def _parse_cmd_code(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped, 0)
        except ValueError:
            return None
    return None


def result_codes_from_cmd_map(cmd_map: dict) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for label, value in cmd_map.items():
        code = _parse_cmd_code(value)
        if code is not None:
            out[label] = code
            continue
        parsed = None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                try:
                    parsed = json.loads(stripped)
                except Exception:
                    print(f"[MODBUS] 忽略无法解析的 cmd_map 项: {label}")
                    continue
        elif isinstance(value, dict):
            parsed = value
        if isinstance(parsed, dict):
            code = _parse_cmd_code(parsed.get("code"))
            if code is not None:
                out[label] = code
            else:
                print(f"[MODBUS] cmd_map 项 {label} 缺少 'code' 数值，已忽略")
        elif value is not None:
            print(f"[MODBUS] cmd_map 项 {label} 类型不支持，已忽略")
    return out

class HSVSlider(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(int)
    def __init__(self, text: str, mn: int, mx: int, val: int, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.label = QtWidgets.QLabel(text)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(mn, mx); self.slider.setValue(val)
        self.val_lbl = QtWidgets.QLabel(str(val)); self.val_lbl.setFixedWidth(40)
        lay.addWidget(self.label); lay.addWidget(self.slider); lay.addWidget(self.val_lbl)
        self.slider.valueChanged.connect(self._on_change)
    def _on_change(self, v: int):
        self.val_lbl.setText(str(v)); self.valueChanged.emit(v)
    def value(self) -> int:  return self.slider.value()

class MaskWindow(QtWidgets.QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle(title); self.resize(420, 320)
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        lay = QtWidgets.QVBoxLayout(self); lay.addWidget(self.label)
    def update_mask(self, mask_np: np.ndarray):
        if mask_np is None or mask_np.size == 0: return
        h, w = mask_np.shape
        qimg = QtGui.QImage(mask_np.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ); self.label.setPixmap(pix)

# TCP
class ModbusRegisterModel:
    def __init__(self, size: int = 16):
        self._lock = threading.Lock()
        self._regs = [0] * max(size, 2)

    def read(self, addr: int, count: int) -> List[int]:
        with self._lock:
            if addr < 0:
                return [0] * max(count, 0)
            end = addr + count
            slice_regs = self._regs[addr:end]
            if len(slice_regs) < count:
                slice_regs.extend([0] * (count - len(slice_regs)))
            return list(slice_regs)

    def write(self, addr: int, values: List[int]):
        if addr < 0:
            return
        with self._lock:
            end = addr + len(values)
            if end > len(self._regs):
                self._regs.extend([0] * (end - len(self._regs)))
            for i, v in enumerate(values):
                self._regs[addr + i] = v & 0xFFFF

    def set_register(self, addr: int, value: int):
        self.write(addr, [value])


class ModbusRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            header = self._recvn(7)
            if not header:
                break
            try:
                tid, pid, length = struct.unpack(">HHH", header[:6])
            except struct.error:
                break
            unit = header[6]
            if length <= 0:
                continue
            payload = self._recvn(length - 1)
            if payload is None:
                break
            if not payload:
                continue
            function = payload[0]
            data = payload[1:]
            response_pdu = self._handle_function(function, data)
            if response_pdu is None:
                continue
            mbap = struct.pack(">HHHB", tid, 0, len(response_pdu) + 1, unit)
            try:
                self.request.sendall(mbap + response_pdu)
            except Exception:
                break

    def _recvn(self, size: int):
        buf = b""
        while len(buf) < size:
            chunk = self.request.recv(size - len(buf))
            if not chunk:
                return None if not buf else buf
            buf += chunk
        return buf

    def _handle_function(self, function: int, data: bytes) -> bytes | None:
        try:
            if function == 3:  # Read Holding Registers
                if len(data) < 4:
                    raise ValueError
                addr, count = struct.unpack(">HH", data[:4])
                regs = self.server.model.read(addr, count)
                payload = struct.pack(">B", len(regs) * 2)
                if regs:
                    payload += struct.pack(">" + "H" * len(regs), *regs)
                return bytes([function]) + payload
            elif function == 6:  # Write Single Register
                if len(data) < 4:
                    raise ValueError
                addr, value = struct.unpack(">HH", data[:4])
                self.server.model.write(addr, [value])
                if self.server.on_write:
                    self.server.on_write(addr, value & 0xFFFF)
                return bytes([function]) + data[:4]
            elif function == 16:  # Write Multiple Registers
                if len(data) < 5:
                    raise ValueError
                addr, count, byte_count = struct.unpack(">HHB", data[:5])
                expected = count * 2
                if byte_count != expected or len(data[5:]) < expected:
                    raise ValueError
                raw = data[5:5 + expected]
                values = list(struct.unpack(">" + "H" * count, raw))
                self.server.model.write(addr, values)
                if self.server.on_write:
                    for i, v in enumerate(values):
                        self.server.on_write(addr + i, v & 0xFFFF)
                return bytes([function]) + struct.pack(">HH", addr, count)
            else:
                return bytes([function | 0x80, 1])
        except Exception:
            return bytes([function | 0x80, 3])


class ModbusTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, host: str, port: int, model: ModbusRegisterModel, on_write=None):
        self.model = model
        self.on_write = on_write
        super().__init__((host, port), ModbusRequestHandler)


def start_modbus_server(host: str, port: int, model: ModbusRegisterModel, on_write=None):
    server = ModbusTCPServer(host, port, model, on_write=on_write)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[MODBUS] listen on {host}:{port}")
    return server

def _side_lengths(pts: np.ndarray):
    pts = pts.reshape(-1, 2)
    return [np.linalg.norm(pts[(i+1)%len(pts)]-pts[i]) for i in range(len(pts))]

def classify_contour(cnt, circularity: float):
    peri = cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    verts = len(poly)
    if verts == 3:
        sides = _side_lengths(poly)
        if max(sides)/(min(sides)+1e-5) <= 1.20:
            return "triangle"
    elif verts == 4:
        x,y,w,h = cv2.boundingRect(poly)
        ar = w / float(h+1e-5)
        if 0.85 <= ar <= 1.15:
            pts = poly.reshape(-1,2)
            v1 = pts[1]-pts[0]; v2=pts[2]-pts[1]
            cosang = abs(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-5))
            if cosang <= 0.15: return "rect"
    elif circularity > 0.75:
        return "circle"
    return None

def detect_gold_circle_robust(bgr, minR=60, maxR=180):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=gray.shape[0]//3, param1=120, param2=40,
                               minRadius=minR, maxRadius=maxR)
    if circles is None: return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    H, S, V = cv2.split(hsv)
    a = lab[:, :, 1]; b = lab[:, :, 2]
    out = []
    for x, y, r in np.round(circles[0]).astype(int):
        mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        spec = (V > 240) & (S < 60)
        valid = (mask > 0) & (~spec)
        if np.count_nonzero(valid) < 800: continue
        b_mean = float(b[valid].mean())
        a_mean = float(np.abs(a[valid].mean()))
        if b_mean > 145 and (b_mean - a_mean) > 20: 
            out.append((x, y, r, b_mean))
    return out

def detect_shapes(frame_bgr: np.ndarray, color_cfgs: List['ColorCfg'], enabled_global: set):
    labels = []
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    for cfg in color_cfgs:
        if not cfg.sliders: continue
        mask = cv2.inRange(hsv, cfg.lower, cfg.upper)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if getattr(cfg, "shape_checks", None):
            enabled_local = {k for k, cb in cfg.shape_checks.items() if cb.isChecked()}
        else:
            enabled_local = enabled_global
        for c in cnts:
            area = cv2.contourArea(c)
            if not (MIN_AREA < area < MAX_AREA): continue
            peri = cv2.arcLength(c, True)
            if peri <= 0: continue
            circ = 4*np.pi*area/(peri*peri+1e-9)
            shp = classify_contour(c, circ)
            if not (shp and shp in enabled_local): continue
            label_txt = f"{cfg.name}-{CHS[shp]}"
            qcolor = QtGui.QColor(*reversed(cfg.bgr))
            if shp == "circle":
                (x,y), r = cv2.minEnclosingCircle(c)
                cv2.circle(frame_bgr, (int(x),int(y)), int(r), cfg.bgr, 2)
                tpos = (int(x-r), int(y-r-6))
            else:
                poly = cv2.approxPolyDP(c, 0.04*peri, True)
                cv2.polylines(frame_bgr, [poly], True, cfg.bgr, 2)
                bx,by,_,_ = cv2.boundingRect(poly)
                tpos = (bx, by-6)
            labels.append((label_txt, tpos, qcolor))
    return labels

class HikGrabber(QtCore.QThread):
    frameSignal = QtCore.pyqtSignal(np.ndarray)
    infoSignal = QtCore.pyqtSignal(str)
    errorSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        if not HIK_SDK_AVAILABLE or MvCamera is None:
            reason = str(HIK_SDK_IMPORT_ERROR) if HIK_SDK_IMPORT_ERROR else "未检测到海康 SDK"
            raise RuntimeError(f"海康 SDK 未就绪: {reason}")
        self.camera: Optional['MvCamera'] = None
        self._running = False
        self._data_buf = None
        self._data_ptr = None
        self._convert_buf = None
        self._convert_ptr = None
        self._convert_buf_size = 0
        self._payload_size = 0
        self._last_emit_ts = 0.0
        self._local_ip_int = self._ip_to_uint(get_local_ip())
        self._bayer_types = {
            PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerBG8,
            PixelType_Gvsp_BayerGB8,
            PixelType_Gvsp_BayerGR8,
        }
        self._last_stream_error = 0
        self._last_convert_error = 0
        self._last_unsupported_pixel = 0

    @staticmethod
    def _ip_to_uint(ip: str) -> Optional[int]:
        try:
            return struct.unpack(">I", socket.inet_aton(ip))[0]
        except Exception:
            return None

    @staticmethod
    def _uint_to_ip(value: int) -> str:
        try:
            return socket.inet_ntoa(struct.pack(">I", value))
        except Exception:
            return ""

    @staticmethod
    def _decode_text(buf) -> str:
        try:
            raw = bytes(bytearray(buf))
        except Exception:
            return ""
        raw = raw.split(b"\0", 1)[0]
        try:
            return raw.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def _is_same_lan(self, info) -> bool:
        if not self._local_ip_int:
            return False
        try:
            gige = info.SpecialInfo.stGigEInfo
            cam_ip = int(gige.nCurrentIp)
            mask = int(gige.nCurrentSubNetMask) or 0xFFFFFFFF
            return (self._local_ip_int & mask) == (cam_ip & mask)
        except Exception:
            return False

    def _format_device_name(self, info) -> str:
        try:
            gige = info.SpecialInfo.stGigEInfo
            name = self._decode_text(gige.chUserDefinedName) or self._decode_text(gige.chModelName)
            ip = self._uint_to_ip(int(gige.nCurrentIp))
            if name and ip:
                return f"{name} ({ip})"
            if ip:
                return f"Hik GIGE ({ip})"
            return name or "Hik GIGE"
        except Exception:
            return "Hik GIGE"

    def _select_device(self):
        dev_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, dev_list)
        if ret != MV_OK:
            raise RuntimeError(f"枚举海康相机失败: 0x{ret:08X}")
        if dev_list.nDeviceNum == 0:
            return None
        accessible_candidates = []
        for idx in range(int(dev_list.nDeviceNum)):
            ptr = dev_list.pDeviceInfo[idx]
            if not ptr:
                continue
            info_copy = MV_CC_DEVICE_INFO()
            ctypes.memmove(
                byref(info_copy),
                ctypes.byref(ptr.contents),
                ctypes.sizeof(MV_CC_DEVICE_INFO),
            )
            if not MvCamera.MV_CC_IsDeviceAccessible(info_copy, MV_ACCESS_Exclusive):
                continue
            display = self._format_device_name(info_copy)
            if self._is_same_lan(info_copy):
                return info_copy, display
            accessible_candidates.append((info_copy, display))
        if accessible_candidates:
            return accessible_candidates[0]
        return None

    def _prepare_payload(self):
        payload = MVCC_INTVALUE()
        ret = self.camera.MV_CC_GetIntValue("PayloadSize", payload)
        if ret != MV_OK or int(payload.nCurValue) <= 0:
            raise RuntimeError(f"获取 PayloadSize 失败: 0x{ret:08X}")
        self._payload_size = int(payload.nCurValue)
        self._data_buf = (c_ubyte * self._payload_size)()
        self._data_ptr = cast(self._data_buf, POINTER(c_ubyte))

    def _get_int_value(self, key: str) -> Optional[int]:
        if not self.camera:
            return None
        value = MVCC_INTVALUE()
        ret = self.camera.MV_CC_GetIntValue(key, value)
        if ret == MV_OK:
            return int(value.nCurValue)
        return None

    def _ensure_convert_buffer(self, size: int):
        if self._convert_buf_size < size:
            self._convert_buf = (c_ubyte * size)()
            self._convert_ptr = cast(self._convert_buf, POINTER(c_ubyte))
            self._convert_buf_size = size

    def _convert_frame(self, frame_info: 'MV_FRAME_OUT_INFO_EX') -> Optional[np.ndarray]:
        width = int(frame_info.nWidth)
        height = int(frame_info.nHeight)
        frame_len = int(frame_info.nFrameLen)
        pixel_type = int(frame_info.enPixelType)
        if frame_len <= 0 or width <= 0 or height <= 0:
            return None
        if pixel_type == PixelType_Gvsp_BGR8_Packed:
            arr = np.frombuffer(self._data_buf, dtype=np.uint8, count=frame_len)
            return arr.reshape(height, width, 3).copy()
        if pixel_type == PixelType_Gvsp_RGB8_Packed:
            arr = np.frombuffer(self._data_buf, dtype=np.uint8, count=frame_len)
            rgb = arr.reshape(height, width, 3)
            return rgb[:, :, ::-1].copy()
        if pixel_type == PixelType_Gvsp_Mono8:
            arr = np.frombuffer(self._data_buf, dtype=np.uint8, count=frame_len)
            gray = arr.reshape(height, width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if pixel_type in self._bayer_types or pixel_type in {PixelType_Gvsp_YUV422_Packed, PixelType_Gvsp_YUV422_YUYV_Packed}:
            dst_size = width * height * 3
            self._ensure_convert_buffer(dst_size)
            convert_param = MV_CC_PIXEL_CONVERT_PARAM()
            convert_param.nWidth = width
            convert_param.nHeight = height
            convert_param.enSrcPixelType = pixel_type
            convert_param.pSrcData = self._data_ptr
            convert_param.nSrcDataLen = frame_len
            convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            convert_param.pDstBuffer = self._convert_ptr
            convert_param.nDstBufferSize = dst_size
            convert_param.nDstLen = dst_size
            ret = self.camera.MV_CC_ConvertPixelType(convert_param)
            if ret != MV_OK:
                if self._last_convert_error != ret:
                    self.infoSignal.emit(f"[HIK] 像素转换失败: 0x{ret:08X}")
                    self._last_convert_error = ret
                return None
            self._last_convert_error = 0
            arr = np.frombuffer(self._convert_buf, dtype=np.uint8, count=dst_size)
            return arr.reshape(height, width, 3).copy()
        if self._last_unsupported_pixel != pixel_type:
            self.infoSignal.emit(f"[HIK] 不支持的像素格式: 0x{pixel_type:08X}")
            self._last_unsupported_pixel = pixel_type
        return None

    def _cleanup_camera(self):
        if self.camera:
            try:
                self.camera.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                self.camera.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                self.camera.MV_CC_DestroyHandle()
            except Exception:
                pass
            self.camera = None
        self._data_buf = None
        self._data_ptr = None
        self._convert_buf = None
        self._convert_ptr = None
        self._convert_buf_size = 0

    def stop(self):
        self._running = False

    def run(self):
        initialized = False
        try:
            ret = MvCamera.MV_CC_Initialize()
            if ret != MV_OK:
                raise RuntimeError(f"初始化海康 SDK 失败: 0x{ret:08X}")
            initialized = True
            selection = self._select_device()
            if not selection:
                raise RuntimeError("未发现可用的海康相机")
            device_info, display_name = selection
            self.camera = MvCamera()
            ret = self.camera.MV_CC_CreateHandle(device_info)
            if ret != MV_OK:
                raise RuntimeError(f"创建相机句柄失败: 0x{ret:08X}")
            ret = self.camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != MV_OK:
                raise RuntimeError(f"打开相机失败: 0x{ret:08X}")
            self._prepare_payload()
            ret = self.camera.MV_CC_StartGrabbing()
            if ret != MV_OK:
                raise RuntimeError(f"启动取流失败: 0x{ret:08X}")
            width = self._get_int_value("Width")
            height = self._get_int_value("Height")
            if width and height:
                self.infoSignal.emit(f"[INFO] {display_name} {width}x{height}")
            else:
                self.infoSignal.emit(f"[INFO] {display_name}")
            self._running = True
            frame_info = MV_FRAME_OUT_INFO_EX()
            grabbed = 0
            last_fps_ts = time.time()
            self._last_emit_ts = 0.0
            while self._running:
                ret = self.camera.MV_CC_GetOneFrameTimeout(self._data_ptr, self._payload_size, frame_info, 1000)
                if ret != MV_OK:
                    if ret != self._last_stream_error:
                        self.infoSignal.emit(f"[HIK] 取流异常: 0x{ret:08X}")
                        self._last_stream_error = ret
                    continue
                self._last_stream_error = 0
                now = time.time()
                grabbed += 1
                if (now - self._last_emit_ts) < (1.0 / UI_TARGET_FPS):
                    if (now - last_fps_ts) >= 1.0:
                        fps = grabbed / (now - last_fps_ts)
                        self.infoSignal.emit(f"[FPS] {fps:.1f}")
                        grabbed = 0
                        last_fps_ts = now
                    continue
                frame = self._convert_frame(frame_info)
                if frame is None:
                    continue
                self._last_emit_ts = now
                self.frameSignal.emit(frame)
                if (now - last_fps_ts) >= 1.0:
                    fps = grabbed / (now - last_fps_ts)
                    self.infoSignal.emit(f"[FPS] {fps:.1f}")
                    grabbed = 0
                    last_fps_ts = now
                QtCore.QThread.msleep(1)
        except Exception as exc:
            self._running = False
            self.infoSignal.emit(f"[HIK] {exc}")
            self.errorSignal.emit(str(exc))
        finally:
            self._cleanup_camera()
            if initialized:
                try:
                    MvCamera.MV_CC_Finalize()
                except Exception:
                    pass


class UsbGrabber(QtCore.QThread):
    frameSignal = QtCore.pyqtSignal(np.ndarray)
    infoSignal  = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, index: int = 0):
        super().__init__(parent)
        self.index = index
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._last_emit_ts = 0.0

    def run(self):
        try:
            if os.name == "nt":
                backend = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
            else:
                backend = cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.index, backend)
            if (not self.cap or not self.cap.isOpened()) and backend != cv2.CAP_ANY:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = cv2.VideoCapture(self.index)
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("无法打开 USB 摄像头")

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width > 0 and height > 0:
                self.infoSignal.emit(f"[INFO] {width}x{height}")
            else:
                self.infoSignal.emit("[INFO] USB Camera")

            self._running = True
            t0 = time.time()
            grabbed = 0
            while self._running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    QtCore.QThread.msleep(5)
                    continue
                grabbed += 1
                if frame.shape[1] > TARGET_DISPLAY_WIDTH:
                    scale = TARGET_DISPLAY_WIDTH / float(frame.shape[1])
                    frame = cv2.resize(frame, (TARGET_DISPLAY_WIDTH, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

                now_ts = time.time()
                if (now_ts - self._last_emit_ts) < (1.0 / UI_TARGET_FPS):
                    QtCore.QThread.msleep(5)
                    continue

                self._last_emit_ts = now_ts
                self.frameSignal.emit(frame.copy())

                now = time.time()
                if now - t0 >= 1.0:
                    self.infoSignal.emit(f"[FPS] {grabbed / (now - t0):.1f}")
                    t0 = now
                    grabbed = 0

                QtCore.QThread.msleep(1)
        except Exception as e:
            self.infoSignal.emit(f"[USB] 取流异常: {e}")
        finally:
            self._stop_and_close()

    def stop(self):
        self._running = False

    def _stop_and_close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

class MainWindow(QtWidgets.QWidget):
    modbus_trigger_sig = QtCore.pyqtSignal()
    def __init__(self, config: dict):
        super().__init__(None, QtCore.Qt.Window)
        self.setWindowTitle(APP_TITLE); self.resize(1200, 720)
        self.config = config
        self.result_codes = result_codes_from_cmd_map(self.config.get("cmd_map", {}))
        self.colors = {c.name: c for c in colors_from_config(self.config)}
        self.mask_windows: Dict[str, MaskWindow] = {}
        self.frame_cnt = 0
        self.fps = 0.0
        self.last_time = time.time()
        self.last_frame_bgr: np.ndarray | None = None
        self._last_paint_ts = 0.0

        svr = self.config.get("server", {})
        self.modbus_host = str(svr.get("host", "0.0.0.0") or "0.0.0.0")
        self.modbus_port = int(svr.get("port", 502))
        self.modbus_model = ModbusRegisterModel(size=16)
        self.modbus_server = None
        self.modbus_error: str | None = None
        self._last_result_count = 0
        self._start_modbus_server(self.modbus_host)
        self.modbus_trigger_sig.connect(self._on_modbus_trigger)

        hbox = QtWidgets.QHBoxLayout(self)
        self.ctrl_panel = QtWidgets.QFrame(); self.ctrl_panel.setFixedWidth(340)
        hbox.addWidget(self.ctrl_panel)
        right_panel = QtWidgets.QVBoxLayout()
        header = QtWidgets.QHBoxLayout()
        self.lbl_cam = QtWidgets.QLabel("Camera —"); self.lbl_fps = QtWidgets.QLabel("FPS —")
        header.addWidget(self.lbl_cam, 1, QtCore.Qt.AlignLeft); header.addWidget(self.lbl_fps, 0, QtCore.Qt.AlignRight)
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
        self.msg_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        msg_layout.addWidget(self.msg_label)

        self.msg_label.setAlignment(QtCore.Qt.AlignCenter)
        self.msg_frame.setStyleSheet("""
        #msgBar {
            background: rgba(255, 255, 255, 140);
            border-radius: 12px;
        }
        #msgBar QLabel {
            color: #1a1a1a;
            font-family: "Microsoft YaHei", "Segoe UI", "PingFang SC";
            font-size: 16px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        """)

        shadow = QtWidgets.QGraphicsDropShadowEffect(self.msg_frame)
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 4)
        shadow.setColor(QtGui.QColor(0, 0, 0, 160))
        self.msg_frame.setGraphicsEffect(shadow)

        right_panel.addWidget(self.msg_frame)
        w = QtWidgets.QWidget(); w.setLayout(right_panel)
        hbox.addWidget(w, 1)

        self.msg_timer = QtCore.QTimer(self); self.msg_timer.setSingleShot(True)
        self.msg_timer.timeout.connect(lambda: self.msg_label.setText(""))

        self._init_controls()
        self._update_modbus_status()

        self.grabber: Optional[QtCore.QThread] = None
        self._start_camera()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(30)

    def _init_controls(self):
        vbox = QtWidgets.QVBoxLayout(self.ctrl_panel); vbox.setAlignment(QtCore.Qt.AlignTop)
        # 相机
        g_cam = QtWidgets.QGroupBox("相机"); lay = QtWidgets.QHBoxLayout(g_cam)
        self.btn_reopen = QtWidgets.QPushButton("重连")
        self.btn_stop   = QtWidgets.QPushButton("停止")
        self.btn_reopen.clicked.connect(self.reopen_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        lay.addWidget(self.btn_reopen); lay.addWidget(self.btn_stop); vbox.addWidget(g_cam)

        for cfg in self.colors.values():
            self._add_color_group(vbox, cfg)

        g_modbus = QtWidgets.QGroupBox("Modbus")
        form = QtWidgets.QFormLayout(g_modbus)
        self.modbus_ip_combo = QtWidgets.QComboBox()
        self.modbus_ip_combo.currentIndexChanged.connect(self._on_modbus_ip_changed)
        self.modbus_port_label = QtWidgets.QLabel(str(self.modbus_port))
        self.modbus_status_lbl = QtWidgets.QLabel("")
        form.addRow("服务器IP:", self.modbus_ip_combo)
        form.addRow("端口:", self.modbus_port_label)
        form.addRow("状态:", self.modbus_status_lbl)
        self.recognize_btn = QtWidgets.QPushButton("手动识别")
        self.recognize_btn.clicked.connect(self.trigger_manual_recognition)
        form.addRow(self.recognize_btn)
        vbox.addWidget(g_modbus)
        vbox.addStretch(1)

        self._ip_choices: List[str] = []
        self._refresh_modbus_ip()
        self.ip_refresh_timer = QtCore.QTimer(self)
        self.ip_refresh_timer.setInterval(2000)
        self.ip_refresh_timer.timeout.connect(self._refresh_modbus_ip)
        self.ip_refresh_timer.start()

    def _start_camera(self, prefer_hik: bool = True):
        self.stop_camera()
        if prefer_hik:
            try:
                grabber = HikGrabber(self)
            except Exception as exc:
                print(f"[HIK] {exc}")
            else:
                self.grabber = grabber
                grabber.frameSignal.connect(self.on_frame_from_hik)
                grabber.infoSignal.connect(self.on_info)
                grabber.errorSignal.connect(self._on_hik_error)
                grabber.start()
                return
        self.grabber = UsbGrabber(self)
        self.grabber.frameSignal.connect(self.on_frame_from_hik)
        self.grabber.infoSignal.connect(self.on_info)
        self.grabber.start()

    def _update_modbus_status(self):
        if self.modbus_server:
            status = "运行"
        elif getattr(self, "modbus_error", None):
            status = f"未启动: {self.modbus_error}"
        else:
            status = "未启动"
        if hasattr(self, "modbus_status_lbl"):
            self.modbus_status_lbl.setText(status)

    def _refresh_modbus_ip(self):
        if not hasattr(self, "modbus_ip_combo"):
            return

        entries = list_local_ipv4_addresses()
        ips = [ip for ip, _ in entries]

        if "0.0.0.0" not in ips:
            ips.insert(0, "0.0.0.0")
            entries.insert(0, ("0.0.0.0", "全部网口"))

        if self.modbus_host not in ips:
            entries.append((self.modbus_host, "当前"))
            ips.append(self.modbus_host)

        if ips != self._ip_choices:
            blocker = QtCore.QSignalBlocker(self.modbus_ip_combo)
            self.modbus_ip_combo.clear()
            for ip, iface in entries:
                if ip == "0.0.0.0":
                    text = f"{ip} (全部网口)"
                elif iface:
                    text = f"{ip} ({iface})"
                else:
                    text = ip
                self.modbus_ip_combo.addItem(text, ip)
            self._ip_choices = ips
            del blocker

        current_idx = self.modbus_ip_combo.findData(self.modbus_host)
        if current_idx < 0:
            current_idx = 0
        if self.modbus_ip_combo.currentIndex() != current_idx:
            blocker = QtCore.QSignalBlocker(self.modbus_ip_combo)
            self.modbus_ip_combo.setCurrentIndex(current_idx)
            del blocker

    def _on_modbus_ip_changed(self, index: int):
        if index < 0:
            return
        data = self.modbus_ip_combo.itemData(index)
        if not data:
            data = self.modbus_ip_combo.itemText(index)
        host = str(data)
        if host and host != self.modbus_host:
            self._start_modbus_server(host)

    def _stop_modbus_server(self):
        if not getattr(self, "modbus_server", None):
            return
        try:
            self.modbus_server.shutdown()
        except Exception:
            pass
        try:
            self.modbus_server.server_close()
        except Exception:
            pass
        self.modbus_server = None

    def _start_modbus_server(self, host: Optional[str] = None):
        if host is not None:
            self.modbus_host = host

        self._stop_modbus_server()
        self.modbus_error = None

        try:
            self.modbus_server = start_modbus_server(
                self.modbus_host, self.modbus_port, self.modbus_model, on_write=self._on_modbus_write
            )
        except Exception as exc:
            print(f"[MODBUS] 启动失败: {exc}")
            self.modbus_server = None
            self.modbus_error = str(exc)

        self.config.setdefault("server", {})
        self.config["server"]["host"] = self.modbus_host
        self.config["server"]["port"] = self.modbus_port

        self._update_modbus_status()
        if hasattr(self, "modbus_ip_combo"):
            self._refresh_modbus_ip()

    def _add_color_group(self, parent_layout, cfg: ColorCfg):
        g = QtWidgets.QGroupBox(cfg.group_title); g.setCheckable(True); g.setChecked(False); g.setFlat(True)
        container = QtWidgets.QWidget(); inner = QtWidgets.QVBoxLayout(container); inner.setContentsMargins(0,0,0,0)
        g.setLayout(QtWidgets.QVBoxLayout()); g.layout().addWidget(container)
        g.toggled.connect(container.setVisible); container.setVisible(False)
        labels = ["H", "S", "V"]; ranges = [(0,179),(0,255),(0,255)]
        for i, ch in enumerate(labels):
            mn, mx = ranges[i]
            key_min = f"{cfg.name}_min{ch}"; key_max = f"{cfg.name}_max{ch}"
            init_min = int(cfg.lower[i]); init_max = int(cfg.upper[i])
            for key, val in ((key_min, init_min),(key_max, init_max)):
                s = HSVSlider(key, mn, mx, val)
                s.valueChanged.connect(lambda _v, c=cfg: self._sync_cfg_from_sliders(c))
                inner.addWidget(s); cfg.sliders[key] = s

        btn = QtWidgets.QPushButton(cfg.mask_button_title)
        btn.clicked.connect(lambda _=0, n=cfg.name: self.toggle_mask(n))
        inner.addWidget(btn)
        # 每色的形状开关（按 JSON 默认勾选）
        shape_box = QtWidgets.QGroupBox("")
        shape_lay = QtWidgets.QHBoxLayout(shape_box); shape_lay.setContentsMargins(6,4,6,4)
        for key, text in (("circle","圆形"), ("triangle","三角形"), ("rect","正方形")):
            cb = QtWidgets.QCheckBox(text)
            cb.setChecked(bool(cfg.shapes_init.get(key, True)))
            shape_lay.addWidget(cb)
            cfg.shape_checks[key] = cb
        inner.addWidget(shape_box)
        parent_layout.addWidget(g)

    def _sync_cfg_from_sliders(self, cfg: ColorCfg):
        lh = cfg.sliders[f"{cfg.name}_minH"].value(); uh = cfg.sliders[f"{cfg.name}_maxH"].value()
        ls = cfg.sliders[f"{cfg.name}_minS"].value(); us = cfg.sliders[f"{cfg.name}_maxS"].value()
        lv = cfg.sliders[f"{cfg.name}_minV"].value(); uv = cfg.sliders[f"{cfg.name}_maxV"].value()
        cfg.lower[:] = [lh, ls, lv]; cfg.upper[:] = [uh, us, uv]

    @QtCore.pyqtSlot(str)
    def _on_hik_error(self, message: str):
        print(f"[HIK] {message}")
        if isinstance(getattr(self, "grabber", None), HikGrabber):
            self.stop_camera()
            self._start_camera(prefer_hik=False)

    def reopen_camera(self):
        self._start_camera(prefer_hik=True)

    def stop_camera(self):
        grabber = getattr(self, "grabber", None)
        if grabber and grabber.isRunning():
            grabber.stop(); grabber.wait(1000)
        self.grabber = None

    def _on_modbus_write(self, addr: int, value: int):
        if addr == 0 and value == 1:
            print("[MODBUS] 收到拍照请求")
            self.modbus_trigger_sig.emit()

    def _handle_recognition_request(self, manual: bool = False):
        model = getattr(self, "modbus_model", None)
        if model:
            if manual:
                model.set_register(0, 1)
            self._publish_modbus_result([])
        self.recognize_once()
        if model:
            model.set_register(0, 0)

    @QtCore.pyqtSlot()
    def _on_modbus_trigger(self):
        self._handle_recognition_request(manual=False)

    @QtCore.pyqtSlot()
    def trigger_manual_recognition(self):
        self._handle_recognition_request(manual=True)

    def _publish_modbus_result(self, values):
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

    @QtCore.pyqtSlot(np.ndarray)
    def on_frame_from_hik(self, frame_bgr: np.ndarray):
        t = time.time()
        if (t - self._last_paint_ts) < (1.0 / UI_PAINT_FPS):
            return
        self._last_paint_ts = t
        self.last_frame_bgr = frame_bgr
        self.frame_cnt += 1
        enabled_global = {"circle", "triangle", "rect"}
        frame_draw = frame_bgr.copy()
        labels = detect_shapes(frame_draw, list(self.colors.values()), enabled_global)

        gold_cfg = self.colors.get("金色", None)
        gold_circle_on = bool(gold_cfg and gold_cfg.shape_checks.get("circle", None) and gold_cfg.shape_checks["circle"].isChecked())
        if gold_circle_on and not any(t.startswith("金色") for t, *_ in labels):
            cands = detect_gold_circle_robust(frame_draw, minR=60, maxR=180)
            if cands:
                x, y, r, _ = max(cands, key=lambda t: t[3])
                cv2.circle(frame_draw, (int(x), int(y)), int(r), (0, 215, 255), 2)
                labels.append(("金色-圆形", (int(x - r), int(y - r - 6)), QtGui.QColor(255, 215, 0)))

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        for cfg in self.colors.values():
            if cfg.sliders:
                mask = cv2.inRange(hsv, cfg.lower, cfg.upper)
                if cfg.name in self.mask_windows and self.mask_windows[cfg.name].isVisible():
                    try: self.mask_windows[cfg.name].update_mask(mask)
                    except Exception as e: print(f"[掩膜更新失败] {cfg.name}: {e}")

        rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        pix  = QtGui.QPixmap.fromImage(qimg)
        painter = QtGui.QPainter(pix)
        painter.setFont(QtGui.QFont("微软雅黑", 16, QtGui.QFont.Bold))
        for text, (tx, ty), qcol in labels:
            painter.setPen(qcol); painter.drawText(tx, ty, text)
        painter.end()
        scaled = pix.scaled(self.video_lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        rounded = rounded_qpixmap(scaled, 18)
        self.video_lbl.setPixmap(rounded)

    def recognize_once(self):
        if self.last_frame_bgr is None:
            print("[识别] 当前没有画面")
            self._publish_modbus_result(0)
            return
        enabled_global = {"circle", "triangle", "rect"}
        img = self.last_frame_bgr.copy()
        labels = detect_shapes(img, list(self.colors.values()), enabled_global)
        gold_cfg = self.colors.get("金色", None)
        gold_circle_on = bool(gold_cfg and gold_cfg.shape_checks.get("circle", None) and gold_cfg.shape_checks["circle"].isChecked())
        if gold_circle_on and not any(t.startswith("金色") for t, *_ in labels):
            cands = detect_gold_circle_robust(img, minR=60, maxR=180)
            if cands:
                x, y, r, _ = max(cands, key=lambda t: t[3])
                cv2.circle(img, (int(x), int(y)), int(r), (0, 215, 255), 2)
                labels.append(("金色-圆形", (int(x - r), int(y - r - 6)), QtGui.QColor(255, 215, 0)))
        result_values: List[int] = []
        if labels:
            self.msg_label.setText("\n".join([t for t,_,_ in labels]))
            self.msg_timer.start(2000)
            for text, *_ in labels:
                code = self.result_codes.get(text)
                if code is not None:
                    result_values.append(int(code))
            if not result_values:
                print("[识别] 未找到匹配的结果编码，保持 0")
        else:
            print("[识别] 未检测到目标")
            self.msg_label.setText("")
        self._publish_modbus_result(result_values)

    def toggle_mask(self, name: str):
        if name in self.mask_windows and self.mask_windows[name].isVisible():
            self.mask_windows[name].close(); return
        if name not in self.mask_windows:
            self.mask_windows[name] = MaskWindow(self.colors[name].mask_button_title, self)
            self.mask_windows[name].destroyed.connect(lambda _, n=name: self.mask_windows.pop(n, None))
        self.mask_windows[name].show(); self.mask_windows[name].raise_(); self.mask_windows[name].activateWindow()

    def on_timer(self):
        if self.frame_cnt % FPS_CALC_INTERVAL == 0 and self.frame_cnt > 0:
            now = time.time(); self.fps = FPS_CALC_INTERVAL / (now - self.last_time + 1e-9)
            self.last_time = now

    @QtCore.pyqtSlot(str)
    def on_info(self, s: str):
        print(s)
        if s.startswith("[INFO]"):
            self.lbl_cam.setText(s.replace("[INFO]", " ").strip())
        elif s.startswith("[FPS]"):
            self.lbl_fps.setText(s.replace("[FPS]","FPS").strip())

    def closeEvent(self, e):
        self._stop_modbus_server()
        self.stop_camera()
        super().closeEvent(e)

def main():
    cfg = load_config(CONFIG_PATH)
    app = QtWidgets.QApplication(sys.argv)
    try: app.setWindowIcon(QIcon(resource_path(APP_ICON)))
    except Exception: pass
    win = MainWindow(cfg); win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
