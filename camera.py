# -*- coding: utf-8 -*-
import os, sys
from pathlib import Path
import struct

def add_mvs_runtime_from_system():
    # 常见安装位置（64 位）
    candidates = [
        Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"),
        Path(r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64"),
    ]
    for p in candidates:
        if p.exists():
            # Python 3.8+ 正确做法：把目录加入本进程 DLL 搜索路径
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(p))   # 影响本进程的 DLL 搜索
            # 兜底再拼到 PATH（部分三方仍依赖）
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
            return True
    return False

if getattr(sys, "frozen", False):
    add_mvs_runtime_from_system()


from MvCameraControl_class import *  # 或你的实际导入

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
import cv2
import ctypes, json, socketserver, threading, time
from MvCameraControl_class import (
    MvCamera,
    MV_CC_DEVICE_INFO_LIST, MV_CC_DEVICE_INFO,
    MVCC_INTVALUE, MVCC_ENUMVALUE, MV_FRAME_OUT_INFO_EX,
    MV_CC_PIXEL_CONVERT_PARAM,
)
import MvCameraControl_class as mv
from CameraParams_header import *
from PixelType_header import *

MV_ACCESS_EXCLUSIVE       = getattr(mv, "MV_ACCESS_Exclusive", 1)
MV_GIGE_DEVICE_SAFE       = getattr(mv, "MV_GIGE_DEVICE", 1)
MV_TRIGGER_MODE_OFF_SAFE  = getattr(mv, "MV_TRIGGER_MODE_OFF", 0)
MV_TRIGGER_MODE_ON_SAFE   = getattr(mv, "MV_TRIGGER_MODE_ON", 1)

CONFIG_PATH = "config.json"
TARGET_DISPLAY_WIDTH = 1280
UI_TARGET_FPS = 15.0
UI_PAINT_FPS = 12.0
CAMERA_INIT_FPS = 5.0
CAM_THROUGHPUT_MBPS = 80
GIGE_PACKET_DELAY = 8000

APP_TITLE = "HIK MVS"
APP_ICON  = "Camera.ico"
CHS = {"circle": "圆形", "triangle": "三角形", "rect": "正方形"}
MIN_AREA, MAX_AREA = 500, 300_000
FPS_CALC_INTERVAL  = 30

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
            "server": {"host": "192.168.0.55", "port": 502},
            "cmd_map": {
            },
            "colors": [
                
            ]
        }
    if isinstance(cfg, list):
        cfg = {"server": {"host":"192.168.0.55","port":502}, "cmd_map": {}, "colors": cfg}
    cfg.setdefault("server", {"host": "192.168.0.55", "port": 502})
    cfg.setdefault("cmd_map", {})
    cfg.setdefault("colors", [])
    return cfg

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

def result_codes_from_cmd_map(cmd_map: dict) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for label, value in cmd_map.items():
        if isinstance(value, int):
            out[label] = value
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                out[label] = int(stripped)
                continue
            try:
                parsed = json.loads(stripped)
            except Exception:
                print(f"[MODBUS] 忽略无法解析的 cmd_map 项: {label}")
                continue
        else:
            parsed = value if isinstance(value, dict) else None
        if isinstance(parsed, dict):
            code = parsed.get("code")
            if isinstance(code, int):
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

def _get_i32(cam, key):
    st = MVCC_INTVALUE()
    if cam.MV_CC_GetIntValue(key, st) != 0: raise RuntimeError(f"Get {key} 失败")
    return st.nCurValue

def _get_enum(cam, key):
    st = MVCC_ENUMVALUE()
    if cam.MV_CC_GetEnumValue(key, st) != 0: raise RuntimeError(f"Get {key} 失败")
    return st.nCurValue

def _set_int(cam, key, val):
    try: return cam.MV_CC_SetIntValue(key, int(val))
    except Exception: return -1

def _set_float(cam, key, val):
    try: return cam.MV_CC_SetFloatValue(key, float(val))
    except Exception: return -1

def _set_enum(cam, key, val):
    try: return cam.MV_CC_SetEnumValue(key, int(val))
    except Exception: return -1

def open_first_gige():
    dev_list = MV_CC_DEVICE_INFO_LIST()
    if MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE_SAFE, dev_list) != 0 or dev_list.nDeviceNum == 0:
        raise RuntimeError("未发现 GigE 相机")
    cam = MvCamera()
    dev_info = ctypes.cast(dev_list.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents
    if cam.MV_CC_CreateHandle(dev_info) != 0:
        raise RuntimeError("CreateHandle 失败")
    if cam.MV_CC_OpenDevice(MV_ACCESS_EXCLUSIVE, 0) != 0:
        cam.MV_CC_DestroyHandle(); raise RuntimeError("OpenDevice 失败")

    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF_SAFE)
    try:
        cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        _set_float(cam, "AcquisitionFrameRate", CAMERA_INIT_FPS)
    except Exception:
        pass
    try:
        _set_int(cam, "GevSCPD", GIGE_PACKET_DELAY)
        _set_int(cam, "GevSCPBandwidth", CAM_THROUGHPUT_MBPS*1024*1024)
    except Exception:
        pass
    return cam

def _reshape_with_stride(raw_view, h, w, pix_type, nbytes):
    arr = np.frombuffer(raw_view, dtype=np.uint8)[:nbytes]
    return arr

_BAYER_CODE = {
    PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerRG2BGR,
    PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerBG2BGR,
    PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGB2BGR,
    PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGR2BGR,
}

def _shift_bayer_tag(base_tag, ox, oy):
    lut = {
        PixelType_Gvsp_BayerRG8: (PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8),
        PixelType_Gvsp_BayerGR8: (PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerBG8, PixelType_Gvsp_BayerGB8),
        PixelType_Gvsp_BayerGB8: (PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8, PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGR8),
        PixelType_Gvsp_BayerBG8: (PixelType_Gvsp_BayerBG8, PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8),
    }
    idx = ((oy & 1) << 1) | (ox & 1)
    return lut.get(base_tag, (base_tag,)*4)[idx]

class HikGrabber(QtCore.QThread):
    frameSignal = QtCore.pyqtSignal(np.ndarray)
    infoSignal  = QtCore.pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = None
        self._running = False
        self._last_emit_ts = 0.0
    def run(self):
        try:
            self.cam = open_first_gige()
            width  = _get_i32(self.cam, "Width")
            height = _get_i32(self.cam, "Height")
            pix    = _get_enum(self.cam, "PixelFormat")
            self.infoSignal.emit(f"[INFO] {width}x{height}")  # 不显示 PixelFormat
            if self.cam.MV_CC_StartGrabbing() != 0:
                raise RuntimeError("StartGrabbing 失败")
            pl = MVCC_INTVALUE(); self.cam.MV_CC_GetIntValue("PayloadSize", pl)
            in_size = int(pl.nCurValue if pl.nCurValue > 0 else width*height*3)
            buf = (ctypes.c_ubyte * in_size)()
            frame_info = MV_FRAME_OUT_INFO_EX()
            self._running = True
            t0 = time.time(); grabbed = 0
            while self._running:
                nret = self.cam.MV_CC_GetOneFrameTimeout(buf, in_size, frame_info, 1000)
                if nret != 0: continue
                grabbed += 1
                w, h = frame_info.nWidth, frame_info.nHeight
                raw = memoryview(buf)[:frame_info.nFrameLen]
                pt  = frame_info.enPixelType
                # 1) SDK 转 BGR8
                out_size = w*h*3
                out_buf  = (ctypes.c_ubyte * out_size)()
                cvt = MV_CC_PIXEL_CONVERT_PARAM()
                cvt.nWidth  = w; cvt.nHeight = h
                cvt.enSrcPixelType = pt
                cvt.enDstPixelType = PixelType_Gvsp_BGR8_Packed
                cvt.pSrcData = buf
                cvt.nSrcDataLen = frame_info.nFrameLen
                cvt.pDstBuffer = out_buf
                cvt.nDstBufferSize = out_size
                ret2 = self.cam.MV_CC_ConvertPixelType(cvt)
                if ret2 == 0:
                    img = np.frombuffer(out_buf, dtype=np.uint8)[: out_size].reshape(h, w, 3)
                else:
                    plane = _reshape_with_stride(raw, h, w, pt, frame_info.nFrameLen)
                    if pt == PixelType_Gvsp_Mono8:
                        gray = plane.reshape(h, w)
                        img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    elif pt in (PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerBG8,
                                PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerGR8):
                        base = pt
                        try:
                            ox = _get_i32(self.cam, "OffsetX")
                            oy = _get_i32(self.cam, "OffsetY")
                        except Exception:
                            ox = oy = 0
                        tag = _shift_bayer_tag(base, ox, oy)
                        bayer = plane.reshape(h, w)
                        img   = cv2.cvtColor(bayer, _BAYER_CODE.get(tag, cv2.COLOR_BayerRG2BGR))
                    elif pt == PixelType_Gvsp_RGB8_Packed:
                        rgb = plane.reshape(h, w, 3)
                        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    elif pt == PixelType_Gvsp_BGR8_Packed:
                        img = plane.reshape(h, w, 3)
                    else:
                        gray = plane.reshape(h, -1)[:, :w]
                        img  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if img.shape[1] > TARGET_DISPLAY_WIDTH:
                    scale = TARGET_DISPLAY_WIDTH / float(img.shape[1])
                    img = cv2.resize(img, (TARGET_DISPLAY_WIDTH, int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
                now_ts = time.time()
                if (now_ts - self._last_emit_ts) < (1.0 / UI_TARGET_FPS):
                    continue
                self._last_emit_ts = now_ts
                self.frameSignal.emit(img)
                now = time.time()
                if now - t0 >= 1.0:
                    self.infoSignal.emit(f"[FPS] {grabbed/(now-t0):.1f}")
                    t0 = now; grabbed = 0
        except Exception as e:
            self.infoSignal.emit(f"[HIK] 取流异常: {e}")
        finally:
            self._stop_and_close()
    def stop(self): self._running = False
    def _stop_and_close(self):
        try:
            if self.cam:
                try: self.cam.MV_CC_StopGrabbing()
                except Exception: pass
                try: self.cam.MV_CC_CloseDevice()
                except Exception: pass
                try: self.cam.MV_CC_DestroyHandle()
                except Exception: pass
        finally:
            self.cam = None

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
        self.modbus_host = str(svr.get("host", "192.168.0.55"))
        self.modbus_port = int(svr.get("port", 502))
        self.modbus_model = ModbusRegisterModel(size=16)
        self.modbus_server = None
        self.modbus_error: str | None = None
        try:
            self.modbus_server = start_modbus_server(
                self.modbus_host, self.modbus_port, self.modbus_model, on_write=self._on_modbus_write
            )
        except Exception as exc:
            print(f"[MODBUS] 启动失败: {exc}")
            self.modbus_server = None
            self.modbus_error = str(exc)
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

        self.grabber = HikGrabber(self)
        self.grabber.frameSignal.connect(self.on_frame_from_hik)
        self.grabber.infoSignal.connect(self.on_info)
        self.grabber.start()

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
        self.modbus_ip_label = QtWidgets.QLabel(self.modbus_host)
        self.modbus_port_label = QtWidgets.QLabel(str(self.modbus_port))
        self.modbus_status_lbl = QtWidgets.QLabel("")
        form.addRow("服务器IP:", self.modbus_ip_label)
        form.addRow("端口:", self.modbus_port_label)
        form.addRow("状态:", self.modbus_status_lbl)
        self.recognize_btn = QtWidgets.QPushButton("手动识别")
        self.recognize_btn.clicked.connect(self.recognize_once)
        form.addRow(self.recognize_btn)
        vbox.addWidget(g_modbus)
        vbox.addStretch(1)

    def _update_modbus_status(self):
        if self.modbus_server:
            status = "运行"
        elif getattr(self, "modbus_error", None):
            status = f"未启动: {self.modbus_error}"
        else:
            status = "未启动"
        if hasattr(self, "modbus_status_lbl"):
            self.modbus_status_lbl.setText(status)

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

    def reopen_camera(self):
        if getattr(self, "grabber", None) and self.grabber.isRunning():
            self.grabber.stop(); self.grabber.wait(1000)
        self.grabber = HikGrabber(self)
        self.grabber.frameSignal.connect(self.on_frame_from_hik)
        self.grabber.infoSignal.connect(self.on_info)
        self.grabber.start()
    def stop_camera(self):
        if getattr(self, "grabber", None) and self.grabber.isRunning():
            self.grabber.stop(); self.grabber.wait(1000)

    def _on_modbus_write(self, addr: int, value: int):
        if addr == 0 and value == 1:
            print("[MODBUS] 收到拍照请求")
            self.modbus_trigger_sig.emit()

    @QtCore.pyqtSlot()
    def _on_modbus_trigger(self):
        self.modbus_model.set_register(1, 0)
        self.recognize_once()
        self.modbus_model.set_register(0, 0)

    def _publish_modbus_result(self, value: int):
        if getattr(self, "modbus_model", None):
            self.modbus_model.set_register(1, value & 0xFFFF)

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
        result_value = 0
        if labels:
            self.msg_label.setText("\n".join([t for t,_,_ in labels]))
            self.msg_timer.start(2000)
            for text, *_ in labels:
                code = self.result_codes.get(text)
                if code is not None:
                    result_value = int(code)
                    break
            if result_value == 0:
                print("[识别] 未找到匹配的结果编码，保持 0")
        else:
            print("[识别] 未检测到目标")
            self.msg_label.setText("")
        self._publish_modbus_result(result_value)

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

    def reopen_camera(self):
        if getattr(self, "grabber", None) and self.grabber.isRunning():
            self.grabber.stop(); self.grabber.wait(1000)
        self.grabber = HikGrabber(self)
        self.grabber.frameSignal.connect(self.on_frame_from_hik)
        self.grabber.infoSignal.connect(self.on_info)
        self.grabber.start()

    def stop_camera(self):
        if getattr(self, "grabber", None) and self.grabber.isRunning():
            self.grabber.stop(); self.grabber.wait(1000)

    def closeEvent(self, e):
        if getattr(self, "modbus_server", None):
            try:
                self.modbus_server.shutdown()
            except Exception:
                pass
            try:
                self.modbus_server.server_close()
            except Exception:
                pass
            self.modbus_server = None
        if getattr(self, "grabber", None):
            try:
                self.grabber.stop(); self.grabber.wait(1000)
            except Exception:
                pass
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
