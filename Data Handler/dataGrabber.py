import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import easyocr
import torch
import moviepy as mp
import sys
import subprocess
from typing import Optional

# Ensure line-buffered output for GUI log streaming
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# --- Paths ---
baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scriptDir = os.path.dirname(os.path.abspath(__file__))
regionsFile = os.path.join(scriptDir, "roi_regions.txt")

def pickLatestVideo(rootDir: str) -> str:
    override = os.environ.get("ANRO_VIDEO")
    if override and os.path.isfile(override):
        return override
    exts = {".mp4", ".mkv"}
    candidates = []
    for name in os.listdir(rootDir):
        path = os.path.join(rootDir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in exts:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No video files found in {rootDir}")
    return max(candidates, key=lambda p: os.path.getmtime(p))

def loadRegionsFromFile(path: str):
    if not os.path.isfile(path):
        return None
    regions = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, coords = line.split("=", 1)
                key = key.strip()
                parts = [p.strip() for p in coords.split(",")]
                if len(parts) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = (int(p) for p in parts)
                except ValueError:
                    continue
                regions[key] = (x1, y1, x2, y2)
    except Exception as e:
        print(f"Warning: failed to load ROI regions from {path}: {e}")
        return None
    return regions if regions else None

videoPath = pickLatestVideo(baseDir)
print(f"Using video: {videoPath}")

# --- Regions of Interest (ROIs) --- (must be generated per video via boundaryFinder.py)
regions = loadRegionsFromFile(regionsFile)
if not regions:
    raise FileNotFoundError(
        f"ROI regions file not found or empty: {regionsFile}. "
        "Run boundaryFinder.py to generate it for this video."
    )
print(f"Loaded ROI regions from {regionsFile}")

# --- Sampling Control ---
# Set how many frames per second to OCR (e.g., 60, 30, 15). None means process all frames.
sampleFpsConst = None

# Debug output for ROI snapshots (set False to disable extra images and conf columns)
debugToggle = False

# --- GPU Initialization ---
# Allow forcing CPU via CLI flag --force-cpu to avoid CUDA hangs
forceCpu = any(arg.lower() == "--force-cpu" for arg in sys.argv)
_useGpu = (torch.cuda.is_available() and not forceCpu)
reader = easyocr.Reader(['en'], gpu=_useGpu)
print(f"EasyOCR initialized. GPU={'ON' if _useGpu else 'OFF'}")
try:
    # Enable cuDNN autotune to pick optimal algorithms for current shapes
    import torch.backends.cudnn as cudnn
    if _useGpu and torch.backends.cudnn.is_available():
        cudnn.benchmark = True
        print("cuDNN benchmark autotune: ON")
    else:
        print("cuDNN benchmark autotune: OFF")
except Exception as _cudnnE:
    print(f"cuDNN setup skipped: {_cudnnE}")
if _useGpu:
    reader.detector.to(torch.device('cuda'))
    reader.recognizer.to(torch.device('cuda'))
    print("EasyOCR detector and recognizer moved to GPU.")
    print("Device:", next(reader.recognizer.parameters()).device)


outputCsv = "reactor_readings_cleaned.csv"
errorLog = "ocr_errors.log"

roiDebugRate = 50  # save every Nth frame
roiDebugDir = "roi_debug"
if debugToggle:
    os.makedirs(roiDebugDir, exist_ok=True)
    # Create a small test clip, to avoid checking ENTIRE video every time
    # Disable audio to reduce I/O and encoding time
    clip = mp.VideoFileClip(videoPath, audio=False).subclipped(0, 15)
    clip.write_videofile(os.path.join(baseDir, "test_clip.mp4"), codec="libx264", audio=False, logger='bar')
    clip.close()
    videoPath = os.path.join(baseDir, "test_clip.mp4")

# Optional ROI paddings (l, t, r, b) to reduce tight crops
roiPad = {
    # Fuel tends to clip the last digit when animating --> add right padding
    "fuel": (0, 0, 10, 0),
    # Rod insertion can lose bottom strokes --> add bottom padding
    "rodInsertion": (0, 0, 0, 4),
}

# --- Sanity Ranges ---
ranges = {
    "temperature":    (300, 20000),
    "pressure":       (500, 15000),
    "fuel":           (0, 100),
    "rodInsertion":  (0, 100),
    "waterLevel":    (0, 100),   # 0–100 %
    "feedwaterFlow": (0, 2.0),   # 0–~2 L/s (0, 0.91, 1.83)
    "fwpFlowRate1": (0.0, 1.5),
    "fwpUtilization1": (0.0, 100.0),
    "fwpRpm1": (0, 5000),
    "fwpFlowRate2": (0.0, 1.5),
    "fwpUtilization2": (0.0, 100.0),
    "fwpRpm2": (0, 5000),
    "totalOutput": (0, 50000),
    "currentPowerOrder": (0, 50000),
    "marginOfError": (1000, 1500),
    "flowRate1": (0.0, 15.0),
    "flowRate2": (0.0, 15.0),
    "rpm1": (0, 5000),
    "rpm2": (0, 5000),
    "valvesPct1": (0.0, 100.0),
    "valvesPct2": (0.0, 100.0),
    "vibration1": (100.0, 500.0),
    "vibration2": (100.0, 500.0),
}


# --- Confidence + constraints ---
# Minimum confidence to accept a fresh value per signal
confThresh = {
    "temperature": 0.20,
    "pressure": 0.20,
    "fuel": 0.40,
    "rodInsertion": 0.25,
    "totalOutput": 0.20,
    "currentPowerOrder": 0.20,
    "marginOfError": 0.20,
    "fwpFlowRate1": 0.20,
    "fwpUtilization1": 0.20,
    "fwpRpm1": 0.20,
    "fwpFlowRate2": 0.20,
    "fwpUtilization2": 0.20,
    "fwpRpm2": 0.20,
    "flowRate1": 0.20,
    "flowRate2": 0.20,
    "rpm1": 0.20,
    "rpm2": 0.20,
    "valvesPct1": 0.20,
    "valvesPct2": 0.20,
    "vibration1": 0.20,
    "vibration2": 0.20,
}
# Decimal handling for fields that include decimals in-range
decimalKeys = {
    "flowRate1",
    "flowRate2",
    "valvesPct1",
    "valvesPct2",
    "fwpFlowRate1",
    "fwpFlowRate2",
    "fwpUtilization1",
    "fwpUtilization2",
}
keyDecimals = {
    "temperature": 1,
    "pressure": 1,
    "fuel": 1,
    "rodInsertion": 1,
    "waterLevel": 1,
    "feedwaterFlow": 2,
    "fwpFlowRate1": 2,
    "fwpUtilization1": 1,
    "fwpRpm1": 0,
    "fwpFlowRate2": 2,
    "fwpUtilization2": 1,
    "fwpRpm2": 0,
    "totalOutput": 0,
    "currentPowerOrder": 0,
    "marginOfError": 0,
    "flowRate1": 2,
    "flowRate2": 2,
    "rpm1": 0,
    "rpm2": 0,
    "valvesPct1": 1,
    "valvesPct2": 1,
    "vibration1": 0,
    "vibration2": 0,
}
# Max allowed change per frame for constrained signals
maxDeltaPerFrame = {"fuel": 0.1}
# EMA for confidence readout smoothing (display only)
confEmaAlpha = 0.4

# --- Helper Functions ---
def clampOrNan(v, key):
    if v is None:
        return None
    lo, hi = ranges.get(key, (-float("inf"), float("inf")))
    return v if lo <= v <= hi else None

def preferNear(prev, cand, maxJump):
    if cand is None or prev is None:
        return cand
    return cand if abs(cand - prev) <= maxJump else prev

def applyCurrentPowerOrderRule(prev, cand):
    if prev is not None and pd.isna(prev):
        prev = None
    if cand is not None and pd.isna(cand):
        cand = None
    if cand is None:
        return None if (prev is None or prev == 0) else prev
    if prev is None or prev == 0:
        return cand
    if cand == 0:
        return 0
    return prev if cand != prev else prev

# --- OCR preprocessing ---
def preprocessVariantsSimple(roi):
    # Pruned variants: CLAHE + Gaussian + (Otsu, Adaptive Gaussian)
    roi = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    vGauss = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    vOtsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kern = np.ones((2, 2), np.uint8)
    variants = [
        cv2.morphologyEx(vGauss, cv2.MORPH_CLOSE, kern, iterations=1),
        cv2.morphologyEx(vOtsu, cv2.MORPH_CLOSE, kern, iterations=1),
    ]
    return variants

def fastVariant(roi):
    # Single fast variant for batching: CLAHE + Gaussian + Otsu + CLOSE
    roi = cv2.copyMakeBorder(roi, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    vOtsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kern = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(vOtsu, cv2.MORPH_CLOSE, kern, iterations=1)

def readWithEasyocrSimple(img, allow):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    res = reader.readtext(img, detail=1, paragraph=False, allowlist=allow)
    if not res:
        return "", 0.0
    best = max(res, key=lambda x: x[2])
    return best[1], float(best[2])

digitFixSafe = str.maketrans({'O':'0','o':'0','S':'5','s':'5','I':'1','l':'1','B':'8'})

# --- Batched OCR helper ---
def readtextMulti(images, allow=None):
    try:
        if hasattr(reader, 'readtext_batched'):
            return reader.readtext_batched(list(images), detail=1, paragraph=False, allowlist=allow)
    except Exception:
        pass
    out = []
    for img in images:
        try:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            out.append(reader.readtext(img, detail=1, paragraph=False, allowlist=allow))
        except Exception:
            out.append([])
    return out

# --- Field-aware numeric OCR ---
def _normalizeText(s: str) -> str:
    s = s.replace(',', '.')
    s = s.replace(' ', '')
    s = s.upper()
    return s

def _isNaPowerOrder(text: str) -> bool:
    t = _normalizeText(text)
    t = re.sub(r"[^A-Z/]", "", t).replace("\\", "/").replace("-", "/")
    return t in ("NA", "N/A")

_patterns = {
    # Anchor full token: exactly NNNN.NK
    'temperature': re.compile(r"^(?P<VAL>\d{3,4}\.\d)K$"),
    # Exactly NNNN.NKPA
    'pressure': re.compile(r"^(?P<VAL>\d{3,5}\.\d)KPA$"),
    # Exactly NN(.N)%
    'percent': re.compile(r"^(?P<VAL>\d{1,3}(?:\.\d)?)%$"),
    # Optional % for valve readouts
    'percent_opt': re.compile(r"^(?P<VAL>\d{1,3}(?:\.\d)?)%?$"),
    # Optional kW suffix
    'kw': re.compile(r"^(?P<VAL>\d{1,5}(?:\.\d)?)(?:KW)?$"),
    # Optional RPM suffix
    'rpm': re.compile(r"^(?P<VAL>\d{1,5}(?:\.\d)?)(?:RPM)?$"),
    # Optional L/S suffix
    'flow': re.compile(r"^(?P<VAL>\d{1,3}(?:\.\d{1,2})?)(?:L/S|LS)?$"),
    # Plain numeric (no units)
    'plain': re.compile(r"^(?P<VAL>\d{1,4}(?:\.\d)?)$"),
}

def _coerceDecimalForRange(key: str, val: float, text: str):
    if val is None or key not in decimalKeys:
        return val
    t = _normalizeText(text)
    if "." in t:
        return val
    lo, hi = ranges.get(key, (-float("inf"), float("inf")))
    if val <= hi:
        return val
    for div in (10.0, 100.0, 1000.0):
        scaled = val / div
        if lo <= scaled <= hi:
            return scaled
    return val

def _extractValueForKey(key: str, text: str):
    t = _normalizeText(text)
    if key in ("fuel", "rodInsertion"):
        m = _patterns['percent'].match(t)
        return float(m.group('VAL')) if m else None
    if key == 'temperature':
        m = _patterns['temperature'].match(t)
        return float(m.group('VAL')) if m else None
    if key == 'pressure':
        m = _patterns['pressure'].match(t)
        return float(m.group('VAL')) if m else None
    if key in ("totalOutput", "currentPowerOrder", "marginOfError"):
        m = _patterns['kw'].match(t)
        val = float(m.group('VAL')) if m else None
        return _coerceDecimalForRange(key, val, t)
    if key in ("rpm1", "rpm2"):
        m = _patterns['rpm'].match(t)
        val = float(m.group('VAL')) if m else None
        return _coerceDecimalForRange(key, val, t)
    if key in ("fwpRpm1", "fwpRpm2"):
        m = _patterns['rpm'].match(t)
        val = float(m.group('VAL')) if m else None
        return _coerceDecimalForRange(key, val, t)
    if key in ("flowRate1", "flowRate2"):
        m = _patterns['flow'].match(t)
        val = float(m.group('VAL')) if m else None
        return _coerceDecimalForRange(key, val, t)
    if key in ("fwpFlowRate1", "fwpFlowRate2"):
        m = _patterns['flow'].match(t)
        val = float(m.group('VAL')) if m else None
        return _coerceDecimalForRange(key, val, t)
    if key in ("valvesPct1", "valvesPct2"):
        m = _patterns['percent_opt'].match(t)
        val = float(m.group('VAL')) if m else None
        return _coerceDecimalForRange(key, val, t)
    if key in ("fwpUtilization1", "fwpUtilization2"):
        m = _patterns['percent_opt'].match(t)
        val = float(m.group('VAL')) if m else None
        return _coerceDecimalForRange(key, val, t)
    if key in ("vibration1", "vibration2"):
        m = _patterns['plain'].match(t)
        return float(m.group('VAL')) if m else None
    return None

def parseWaterLevel(text: str, prev: Optional[float] = None):
    """Parse water level while fixing missing/misplaced decimals using heuristics."""
    if not text:
        return None

    raw = _normalizeText(text).translate(digitFixSafe).replace("%", "")
    raw = raw.strip()
    if not raw:
        return None

    digitsOnly = re.sub(r"[^0-9]", "", raw)
    candidates = {}

    def _addCandidate(val, penalty):
        if val is None:
            return
        v = round(float(val), 1)
        if 0.0 <= v <= 100.0:
            prevPenalty = candidates.get(v)
            if prevPenalty is None or penalty < prevPenalty:
                candidates[v] = penalty

    def _tryFloat(textValue, penalty):
        try:
            _addCandidate(float(textValue), penalty)
        except ValueError:
            pass

    basePenalty = 0.0 if "." in raw else 0.08
    _tryFloat(raw, basePenalty)

    if raw.count(".") > 1:
        collapsed = raw.replace(".", "", raw.count(".") - 1)
        _tryFloat(collapsed, basePenalty + 0.02)

    if digitsOnly:
        try:
            digitsVal = int(digitsOnly)
            # Treat plain integer, but penalize since display normally shows a decimal.
            _addCandidate(float(digitsVal), 0.25)
            decimalPenalty = 0.05 if "." not in raw else 0.10
            _addCandidate(digitsVal / 10.0, decimalPenalty)
            # Allow one more shift (e.g., OCR inserted decimal too early).
            if digitsVal >= 10:
                _addCandidate(digitsVal / 100.0, decimalPenalty + 0.05)
        except ValueError:
            pass

    if not candidates:
        return None

    if prev is None:
        chosen = min(candidates.items(), key=lambda kv: (kv[1], -kv[0]))
    else:
        chosen = min(candidates.items(), key=lambda kv: (abs(kv[0] - prev), kv[1]))
    return round(chosen[0], 1)

def parseFeedwaterFlow(text: str):
    """
    Accepts '1.83 L/S', '0 L/S', '0.91L/S', '1.83', '0.91', etc.
    Snaps to one of {0, 0.91, 1.83}.
    """
    if not text:
        return None
    t = _normalizeText(text)
    t = t.replace("L/S","").replace("LS","")
    t = re.sub(r"[^0-9.]", "", t)

    try:
        raw = float(t)
    except:
        return None

    candidates = [0.0, 0.91, 1.83]
    snapped = min(candidates, key=lambda x: abs(x - raw))
    return snapped

def _unitBonus(key: str, text: str) -> float:
    t = _normalizeText(text).upper()
    if key == 'temperature' and 'K' in t:
        return 0.15
    if key == 'pressure' and 'KPA' in t.upper():
        return 0.15
    if key in ('fuel','rodInsertion') and '%' in t:
        return 0.10
    if key in ("totalOutput", "currentPowerOrder", "marginOfError") and 'KW' in t:
        return 0.10
    if key in ("rpm1", "rpm2") and 'RPM' in t:
        return 0.10
    if key in ("fwpRpm1", "fwpRpm2") and 'RPM' in t:
        return 0.10
    if key in ("flowRate1", "flowRate2") and ('L/S' in t or 'LS' in t):
        return 0.10
    if key in ("fwpFlowRate1", "fwpFlowRate2") and ('L/S' in t or 'LS' in t):
        return 0.10
    if key in ("valvesPct1", "valvesPct2") and '%' in t:
        return 0.10
    if key in ("fwpUtilization1", "fwpUtilization2") and '%' in t:
        return 0.10
    return 0.0

def _quantizeValue(key: str, v: float) -> float:
    if v is None:
        return None
    if key in keyDecimals:
        return round(v, keyDecimals[key])
    return v


if _useGpu:
    print("CUDA available:", torch.cuda.is_available())
    print("Device name:", torch.cuda.get_device_name(0))

def ocrNumericForKey(key: str, variants):
    # Aggregate all detections across variants, bucketed by quantized value
    allow = "0123456789.%KkPpAaWwRrMmLlSs/Nn"
    buckets = {}  # val_bucket -> sum_score
    confSum = {}  # val_bucket -> combined confidence (noisy-or)
    bestTxtFor = {}  # representative text per bucket
    bestConfFor = {}  # val_bucket -> best single conf

    # Batch across variants in a single EasyOCR call to reduce overhead
    bestSeenConf = 0.0
    bestNaConf, bestNaTxt = 0.0, ""
    for img in list(variants):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        results = reader.readtext(img, detail=1, paragraph=False, allowlist=allow)
        for (_box, txt, conf) in results:
            txt = txt.translate(digitFixSafe)
            tnorm = _normalizeText(txt)
            if key == "currentPowerOrder" and _isNaPowerOrder(tnorm):
                raw = float(conf)
                if raw > bestNaConf:
                    bestNaConf, bestNaTxt = raw, txt
                if raw > bestSeenConf:
                    bestSeenConf = raw
                continue
            # Reject mixed junk: disallow letters not in the expected unit set
            if key in ("fuel", "rodInsertion"):
                allowedLetters = {"%"}
            elif key == "temperature":
                allowedLetters = {"K"}
            elif key == "pressure":
                allowedLetters = {"K", "P", "A"}
            elif key in ("totalOutput", "currentPowerOrder", "marginOfError"):
                allowedLetters = {"K", "W"}
            elif key in ("rpm1", "rpm2", "fwpRpm1", "fwpRpm2"):
                allowedLetters = {"R", "P", "M"}
            elif key in ("flowRate1", "flowRate2", "fwpFlowRate1", "fwpFlowRate2"):
                allowedLetters = {"L", "S"}
            elif key in ("fwpUtilization1", "fwpUtilization2"):
                allowedLetters = {"%"}
            else:
                allowedLetters = set()
            if any(ch.isalpha() and ch not in allowedLetters for ch in tnorm):
                continue
            val = _extractValueForKey(key, txt)
            if val is None:
                continue
            vb = _quantizeValue(key, val)
            raw = float(conf)
            score = raw + _unitBonus(key, txt) + min(len(txt), 10) * 0.01
            buckets[vb] = buckets.get(vb, 0.0) + score
            prev = confSum.get(vb, 0.0)
            confSum[vb] = 1.0 - (1.0 - prev) * (1.0 - raw)
            if vb not in bestConfFor or raw > bestConfFor[vb]:
                bestConfFor[vb] = raw
                bestTxtFor[vb] = txt
            if raw > bestSeenConf:
                bestSeenConf = raw
        # Early-exit if we already have a strong hit
        if bestSeenConf > 0.85:
            break

    if not buckets:
        if bestNaConf > 0.0:
            return None, min(0.99, bestNaConf), bestNaTxt
        return None, 0.0, ""

    # Pick the bucket with the highest aggregate score
    bestVal = max(buckets.items(), key=lambda kv: kv[1])[0]
    # Report confidence as the best single detection for that bucket (not summed)
    bestConf = min(0.99, bestConfFor.get(bestVal, 0.0))
    bestTxt = bestTxtFor.get(bestVal, "")
    if bestNaConf > bestConf:
        return None, min(0.99, bestNaConf), bestNaTxt
    return bestVal, bestConf, bestTxt

# --- State OCR ---
def ocrStateVariantsSimple(variants, keywords):
    # Robust OPEN/CLOSED detection across variants with basic normalization
    def _norm(s: str) -> str:
        t = (s or "").upper().replace('0', 'O').replace('1', 'I')
        t = t.replace('CIOSED', 'CLOSED').replace('CLO5ED', 'CLOSED')
        t = t.replace('OPFN', 'OPEN').replace('0PEN', 'OPEN')
        return t.strip()

    def looksOpen(tu: str) -> bool:
        if 'OPEN' in tu:
            return True
        # Handle partials like 'PEN', 'QPEN'
        if tu.startswith('PEN') or ' QPEN' in (' ' + tu) or ' PEN' in (' ' + tu):
            return True
        return False

    bestT, bestC = "", 0.0
    openSum, closedSum = 0.0, 0.0
    # Batch across variants
    allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ "
    for img in list(variants):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        results = reader.readtext(img, detail=1, paragraph=False, allowlist=allow)
        for (_box, t, c) in results:
            tu = _norm(t)
            cu = float(c)
            if looksOpen(tu):
                openSum += cu
            if 'CLOSED' in tu:
                closedSum += cu
            if cu > bestC:
                bestT, bestC = tu, cu
        if bestC > 0.85:
            break
    # Prefer the class with more aggregated evidence; tie favors OPEN
    foundOpen = (openSum >= closedSum) and (openSum > 0.0)
    return foundOpen, min(0.99, bestC), bestT

_feedRe = re.compile(r"([0-2])\s*/\s*2")
def parseFeedwaterCount(text: str):
    if not text:
        return None
    # Normalize and keep only digits and '/'
    t = re.sub(r"[^0-9/]", "", text.upper())
    if not t:
        return None
    m = _feedRe.search(t)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def ocrFeedwaterCountFromVariants(roi, variants):
    # Crop to where the ratio is (left side) to avoid letters in ACTIVE
    h, w = roi.shape[:2]
    ratioRoi = roi[:, : max(1, int(w * 0.60))]  # slightly wider to avoid cutting digits
    # Build variants from the ratio crop
    ratioVariants = preprocessVariantsSimple(ratioRoi)
    # Also consider the full-ROI variants as a fallback
    allVariants = list(ratioVariants) + list(variants)
    byVal = {0: 0.0, 1: 0.0, 2: 0.0}  # aggregate score (sum)
    confComb = {0: 0.0, 1: 0.0, 2: 0.0}  # combined confidence via noisy-or
    bestTxt, bestConf = "", 0.0
    allow = "0123456789/ "
    for img in list(allVariants):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        res = reader.readtext(img, detail=1, paragraph=False, allowlist=allow)
        for (_box, txt, conf) in res:
            # Keep only digits and '/'; tolerate stray characters
            t = re.sub(r"[^0-9/]", "", txt)
            if not t:
                continue
            m = _feedRe.search(t)
            if not m:
                continue
            try:
                num = int(m.group(1))
            except ValueError:
                continue
            if 0 <= num <= 2:
                raw = float(conf)
                byVal[num] += raw
                prev = confComb.get(num, 0.0)
                confComb[num] = 1.0 - (1.0 - prev) * (1.0 - raw)
                if raw > bestConf:
                    bestConf, bestTxt = float(conf), t
        if bestConf > 0.85:
            break

    picked = max(byVal.items(), key=lambda kv: kv[1])[0]
    if byVal[picked] == 0.0:
        return None, 0.0, ""
    # Return a normalized raw display with ACTIVE appended
    normRaw = f"{picked}/2 ACTIVE"
    return picked, min(0.99, bestConf), normRaw

# --- Stability buffers ---
stateWindow = 5  # smoothing window for states
stateBuffers = {"coolant": [], "feedwater": []}

# Pending change confirmation (e.g., require 2 consecutive frames)
# Pending change confirmation (e.g., require 2 consecutive frames)
pendingUpdates = {
    "fuel": {"cand": None, "count": 0},
    # Debounce single-frame glitches in water level
    "waterLevel": {"cand": None, "count": 0},
    # Debounce single-frame glitches in feedwaterFlow (0 / 0.91 / 1.83)
    "feedwaterFlow": {"cand": None, "count": 0},
}

# Max per-frame believable jump for water level (in %)
maxDeltaWaterPerFrame = 1.0

# Pending large-jump acceptance to avoid lock-in
maxJumpPerFrame = {
    "temperature": 120.0,
    "pressure": 120.0,
    "rodInsertion": 2.0,
    "totalOutput": 200.0,
    "fwpRpm1": 10.0,
    "fwpRpm2": 10.0,
    "fwpFlowRate1": 0.02,
    "fwpFlowRate2": 0.02,
    "fwpUtilization1": 0.2,
    "fwpUtilization2": 0.2,
    "rpm1": 10.0,
    "rpm2": 10.0,
    "flowRate1": 0.02,
    "flowRate2": 0.02,
    "valvesPct1": 0.2,
    "valvesPct2": 0.2,
    "vibration1": 20.0,
    "vibration2": 20.0,
}
jumpStabilityThresh = {
    "temperature": 30.0,
    "pressure": 50.0,
    "rodInsertion": 1.0,
    "totalOutput": float("inf"),
    "fwpRpm1": float("inf"),
    "fwpRpm2": float("inf"),
    "fwpFlowRate1": float("inf"),
    "fwpFlowRate2": float("inf"),
    "fwpUtilization1": float("inf"),
    "fwpUtilization2": float("inf"),
    "rpm1": float("inf"),
    "rpm2": float("inf"),
    "flowRate1": float("inf"),
    "flowRate2": float("inf"),
    "valvesPct1": float("inf"),
    "valvesPct2": float("inf"),
    "vibration1": float("inf"),
    "vibration2": float("inf"),
}  # how close consecutive jump candidates must be
jumpAcceptCount = {
    "temperature": 2,
    "pressure": 2,
    "rodInsertion": 2,
    "totalOutput": 3,
    "fwpRpm1": 3,
    "fwpRpm2": 3,
    "fwpFlowRate1": 3,
    "fwpFlowRate2": 3,
    "fwpUtilization1": 3,
    "fwpUtilization2": 3,
    "rpm1": 3,
    "rpm2": 3,
    "flowRate1": 3,
    "flowRate2": 3,
    "valvesPct1": 3,
    "valvesPct2": 3,
    "vibration1": 3,
    "vibration2": 3,
}
pendingJumps = {k: {"cand": None, "count": 0} for k in maxJumpPerFrame}

# Lag-spike detection (network hiccups): if a value is unchanged for >=1s then jumps a lot,
# treat it as a lag spike unless confirmed by consecutive frames.
lagHoldSeconds = 1.0
lagJumpFraction = 0.05  # 5% of full range
lagDetectKeys = {k for k in ranges.keys()}
lagJumpThreshold = {
    k: max(
        maxJumpPerFrame.get(k, 0.0) * 3.0,
        (ranges[k][1] - ranges[k][0]) * lagJumpFraction
    )
    for k in lagDetectKeys
}
pendingLag = {k: {"cand": None, "count": 0} for k in lagDetectKeys}

# Raw CSV output (no smoothing/interpolation)
rawOutputCsv = "reactor_readings_raw.csv"

# Confidence EMA memory (display only)
confEma = {}
roiHashCache = {}

def smoothState(key: str, value: float):
    # Binary majority smoothing (returns 0/1)
    if key not in stateBuffers:
        stateBuffers[key] = []
    stateBuffers[key].append(1 if value else 0)
    stateBuffers[key] = stateBuffers[key][-stateWindow:]
    # Dynamic majority threshold so early frames don't default to 0
    needed = (len(stateBuffers[key]) // 2) + 1
    return 1 if sum(stateBuffers[key]) >= needed else 0

def smoothMultistate(key, value):
    # Clamp to expected finite set and append
    try:
        v = int(value)
    except Exception:
        v = value
    if key == "feedwater":
        try:
            v = max(0, min(2, int(v)))
        except Exception:
            # leave as-is if cannot coerce; will be ignored by counts logic
            pass
    stateBuffers[key].append(v)
    stateBuffers[key] = stateBuffers[key][-stateWindow:]
    vals = stateBuffers[key]
    counts = {v: vals.count(v) for v in set(vals)}
    maxCount = max(counts.values())
    candidates = [v for v, c in counts.items() if c == maxCount]
    for v in reversed(vals):
        if v in candidates:
            return v
    return value

# --- OCR Loop ---
cap = cv2.VideoCapture(videoPath)
if not cap.isOpened():
    print(f"Error: failed to open video: {videoPath}")
totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video opened: {vw}x{vh} @ {fps:.3f} fps, frames={totalFrames}")

def getSampleIndices(totalFrames: int, fps: float, sampleFps) -> list:
    if not sampleFps or sampleFps >= fps:
        return list(range(totalFrames))
    # Time-based selection to distribute samples evenly across non-integer FPS
    indices = []
    t = 0.0
    stepT = 1.0 / float(sampleFps)
    while True:
        idx = int(round(t * fps))
        if idx >= totalFrames:
            break
        if not indices or idx != indices[-1]:
            indices.append(idx)
        t += stepT
    return indices

sampleIndices = getSampleIndices(totalFrames, fps, sampleFpsConst)

import tempfile
errors = []

_streamCols = ["timestamp"]
for _k, _c in regions.items():
    if not (_c[0] == _c[1] == _c[2] == _c[3] == 0):
        _streamCols += [_k, f"_raw_{_k}", f"_conf_{_k}"]
_seenCol = set()
_streamCols = [c for c in _streamCols if not (c in _seenCol or _seenCol.add(c))]
_rawCols = ["timestamp"] + [c for c in _streamCols if c in regions]

_streamFd, cleanedStreamPath = tempfile.mkstemp(suffix="_rows.csv")
os.close(_streamFd)
_rawStreamFd, rawStreamPath = tempfile.mkstemp(suffix="_rawrows.csv")
os.close(_rawStreamFd)

rowBuf, rawBuf = [], []
streamChunk = 2000
_streamState = {"rows": False, "raw": False}

def _flushBuf(buf, path, cols, stateKey):
    if not buf:
        return
    pd.DataFrame(buf).reindex(columns=cols).to_csv(
        path, index=False, mode="a", header=not _streamState[stateKey]
    )
    _streamState[stateKey] = True
    buf.clear()

lastOk = {
    "temperature": None,
    "pressure": None,
    "fuel": None,
    "rodInsertion": None,
    "waterLevel": None,
    "feedwaterFlow": None,
    "fwpFlowRate1": None,
    "fwpUtilization1": None,
    "fwpRpm1": None,
    "fwpFlowRate2": None,
    "fwpUtilization2": None,
    "fwpRpm2": None,
    "totalOutput": None,
    "currentPowerOrder": None,
    "rpm1": None,
    "rpm2": None,
    "flowRate1": None,
    "flowRate2": None,
    "valvesPct1": None,
    "valvesPct2": None,
    "vibration1": None,
    "vibration2": None,
}
lastConf = {k: None for k in lastOk.keys()}
lastRaw = {k: None for k in lastOk.keys()}
lastChangeTime = {k: None for k in lastOk.keys()}

print(f"Processing {len(sampleIndices)} sampled frames (of {totalFrames}) from {videoPath}...")

def iterSampledFrames(cap, sampleIndices):
    if not sampleIndices:
        return
    ptr = 0
    target = sampleIndices[ptr]
    frameIdx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frameIdx == target:
            yield frameIdx, frame
            ptr += 1
            if ptr >= len(sampleIndices):
                break
            target = sampleIndices[ptr]
        frameIdx += 1

for frameIdx, frame in tqdm(
    iterSampledFrames(cap, sampleIndices),
    total=len(sampleIndices),
    desc="Extracting OCR Data",
    ncols=80,
    file=sys.stdout,
):

    prevLastOk = lastOk.copy()
    prevLastConf = lastConf.copy()
    prevLastRaw = lastRaw.copy()
    prevLastChange = lastChangeTime.copy()

    # Frame hash skip: avoid reprocessing identical frames
    try:
        _fh = hash(frame.tobytes())
        if '_last_frame_hash' in globals() and _fh == globals().get('_last_frame_hash'):
            continue
        globals()['_last_frame_hash'] = _fh
    except Exception:
        pass

    timestamp = frameIdx / fps
    row = {"timestamp": timestamp}
    rawVals = {"timestamp": timestamp}
    for key, coords in regions.items():
        if coords[0] == coords[1] == coords[2] == coords[3] == 0:
            continue
        row[f"_conf_{key}"] = np.nan

    # First: build ROIs and fast variants per key (heavy variants built lazily on demand)
    roiMap, fastMap, varMap, roiCoords = {}, {}, {}, {}
    roiHashes, cacheHits = {}, {}
    for key, (x1, y1, x2, y2) in regions.items():
        try:
            # Skip disabled ROIs (all zeros)
            if x1 == x2 == y1 == y2 == 0:
                continue
            h, w = frame.shape[:2]
            xLo, xHi = (x1, x2) if x1 <= x2 else (x2, x1)
            yLo, yHi = (y1, y2) if y1 <= y2 else (y2, y1)
            lpad, tpad, rpad, bpad = roiPad.get(key, (0, 0, 0, 0))
            x1p = max(0, xLo - lpad)
            y1p = max(0, yLo - tpad)
            x2p = min(w, xHi + rpad)
            y2p = min(h, yHi + bpad)
            if x2p <= x1p or y2p <= y1p:
                raise ValueError(f"Invalid ROI for {key}: {(x1,y1,x2,y2)} -> {(x1p,y1p,x2p,y2p)}")
            roi = frame[y1p:y2p, x1p:x2p]
            roiMap[key] = roi
            roiCoords[key] = (x1p, y1p, x2p, y2p)
            roiHash = hash(roi.tobytes())
            roiHashes[key] = roiHash
            cached = roiHashCache.get(key)
            if cached is not None and cached.get("hash") == roiHash:
                cacheHits[key] = cached
            else:
                # Defer heavy variants; compute only if fast pass is weak
                fastMap[key] = fastVariant(roi)
            if debugToggle and (frameIdx % roiDebugRate == 0):
                try:
                    if roi is not None and getattr(roi, 'size', 0) > 0:
                        cv2.imwrite(os.path.join(roiDebugDir, f"{frameIdx:04d}_{key}_raw.png"), roi)
                        for vi, vimg in enumerate(varMap.get(key, [])[:2]):
                            if vimg is not None and getattr(vimg, 'size', 0) > 0:
                                cv2.imwrite(os.path.join(roiDebugDir, f"{frameIdx:04d}_{key}_proc{vi}.png"), vimg)
                except Exception as dbgE:
                    errors.append(f"Frame {frameIdx} | {key}: debug save failed: {dbgE}")
        except Exception as e:
            errors.append(f"Frame {frameIdx} | {key}: ROI build failed: {e}")

    # Second: batched fast OCR per group
    numericKeys = [k for k in (
        "temperature",
        "pressure",
        "fuel",
        "rodInsertion",
        "waterLevel",
        "feedwaterFlow",
        "fwpFlowRate1",
        "fwpUtilization1",
        "fwpRpm1",
        "fwpFlowRate2",
        "fwpUtilization2",
        "fwpRpm2",
        "totalOutput",
        "currentPowerOrder",
        "marginOfError",
        "flowRate1",
        "flowRate2",
        "rpm1",
        "rpm2",
        "valvesPct1",
        "valvesPct2",
        "vibration1",
        "vibration2"
    ) if k in roiMap and k not in cacheHits]

    stateKeys = [k for k in ("coolant",) if k in roiMap and k not in cacheHits]
    feedKeys = [k for k in ("feedwater",) if k in roiMap and k not in cacheHits]

    pre = {k: (v.get("val"), v.get("conf"), v.get("raw")) for k, v in cacheHits.items()}
    # Numeric batch (all numeric keys in one call)
    if numericKeys:
        allowNum = "0123456789.%KkPpAaWwRrMmLlSs/Nn"
        numImgs = []
        for k in numericKeys:
            img = fastMap[k]
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            numImgs.append(img)
        numResLists = readtextMulti(numImgs, allow=allowNum)
        for k, res in zip(numericKeys, numResLists):
            bestVal, bestConf, bestTxt = None, 0.0, ""
            for (_box, txt, conf) in res:
                txt2 = (txt or "").translate(digitFixSafe)
                v = _extractValueForKey(k, txt2)
                if v is None:
                    continue
                if float(conf) > bestConf:
                    bestConf, bestVal, bestTxt = float(conf), round(float(v), 1), txt
            pre[k] = (bestVal, bestConf, bestTxt)

    # State batch (coolant open/closed) in one call
    if stateKeys:
        allowState = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/ "
        stImgs = []
        for k in stateKeys:
            img = fastMap[k]
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            stImgs.append(img)
        stResLists = readtextMulti(stImgs, allow=allowState)
        for k, res in zip(stateKeys, stResLists):
            openSum, closedSum, bestC, bestT = 0.0, 0.0, 0.0, ""
            for (_box, txt, conf) in res:
                t = (txt or "").upper().replace('0','O').replace('1','I')
                if 'OPEN' in t or t.startswith('PEN') or ' QPEN' in (' ' + t) or ' PEN' in (' ' + t):
                    openSum += float(conf)
                if 'CLOSED' in t:
                    closedSum += float(conf)
                if float(conf) > bestC:
                    bestC, bestT = float(conf), t
            foundOpen = (openSum >= closedSum) and (openSum > 0.0)
            pre[k] = (1 if foundOpen else 0, min(0.99, bestC), bestT)

    # Feedwater batch (0/1/2) in one call
    if feedKeys:
        import re as _re2
        pat = _re2.compile(r"([0-2])\s*/\s*2")
        allowFeed = "0123456789/ "
        fwImgs = []
        for k in feedKeys:
            img = fastMap[k]
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            fwImgs.append(img)
        fwResLists = readtextMulti(fwImgs, allow=allowFeed)
        for k, res in zip(feedKeys, fwResLists):
            bestC, bestT, bestN = 0.0, "", None
            for (_box, txt, conf) in res:
                t2 = "".join(ch for ch in (txt or "") if ch.isdigit() or ch == '/')
                m = pat.search(t2)
                if not m:
                    continue
                n = int(m.group(1))
                if 0 <= n <= 2 and float(conf) > bestC:
                    bestC, bestT, bestN = float(conf), t2, n
            pre[k] = (bestN, min(0.99, bestC), bestT)

    # --- Third: per-key acceptance + fallbacks and per-field confidence capture ---
    for key in roiMap.keys():   # <--- IMPORTANT FIX: only process keys that have real ROIs
        try:
            roi = roiMap[key]
            variants = varMap.get(key, [])
            conf = 0.0
            rawt = ""
            v = None

            # --------------------------
            # COOLANT STATE (binary)
            # --------------------------
            if key == "coolant":
                stateOpen, conf, rawt = pre.get(key, (None,0.0,""))
                if stateOpen is None or conf < 0.20:
                    if not variants:
                        variants = preprocessVariantsSimple(roi)
                        varMap[key] = variants
                    stateOpen, conf, rawt = ocrStateVariantsSimple(variants, ["OPEN","PEN","QPEN"])

                prevState = stateBuffers.get("coolant", [])[-1] if stateBuffers["coolant"] else 1
                stableVal = (1 if stateOpen else 0) if conf >= 0.20 else prevState
                v = smoothState("coolant", stableVal)

                row[key] = v
                row["_raw_"+key] = rawt
                row["_conf_"+key] = float(conf)
                rawVals[key] = rawt
                roiHashCache[key] = {"hash": roiHashes.get(key), "val": row[key], "conf": row["_conf_"+key], "raw": row["_raw_"+key]}
                continue

            # --------------------------
            # FEEDWATER HEADERS (0–2)
            # --------------------------
            if key == "feedwater":
                count, conf, rawt = pre.get(key, (None,0.0,""))
                if count is None or conf < 0.20:
                    if not variants:
                        variants = preprocessVariantsSimple(roi)
                        varMap[key] = variants
                    count, conf, rawt = ocrFeedwaterCountFromVariants(roi, variants)

                if count is None:
                    prev = stateBuffers["feedwater"][-1] if stateBuffers["feedwater"] else 0
                    use = prev
                else:
                    use = max(0, min(2, int(count)))

                v = smoothMultistate("feedwater", use)

                row[key] = v
                row["_raw_"+key] = f"{v}/2 ACTIVE"
                row["_conf_"+key] = float(conf)
                rawVals[key] = rawt
                roiHashCache[key] = {"hash": roiHashes.get(key), "val": row[key], "conf": row["_conf_"+key], "raw": row["_raw_"+key]}
                continue

            # --------------------------
            # WATER LEVEL (%)
            # debounced + anti-spike logic
            # --------------------------
            if key == "waterLevel":
                txt = pre.get(key, (None,0.0,""))[2]
                conf = pre.get(key, (None,0.0,""))[1]
                prevVal = lastOk.get("waterLevel")
                val = parseWaterLevel(txt, prevVal)

                # fallback to strict OCR
                if val is None or conf < 0.20:
                    if not variants:
                        variants = preprocessVariantsSimple(roi)
                        varMap[key] = variants
                    bestV, bestC, bestT = None, 0.0, ""
                    for img in variants:
                        if len(img.shape)==2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        res = reader.readtext(img, detail=1, paragraph=False, allowlist="0123456789.%")
                        for (_b,t,c) in res:
                            pv = parseWaterLevel(t, prevVal)
                            if pv is not None and c > bestC:
                                bestV,bestC,bestT = pv,float(c),t
                    if bestV is not None:
                        val, conf, txt = bestV, bestC, bestT

                # debouncing large jumps
                prev = prevVal
                if val is not None and prev is not None and abs(val - prev) > maxDeltaWaterPerFrame:
                    pu = pendingUpdates["waterLevel"]
                    if pu["cand"] is not None and abs(val - pu["cand"]) <= 0.1:
                        pu["count"] += 1
                    else:
                        pu["cand"], pu["count"] = val, 1
                    if pu["count"] >= 2:
                        prev = val
                        pendingUpdates["waterLevel"] = {"cand":None,"count":0}
                    val = prev
                else:
                    pendingUpdates["waterLevel"] = {"cand":None,"count":0}

                if val is not None:
                    if val != lastOk.get("waterLevel"):
                        lastChangeTime["waterLevel"] = timestamp
                    lastOk["waterLevel"] = val
                    lastConf["waterLevel"] = float(conf)
                    lastRaw["waterLevel"] = txt

                row[key] = val
                row["_raw_"+key] = txt
                row["_conf_"+key] = float(conf)
                rawVals[key] = txt
                roiHashCache[key] = {"hash": roiHashes.get(key), "val": row[key], "conf": row["_conf_"+key], "raw": row["_raw_"+key]}
                continue

            # --------------------------
            # FEEDWATER FLOW (0 / 0.91 / 1.83)
            # with 2-frame confirmation
            # --------------------------
            if key == "feedwaterFlow":
                txt = pre.get(key, (None,0.0,""))[2]
                conf = pre.get(key, (None,0.0,""))[1]
                val = parseFeedwaterFlow(txt)

                # fallback OCR
                if val is None or conf < 0.20:
                    if not variants:
                        variants = preprocessVariantsSimple(roi)
                        varMap[key] = variants
                    bestV,bestC,bestT = None, 0.0, ""
                    for img in variants:
                        if len(img.shape)==2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        res = reader.readtext(img, detail=1, paragraph=False, allowlist="0123456789.")
                        for (_b,t,c) in res:
                            pv = parseFeedwaterFlow(t)
                            if pv is not None and c > bestC:
                                bestV,bestC,bestT = pv,float(c),t
                    if bestV is not None:
                        val, conf, txt = bestV, bestC, bestT

                # two-frame confirmation
                prev = lastOk.get("feedwaterFlow")
                if prev is not None and val is not None and val != prev:
                    pu = pendingUpdates["feedwaterFlow"]
                    if pu["cand"] is not None and abs(val - pu["cand"]) < 1e-3:
                        pu["count"] += 1
                    else:
                        pu["cand"], pu["count"] = val, 1
                    if pu["count"] >= 2:
                        prev = val
                        pendingUpdates["feedwaterFlow"] = {"cand":None,"count":0}
                    val = prev
                else:
                    pendingUpdates["feedwaterFlow"] = {"cand":None,"count":0}

                if val is not None:
                    if val != lastOk.get("feedwaterFlow"):
                        lastChangeTime["feedwaterFlow"] = timestamp
                    lastOk["feedwaterFlow"] = val
                    lastConf["feedwaterFlow"] = float(conf)
                    lastRaw["feedwaterFlow"] = txt

                row[key] = val
                row["_raw_"+key] = txt
                row["_conf_"+key] = float(conf)
                rawVals[key] = txt
                roiHashCache[key] = {"hash": roiHashes.get(key), "val": row[key], "conf": row["_conf_"+key], "raw": row["_raw_"+key]}
                continue

            # --------------------------
            # NUMERIC FIELDS
            # temperature, pressure, fuel, rodInsertion
            # --------------------------
            vParsed, conf, rawt = pre.get(key, (None,0.0,""))
            if vParsed is None or conf < confThresh.get(key, 0.0):
                if not variants:
                    variants = preprocessVariantsSimple(roi)
                    varMap[key] = variants
                vParsed, conf, rawt = ocrNumericForKey(key, variants)

            v = clampOrNan(vParsed, key)
            if key in ["fuel","rodInsertion"] and v is not None and 100.0 < v < 200.0:
                v -= 100.0
            if key in ("currentPowerOrder", "marginOfError"):
                v = applyCurrentPowerOrderRule(lastOk.get(key), v)

            lagAccepted = False
            if key in lagDetectKeys:
                prevLag = lastOk.get(key)
                if v is not None and prevLag is not None and v != prevLag:
                    lastChange = lastChangeTime.get(key)
                    if lastChange is None:
                        lastChange = timestamp
                        lastChangeTime[key] = timestamp
                    holdSec = float(timestamp - lastChange)
                    thresh = lagJumpThreshold.get(key)
                    if holdSec >= lagHoldSeconds and thresh is not None and abs(v - prevLag) >= thresh:
                        pl = pendingLag[key]
                        if pl["cand"] is not None and abs(v - pl["cand"]) <= jumpStabilityThresh.get(key, thresh):
                            pl["count"] += 1
                        else:
                            pl["cand"], pl["count"] = v, 1
                        if pl["count"] >= 2:
                            v = pl["cand"]
                            pendingLag[key] = {"cand": None, "count": 0}
                            lagAccepted = True
                        else:
                            v = prevLag
                    else:
                        pendingLag[key] = {"cand": None, "count": 0}
                else:
                    pendingLag[key] = {"cand": None, "count": 0}

            # smoothing using last_ok + jump filters
            if key in maxJumpPerFrame and not lagAccepted:
                prev = lastOk[key]
                if v is not None and prev is not None and abs(v - prev) > maxJumpPerFrame[key]:
                    pj = pendingJumps[key]
                    if pj["cand"] is not None and abs(v - pj["cand"]) <= jumpStabilityThresh[key]:
                        pj["count"] += 1
                    else:
                        pj["cand"], pj["count"] = v, 1
                    requiredCount = jumpAcceptCount.get(key, 2)
                    if pj["count"] >= requiredCount and conf >= confThresh.get(key, 0.0):
                        prev = v
                        pendingJumps[key] = {"cand":None,"count":0}
                    v = prev
                else:
                    pendingJumps[key] = {"cand":None,"count":0}

            if v is not None:
                if v != lastOk.get(key):
                    lastChangeTime[key] = timestamp
                lastOk[key] = v
                lastConf[key] = float(conf)
                lastRaw[key] = rawt

            row[key] = v
            row["_raw_"+key] = rawt
            row["_conf_"+key] = float(conf)
            rawVals[key] = rawt
            roiHashCache[key] = {"hash": roiHashes.get(key), "val": row[key], "conf": row["_conf_"+key], "raw": row["_raw_"+key]}


        except Exception as e:
            errors.append(f"Frame {frameIdx} | {key}: {e}")
            row[key] = lastOk.get(key, None)
            row["_conf_"+key] = 0.0

    # Global lag-spike filter: if all numbers were static >= lagHoldSeconds
    # and current frame introduces only large jumps, treat as lag spike.
    lastAnyChange = max([t for t in prevLastChange.values() if t is not None], default=None)
    holdSec = float(timestamp - lastAnyChange) if lastAnyChange is not None else 0.0
    changedKeys = []
    bigJumpKeys = []
    for k in lastOk.keys():
        if k not in row:
            continue
        prevV = prevLastOk.get(k)
        curV = row.get(k)
        if prevV is None or curV is None:
            continue
        if curV != prevV:
            changedKeys.append(k)
            thresh = lagJumpThreshold.get(k)
            if thresh is not None and abs(curV - prevV) >= thresh:
                bigJumpKeys.append(k)

    lagSpike = (
        bool(changedKeys)
        and holdSec >= lagHoldSeconds
        and len(bigJumpKeys) == len(changedKeys)
    )

    if lagSpike:
        # Revert to previous values and metadata
        for k in lastOk.keys():
            if k in row:
                row[k] = prevLastOk.get(k)
                if f"_conf_{k}" in row:
                    row[f"_conf_{k}"] = prevLastConf.get(k)
                if f"_raw_{k}" in row:
                    row[f"_raw_{k}"] = prevLastRaw.get(k)
            if k in changedKeys:
                roiHashCache.pop(k, None)
        lastOk = prevLastOk
        lastConf = prevLastConf
        lastRaw = prevLastRaw
        lastChangeTime = prevLastChange

    if debugToggle and (frameIdx % roiDebugRate == 0):
        try:
            overlay = frame.copy()
            for k, (x1p, y1p, x2p, y2p) in roiCoords.items():
                cv2.rectangle(overlay, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
                val = row.get(k)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    disp = "NA"
                else:
                    decimals = keyDecimals.get(k, None)
                    if decimals is None and isinstance(val, (int, np.integer)):
                        disp = f"{int(val)}"
                    elif decimals is not None and isinstance(val, (int, float, np.floating)):
                        disp = f"{float(val):.{int(decimals)}f}"
                    else:
                        disp = str(val)
                label = f"{k}:{disp}"
                cv2.putText(
                    overlay,
                    label,
                    (x1p + 2, max(0, y1p - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            cv2.imwrite(os.path.join(roiDebugDir, f"{frameIdx:04d}_overlay.png"), overlay)
        except Exception as dbgE:
            errors.append(f"Frame {frameIdx}: overlay save failed: {dbgE}")

    rowBuf.append(row)
    rawBuf.append(rawVals)
    if len(rowBuf) >= streamChunk:
        _flushBuf(rowBuf, cleanedStreamPath, _streamCols, "rows")
        _flushBuf(rawBuf, rawStreamPath, _rawCols, "raw")

cap.release()

_flushBuf(rowBuf, cleanedStreamPath, _streamCols, "rows")
_flushBuf(rawBuf, rawStreamPath, _rawCols, "raw")

# --- Log errors ---
if errors:
    with open(errorLog, "w", encoding="utf-8") as f:
        f.write("\n".join(errors))
    print(f"\n{len(errors)} OCR errors logged to {errorLog}")

# --- Data Cleanup ---
if _streamState["rows"]:
    df = pd.read_csv(cleanedStreamPath)
else:
    df = pd.DataFrame(columns=_streamCols)
for col in [
    "temperature",
    "pressure",
    "fuel",
    "rodInsertion",
    "feedwater",
    "coolant",
    "waterLevel",
    "feedwaterFlow",
    "fwpFlowRate1",
    "fwpUtilization1",
    "fwpRpm1",
    "fwpFlowRate2",
    "fwpUtilization2",
    "fwpRpm2",
    "totalOutput",
    "currentPowerOrder",
    "marginOfError",
    "flowRate1",
    "flowRate2",
    "rpm1",
    "rpm2",
    "valvesPct1",
    "valvesPct2",
    "vibration1",
    "vibration2",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for key, (lo, hi) in ranges.items():
    if key in df:
        df.loc[(df[key] < lo) | (df[key] > hi), key] = np.nan

"""
Preserve exact reads where confidence is high. We compute smoothing
and limited interpolation, then restore original values on rows where
the corresponding _conf_ value is above the per-key threshold.
"""

# Capture originals and confidence masks before smoothing
origSeries = {}
confMasks = {}
for key in ["temperature", "pressure", "fuel", "rodInsertion"]:
    if key in df.columns:
        origSeries[key] = df[key].copy()
        confCol = f"_conf_{key}"
        if confCol in df.columns:
            confVals = pd.to_numeric(df[confCol], errors="coerce")
            confMasks[key] = confVals >= confThresh.get(key, 0.0)
        else:
            confMasks[key] = pd.Series(False, index=df.index)

for col in ["temperature", "pressure"]:
    if col in df.columns:
        df[col] = df[col].rolling(5, min_periods=1, center=True).median()
for col in ["fuel", "rodInsertion"]:
    if col in df.columns:
        df[col] = df[col].rolling(3, min_periods=1, center=True).median()

# Tone down interpolation to avoid propagating errors too far
# - Do not interpolate fuel; allow short forward/back fills only
if "fuel" in df.columns:
    df["fuel"] = df["fuel"].ffill(limit=2).bfill(limit=1)

# - Temperature/pressure: allow short linear interpolation for gaps up to 2
for col in ["temperature", "pressure"]:
    if col in df.columns:
        df[col] = df[col].interpolate(method="linear", limit=2, limit_direction="forward").ffill(limit=2).bfill(limit=1)

# - Rod insertion: keep small smoothing already applied; avoid bridging long NaN runs
if "rodInsertion" in df.columns:
    df["rodInsertion"] = df["rodInsertion"].ffill(limit=2).bfill(limit=1)

# If the last fuel value is missing, carry forward the last known value
if "fuel" in df.columns and len(df) > 0:
    try:
        if pd.isna(df["fuel"].iloc[-1]):
            df.loc[df.index[-1], "fuel"] = df["fuel"].ffill().iloc[-1]
    except Exception:
        pass

# Restore original values for high-confidence rows (where available)
for key in ["temperature", "pressure", "fuel", "rodInsertion"]:
    if key in df.columns and key in origSeries:
        maskOk = confMasks.get(key, pd.Series(False, index=df.index)) & origSeries[key].notna()
        df.loc[maskOk, key] = origSeries[key][maskOk]

# Enforce per-key quantization
for key, decimals in keyDecimals.items():
    if key in df.columns and decimals is not None:
        df[key] = df[key].round(decimals)

# --- Save Clean Data ---
outputPath = os.path.join(os.path.dirname(videoPath), outputCsv)
df.to_csv(outputPath, index=False)

print(f"\nCleaned data saved to {outputPath}")

# --- Save Raw Data (no smoothing/interpolation) ---
rawOutputPath = None
try:
    if _streamState["raw"]:
        dfRaw = pd.read_csv(rawStreamPath)
        for col in [
            "temperature",
            "pressure",
            "fuel",
            "rodInsertion",
            "feedwater",
            "coolant",
            "waterLevel",
            "feedwaterFlow",
            "fwpFlowRate1",
            "fwpUtilization1",
            "fwpRpm1",
            "fwpFlowRate2",
            "fwpUtilization2",
            "fwpRpm2",
            "totalOutput",
            "currentPowerOrder",
            "marginOfError",
            "flowRate1",
            "flowRate2",
            "rpm1",
            "rpm2",
            "valvesPct1",
            "valvesPct2",
            "vibration1",
            "vibration2",
        ]:
            if col in dfRaw.columns:
                dfRaw[col] = pd.to_numeric(dfRaw[col], errors="coerce")
        rawOutputPath = os.path.join(os.path.dirname(videoPath), rawOutputCsv)
        dfRaw.to_csv(rawOutputPath, index=False)
        print(f"Raw data (no interpolation) saved to {rawOutputPath}")
except Exception as e:
    print(f"Warning: failed to write raw CSV: {e}")

# --- Optional: Invoke resuscitation on low-confidence (_conf_*) fields ---
try:
    scriptPath = os.path.join(os.path.dirname(__file__), "dataResuscitation.py")
    if os.path.isfile(scriptPath):
        # Build ROI args from current config
        def _fmtRoi(t):
            return ",".join(str(int(v)) for v in t)
        resuscArgs = [
            "--video", videoPath,
            "--cleaned", outputPath,
            "--fps", str(float(fps)),
            "--roi-temperature", _fmtRoi(regions.get("temperature", (0,0,0,0))),
            "--roi-pressure", _fmtRoi(regions.get("pressure", (0,0,0,0))),
            "--roi-fuel", _fmtRoi(regions.get("fuel", (0,0,0,0))),
            "--roi-rod", _fmtRoi(regions.get("rodInsertion", (0,0,0,0))),
            "--roi-coolant", _fmtRoi(regions.get("coolant", (0,0,0,0))),
            "--roi-feedwater", _fmtRoi(regions.get("feedwater", (0,0,0,0))),
            "--roi-fwp-flow-rate1", _fmtRoi(regions.get("fwpFlowRate1", (0,0,0,0))),
            "--roi-fwp-utilization1", _fmtRoi(regions.get("fwpUtilization1", (0,0,0,0))),
            "--roi-fwp-rpm1", _fmtRoi(regions.get("fwpRpm1", (0,0,0,0))),
            "--roi-fwp-flow-rate2", _fmtRoi(regions.get("fwpFlowRate2", (0,0,0,0))),
            "--roi-fwp-utilization2", _fmtRoi(regions.get("fwpUtilization2", (0,0,0,0))),
            "--roi-fwp-rpm2", _fmtRoi(regions.get("fwpRpm2", (0,0,0,0))),
            "--roi-total-output", _fmtRoi(regions.get("totalOutput", (0,0,0,0))),
            "--roi-current-power-order", _fmtRoi(regions.get("currentPowerOrder", (0,0,0,0))),
            "--roi-margin-of-error", _fmtRoi(regions.get("marginOfError", (0,0,0,0))),
            "--roi-flow-rate1", _fmtRoi(regions.get("flowRate1", (0,0,0,0))),
            "--roi-flow-rate2", _fmtRoi(regions.get("flowRate2", (0,0,0,0))),
            "--roi-rpm1", _fmtRoi(regions.get("rpm1", (0,0,0,0))),
            "--roi-rpm2", _fmtRoi(regions.get("rpm2", (0,0,0,0))),
            "--roi-valves-pct1", _fmtRoi(regions.get("valvesPct1", (0,0,0,0))),
            "--roi-valves-pct2", _fmtRoi(regions.get("valvesPct2", (0,0,0,0))),
            "--roi-vibration1", _fmtRoi(regions.get("vibration1", (0,0,0,0))),
            "--roi-vibration2", _fmtRoi(regions.get("vibration2", (0,0,0,0))),
        ]
        # Pass raw CSV if available
        if rawOutputPath and os.path.isfile(rawOutputPath):
            resuscArgs += ["--raw", rawOutputPath]
        print("\nLaunching resuscitation on low-confidence (_conf_*) fields...\n")
        if getattr(sys, "frozen", False):
            import importlib.util
            oldArgv = sys.argv[:]
            try:
                sys.argv = [scriptPath] + resuscArgs
                spec = importlib.util.spec_from_file_location("dataResuscitation", scriptPath)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "main"):
                    mod.main()
            except Exception as e:
                print(f"Warning: failed to run resuscitation in-process: {e}")
            finally:
                sys.argv = oldArgv
        else:
            args = [sys.executable, scriptPath] + resuscArgs
            subprocess.run(args, check=False)

        # dataResuscitation now writes directly to cleaned CSV.
        # Strip debug columns from cleaned output if DEBUG_TOGGLE is off.
        if not debugToggle:
            try:
                dfClean = pd.read_csv(outputPath)
                # Drop debug-only columns when debug is off
                dfClean = dfClean[[c for c in dfClean.columns if not c.startswith(("_raw_", "_conf_"))]]
                dfClean.to_csv(outputPath, index=False)
                print("Stripped _raw_* and _conf_* (DEBUG_TOGGLE=False).")
            except Exception as e:
                print(f"Warning: failed to strip debug columns: {e}")
    else:
        print("Note: dataResuscitation.py not found; skipping automatic re-OCR step.")
        # If debug is off, strip debug columns from cleaned CSV now
        if not debugToggle:
            try:
                dfClean = pd.read_csv(outputPath)
                dfClean = dfClean[[c for c in dfClean.columns if not c.startswith(("_raw_", "_conf_"))]]
                dfClean.to_csv(outputPath, index=False)
                print("Stripped _raw_* and _conf_* (DEBUG_TOGGLE=False).")
            except Exception as e:
                print(f"Warning: failed to strip debug columns: {e}")
except Exception as e:
    print(f"Warning: failed to launch resuscitation: {e}")

# --- Auto-run features builder (unified features + anomaly flags) ---
try:
    fbScript = os.path.join(os.path.dirname(__file__), "featuresBuilder.py")
    if os.path.isfile(fbScript):
        fbArgs = [
            "--input", outputPath,
            "--slope-window", "3",
            "--min-fuel", "75",
            "--rod", "55",
            "--rod-tol", "5",
            "--rv-k-per-s", "7.5",
            # Keep reasonable anomaly thresholds; can be tuned later
            "--rise-k-per-s", "10.0",
            "--dip-k-per-s", "10.0",
            "--require-coolant-open",
        ]
        print("\nBuilding features and anomaly flags via featuresBuilder.py...\n")
        if getattr(sys, "frozen", False):
            import importlib.util
            oldArgv = sys.argv[:]
            try:
                sys.argv = [fbScript] + fbArgs
                spec = importlib.util.spec_from_file_location("featuresBuilder", fbScript)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "main"):
                    mod.main()
            except Exception as e:
                print(f"Warning: failed to run featuresBuilder in-process: {e}")
            finally:
                sys.argv = oldArgv
        else:
            argsFb = [sys.executable, fbScript] + fbArgs
            subprocess.run(argsFb, check=False)
    else:
        print("Note: featuresBuilder.py not found; skipping features build.")
except Exception as e:
    print(f"Warning: failed to run features builder: {e}")

for _p in (cleanedStreamPath, rawStreamPath):
    try:
        if _p and os.path.isfile(_p):
            os.remove(_p)
    except Exception:
        pass
