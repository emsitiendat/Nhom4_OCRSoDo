import cv2
import csv
import os
import re
import shutil
import time
import unicodedata
import numpy as np
from pathlib import Path
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = 'False'
from paddleocr import PaddleOCR
RAW_DIR = Path('dataset/raw')
REAL_DIR = Path('dataset/real')
TAMPER_DIR = Path('dataset/tampered')
REVIEW_DIR = Path('dataset/review_tuan1')
META_FILE = Path('dataset/metadata_tuan1.csv')
MAX_FILES = None
WEEK1_RUN_INFO = Path('dataset/week1_run_info.txt')
OCR_MIN_SCORE = 0.35
CLEAR_OLD_OUTPUT = True
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
FAKE_TEXT_COLOR = (220, 70, 20)
FAKE_DIAGRAM_COLOR = (255, 80, 0)
SYNTHETIC_WATERMARK = 'SYNTHETIC TAMPER'
MIN_RED_PIXELS = 250
PII_KEYWORDS = ['ho va ten', 'ho ten', 'thuong tru', 'noi thuong tru', 'dia chi thuong tru', 'cmnd', 'cccd', 'chung minh nhan dan', 'can cuoc cong dan', 'giay to nhan than']
PROTECTED_KEYWORDS = ['thua dat so', 'so thua', 'to ban do', 'dien tich', 'dia chi', 'muc dich su dung', 'hinh thuc su dung', 'thoi han su dung', 'nguon goc su dung']
FIELD_KEYWORDS = {'dien_tich': ['dien tich'], 'so_thua': ['thua dat so', 'so thua'], 'to_ban_do': ['to ban do so', 'to ban do']}

def tao_thu_muc():
    for folder in [REAL_DIR, TAMPER_DIR, REVIEW_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    if CLEAR_OLD_OUTPUT:
        for folder in [REAL_DIR, TAMPER_DIR, REVIEW_DIR]:
            for p in folder.iterdir():
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
        if META_FILE.exists():
            META_FILE.unlink()

def doc_anh(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def luu_anh(path, img, jpg=False):
    if jpg:
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    if ok:
        buf.tofile(str(path))
    return ok

def lay_ds_anh():
    files = sorted([p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS], key=lambda p: p.name.lower())
    runtime_limit = os.environ.get('WEEK1_MAX_FILES', '').strip()
    limit = MAX_FILES
    if runtime_limit:
        try:
            parsed = int(runtime_limit)
            if parsed > 0:
                limit = parsed
        except ValueError:
            pass
    if limit is not None:
        files = files[:limit]
    return files

def bo_dau(text):
    text = str(text).lower()
    text = text.replace('đ', 'd')
    text = unicodedata.normalize('NFD', text)
    text = ''.join((c for c in text if unicodedata.category(c) != 'Mn'))
    text = re.sub('\\s+', ' ', text).strip()
    return text

def khoi_tao_ocr():
    print('dang khoi tao PaddleOCR...')
    return PaddleOCR(lang='vi', device='cpu', enable_mkldnn=False, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)

def box_xyxy_sang_xywh(box):
    arr = np.asarray(box).reshape(-1)
    if len(arr) < 4:
        return None
    x1 = int(arr[0])
    y1 = int(arr[1])
    x2 = int(arr[2])
    y2 = int(arr[3])
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]

def polygon_sang_xywh(poly):
    arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x1 = int(np.floor(arr[:, 0].min()))
    y1 = int(np.floor(arr[:, 1].min()))
    x2 = int(np.ceil(arr[:, 0].max()))
    y2 = int(np.ceil(arr[:, 1].max()))
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]

def parse_paddle_v3(result):
    items = []
    if result is None:
        return items
    for res in result:
        try:
            texts = res['rec_texts']
            scores = res['rec_scores']
            try:
                boxes = res['rec_boxes']
                use_xyxy = True
            except Exception:
                boxes = res['rec_polys']
                use_xyxy = False
            n = min(len(texts), len(scores), len(boxes))
            for i in range(n):
                text = str(texts[i]).strip()
                try:
                    score = float(scores[i])
                except Exception:
                    score = 1.0
                if not text:
                    continue
                if score < OCR_MIN_SCORE:
                    continue
                if use_xyxy:
                    box = box_xyxy_sang_xywh(boxes[i])
                else:
                    box = polygon_sang_xywh(boxes[i])
                if box is None:
                    continue
                items.append({'text': text, 'score': score, 'box': box})
        except Exception as e:
            print('loi parse OCR:', e)
    return sorted(items, key=lambda z: (z['box'][1], z['box'][0]))

def chay_ocr(ocr, img):
    try:
        result = ocr.predict(img)
        return parse_paddle_v3(result)
    except Exception as e:
        print('predict() loi:', e)
        return []

def kep_box(box, img_shape, pad=5):
    x, y, w, h = box
    H, W = img_shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]

def cat_roi(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w]

def them_watermark(img):
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(w, h) / 1200)
    thickness = 2
    (tw, th), _ = cv2.getTextSize(SYNTHETIC_WATERMARK, font, scale, thickness)
    x = max(10, w - tw - 20)
    y = max(th + 10, h - 20)
    overlay = out.copy()
    cv2.rectangle(overlay, (x - 8, y - th - 8), (x + tw + 8, y + 8), (255, 255, 255), -1)
    out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
    cv2.putText(out, SYNTHETIC_WATERMARK, (x, y), font, scale, (0, 0, 200), thickness, cv2.LINE_AA)
    return out

def la_truong_duoc_bao_ve(text):
    norm = bo_dau(text)
    return any((key in norm for key in PROTECTED_KEYWORDS))

def la_pii_line(text):
    norm = bo_dau(text)
    if la_truong_duoc_bao_ve(text):
        return False
    if any((key in norm for key in PII_KEYWORDS)):
        return True
    if re.search('^\\s*(ong|ba)\\b', norm):
        return True
    if re.search('\\b\\d{9,12}\\b', norm):
        return True
    return False

def cung_dong(a, b):
    _, y1, _, h1 = a['box']
    _, y2, _, h2 = b['box']
    c1 = y1 + h1 / 2
    c2 = y2 + h2 / 2
    return abs(c1 - c2) <= max(h1, h2) * 0.9

def tim_pii_boxes(items, img_shape):
    boxes = []
    for i, item in enumerate(items):
        text = item['text']
        if not la_pii_line(text):
            continue
        boxes.append(kep_box(item['box'], img_shape, pad=6))
        norm = bo_dau(text)
        if any((key in norm for key in ['ho va ten', 'ho ten', 'thuong tru', 'noi thuong tru', 'dia chi thuong tru'])):
            if i + 1 < len(items):
                nxt = items[i + 1]
                if not la_truong_duoc_bao_ve(nxt['text']) and cung_dong(item, nxt):
                    boxes.append(kep_box(nxt['box'], img_shape, pad=6))
    return boxes

def mosaic_boxes(img, boxes):
    out = img.copy()
    for x, y, w, h in boxes:
        roi = out[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        small_w = max(4, w // 20)
        small_h = max(4, h // 20)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
        out[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return out

def co_keyword(text, keywords):
    norm = bo_dau(text)
    return any((key in norm for key in keywords))

def lay_so_trong_text(text):
    match = re.search('\\d+(?:[.,]\\d+)?[a-zA-Z]?', text)
    if not match:
        return None
    return {'value': match.group(), 'start': match.start(), 'end': match.end()}

def sub_box_text(box, text, start, end):
    x, y, w, h = box
    n = max(1, len(text))
    sx = int(x + w * start / n)
    ex = int(x + w * end / n)
    return [sx, y, max(12, ex - sx), h]

def box_hop_le_de_tamper(box, img_shape):
    x, y, w, h = box
    H, W = img_shape[:2]
    return y + h / 2 <= H * 0.5 and w >= 10 and (h >= 5)

def tim_gia_tri_gan_label(items, label_index):
    label = items[label_index]
    lx, ly, lw, lh = label['box']
    candidates = []
    for j, item in enumerate(items):
        if j == label_index:
            continue
        if not re.search('\\d', item['text']):
            continue
        x, y, w, h = item['box']
        dy = abs(y + h / 2 - (ly + lh / 2))
        same_line = dy <= max(lh, h) * 1.5
        below = y >= ly and y - ly <= lh * 3
        if not (same_line or below):
            continue
        dx = abs(x - (lx + lw))
        candidates.append((dy + 0.25 * dx, item))
    if not candidates:
        return None
    candidates.sort(key=lambda z: z[0])
    return candidates[0][1]

def tim_target_text(items, img_shape, preferred):
    order = list(dict.fromkeys([preferred, 'dien_tich', 'so_thua', 'to_ban_do']))
    for field in order:
        for i, item in enumerate(items):
            if not box_hop_le_de_tamper(item['box'], img_shape):
                continue
            if not co_keyword(item['text'], FIELD_KEYWORDS[field]):
                continue
            number = lay_so_trong_text(item['text'])
            if number is not None:
                box = kep_box(sub_box_text(item['box'], item['text'], number['start'], number['end']), img_shape, pad=3)
                if box_hop_le_de_tamper(box, img_shape):
                    return {'field': field, 'box': box, 'old_value': number['value']}
            value_item = tim_gia_tri_gan_label(items, i)
            if value_item is None:
                continue
            if not box_hop_le_de_tamper(value_item['box'], img_shape):
                continue
            number2 = lay_so_trong_text(value_item['text'])
            old_value = number2['value'] if number2 else value_item['text']
            return {'field': field, 'box': kep_box(value_item['box'], img_shape, pad=3), 'old_value': old_value}
    return None

def doi_so(old_value, index):
    chars = list(str(old_value))
    positions = [i for i, c in enumerate(chars) if c.isdigit()]
    if not positions:
        return str(500 + index)
    pos = positions[index % len(positions)]
    old_digit = int(chars[pos])
    new_digit = (old_digit + 3) % 10
    if new_digit == old_digit:
        new_digit = (old_digit + 1) % 10
    chars[pos] = str(new_digit)
    return ''.join(chars)

def xoa_box(img, box):
    x, y, w, h = box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

def tinh_font_phu_hop(text, box, font):
    x, y, w, h = box
    max_width = max(12, int(w * 0.92))
    max_height = max(8, int(h * 0.78))
    scale = max(0.25, min(2.0, h / 28.0))
    thickness = max(1, int(round(h / 18.0)))
    while scale > 0.2:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_width and th <= max_height:
            return (scale, thickness, tw, th)
        scale -= 0.03
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    return (scale, thickness, tw, th)

def ve_text_gia(img, box, text):
    out = img.copy()
    x, y, w, h = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, tw, th = tinh_font_phu_hop(text, box, font)
    tx = x + max(1, (w - tw) // 2)
    baseline = y + max(th + 1, (h + th) // 2)
    baseline = min(y + h - 2, baseline)
    cv2.putText(out, text, (tx, baseline), font, scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, (tx, baseline), font, scale, FAKE_TEXT_COLOR, thickness, cv2.LINE_AA)
    return out

def tao_text_tamper(real_img, items, index):
    cycle = ['dien_tich', 'so_thua', 'to_ban_do']
    preferred = cycle[(index - 1) % len(cycle)]
    target = tim_target_text(items, real_img.shape, preferred)
    if target is None:
        return None
    fake_value = doi_so(target['old_value'], index)
    out = xoa_box(real_img, target['box'])
    out = ve_text_gia(out, target['box'], fake_value)
    out = them_watermark(out)
    return {'img': out, 'type': 'text', 'target': target['field'], 'note': f"{target['old_value']} -> {fake_value}", 'box': target['box']}

def lay_diagram_box(img_shape):
    H, W = img_shape[:2]
    return kep_box([int(W * 0.5), int(H * 0.04), int(W * 0.48), int(H * 0.48)], img_shape, pad=0)

def tim_hough_lines(img, box):
    roi = cat_roi(img, box)
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=10)

def chon_line_phu_hop(lines, roi_w, roi_h):
    if lines is None:
        return None
    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        length = np.hypot(dx, dy)
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        if mx < roi_w * 0.15:
            continue
        if mx > roi_w * 0.85:
            continue
        if my < roi_h * 0.15:
            continue
        if my > roi_h * 0.8:
            continue
        min_len = min(roi_w, roi_h) * 0.15
        if length < min_len:
            continue
        max_len = max(roi_w, roi_h) * 0.75
        if length > max_len:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle
        if angle < 5 and length > roi_w * 0.5:
            continue
        cx = roi_w / 2
        cy = roi_h / 2
        center_dist = np.hypot(mx - cx, my - cy)
        score = length - 0.35 * center_dist
        candidates.append((score, (x1, y1, x2, y2)))
    if not candidates:
        return None
    candidates.sort(key=lambda z: z[0], reverse=True)
    return candidates[0][1]

def tao_diagram_tamper(real_img, index):
    box = lay_diagram_box(real_img.shape)
    x, y, w, h = box
    lines = tim_hough_lines(real_img, box)
    best = chon_line_phu_hop(lines, w, h)
    if best is None:
        return None
    out = real_img.copy()
    mode = (index - 1) % 3
    x1, y1, x2, y2 = best
    x1 += x
    x2 += x
    y1 += y
    y2 += y
    dx = x2 - x1
    dy = y2 - y1
    if mode == 0:
        note = 'line_break'
        p1 = (int(x1 + dx * 0.42), int(y1 + dy * 0.42))
        p2 = (int(x1 + dx * 0.58), int(y1 + dy * 0.58))
        cv2.line(out, p1, p2, (245, 245, 245), thickness=6, lineType=cv2.LINE_AA)
        target_box = [min(p1[0], p2[0]) - 5, min(p1[1], p2[1]) - 5, abs(p2[0] - p1[0]) + 10, abs(p2[1] - p1[1]) + 10]
    elif mode == 1:
        note = 'parallel_line_add'
        length = np.hypot(dx, dy)
        if length <= 1:
            return None
        nx = -dy / length
        ny = dx / length
        offset = min(18, max(8, int(min(w, h) * 0.035)))
        sx1 = int(x1 + nx * offset)
        sy1 = int(y1 + ny * offset)
        sx2 = int(x2 + nx * offset)
        sy2 = int(y2 + ny * offset)
        cv2.line(out, (sx1, sy1), (sx2, sy2), FAKE_DIAGRAM_COLOR, thickness=3, lineType=cv2.LINE_AA)
        target_box = [min(sx1, sx2) - 5, min(sy1, sy2) - 5, abs(sx2 - sx1) + 10, abs(sy2 - sy1) + 10]
    else:
        note = 'short_line_add'
        cx = int(x1 + dx * 0.35)
        cy = int(y1 + dy * 0.35)
        length = np.hypot(dx, dy)
        if length <= 1:
            return None
        ux = dx / length
        uy = dy / length
        nx = -uy
        ny = ux
        seg_len = max(25, int(length * 0.25))
        offset = max(10, int(min(w, h) * 0.04))
        center_x = int(cx + nx * offset)
        center_y = int(cy + ny * offset)
        sx1 = int(center_x - ux * seg_len / 2)
        sy1 = int(center_y - uy * seg_len / 2)
        sx2 = int(center_x + ux * seg_len / 2)
        sy2 = int(center_y + uy * seg_len / 2)
        cv2.line(out, (sx1, sy1), (sx2, sy2), FAKE_DIAGRAM_COLOR, thickness=3, lineType=cv2.LINE_AA)
        target_box = [min(sx1, sx2) - 5, min(sy1, sy2) - 5, abs(sx2 - sx1) + 10, abs(sy2 - sy1) + 10]
    target_box = kep_box(target_box, real_img.shape, pad=2)
    out = them_watermark(out)
    return {'img': out, 'type': 'diagram', 'target': 'diagram_roi', 'note': note, 'box': box, 'anomaly_box': target_box}

def lay_seal_search_box(img_shape):
    H, W = img_shape[:2]
    return kep_box([0, int(H * 0.58), int(W * 0.6), int(H * 0.42)], img_shape, pad=0)

def tao_red_mask(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 60, 45], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array([165, 60, 45], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(mask1, mask2)
    return mask

def tim_seal_data(img):
    search_box = lay_seal_search_box(img.shape)
    sx, sy, sw, sh = search_box
    roi = img[sy:sy + sh, sx:sx + sw]
    if roi.size == 0:
        return None
    red_mask = tao_red_mask(roi)
    if np.count_nonzero(red_mask) < MIN_RED_PIXELS:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    grouped = cv2.dilate(red_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    best = contours[0]
    bx, by, bw, bh = cv2.boundingRect(best)
    if bw < 20 or bh < 20:
        return None
    seal_box = kep_box([sx + bx, sy + by, bw, bh], img.shape, pad=8)
    x, y, w, h = seal_box
    seal_roi = img[y:y + h, x:x + w]
    if seal_roi.size == 0:
        return None
    seal_mask = tao_red_mask(seal_roi)
    if np.count_nonzero(seal_mask) < MIN_RED_PIXELS:
        return None
    return {'box': seal_box, 'mask': seal_mask}

def tao_seal_tamper(real_img, index):
    data = tim_seal_data(real_img)
    if data is None:
        return None
    box = data['box']
    red_mask = data['mask']
    x, y, w, h = box
    seal_roi = real_img[y:y + h, x:x + w].copy()
    if seal_roi.size == 0:
        return None
    out = real_img.copy()
    mode = (index - 1) % 3
    if mode == 0:
        note = 'seal_partial_remove'
        H, W = red_mask.shape
        region_mask = np.zeros_like(red_mask)
        region_mask[:, int(W * 0.58):W] = 255
        target_mask = cv2.bitwise_and(red_mask, region_mask)
        gray = cv2.cvtColor(seal_roi, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        seal_roi[target_mask > 0] = gray_bgr[target_mask > 0]
        anomaly_box = [x + int(w * 0.55), y, int(w * 0.45), h]
    elif mode == 1:
        note = 'seal_local_blur'
        H, W = red_mask.shape
        region_mask = np.zeros_like(red_mask)
        x1 = int(W * 0.2)
        x2 = int(W * 0.8)
        y1 = int(H * 0.2)
        y2 = int(H * 0.8)
        region_mask[y1:y2, x1:x2] = 255
        target_mask = cv2.bitwise_and(red_mask, region_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        target_mask = cv2.dilate(target_mask, kernel, iterations=1)
        blurred = cv2.GaussianBlur(seal_roi, (13, 13), 0)
        seal_roi[target_mask > 0] = blurred[target_mask > 0]
        anomaly_box = [x + x1, y + y1, x2 - x1, y2 - y1]
    else:
        note = 'seal_color_fade'
        hsv = cv2.cvtColor(seal_roi, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        saturation[red_mask > 0] *= 0.2
        hsv[:, :, 1] = np.clip(saturation, 0, 255).astype(np.uint8)
        seal_roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        anomaly_box = box
    out[y:y + h, x:x + w] = seal_roi
    out = them_watermark(out)
    return {'img': out, 'type': 'seal', 'target': 'seal_roi', 'note': note, 'box': box, 'anomaly_box': anomaly_box}

def resize_preview(img, max_w=420):
    h, w = img.shape[:2]
    if w <= max_w:
        return img.copy()
    scale = max_w / w
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def dat_nhan(img, text):
    out = img.copy()
    cv2.rectangle(out, (8, 8), (min(out.shape[1] - 5, 20 + 15 * len(text)), 44), (255, 255, 255), -1)
    cv2.putText(out, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    return out

def tao_review(raw, real, pii_boxes, text_result, diagram_result, seal_result, index):
    marked = raw.copy()
    for x, y, w, h in pii_boxes:
        cv2.rectangle(marked, (x, y), (x + w, y + h), (255, 0, 0), 3)
    if text_result is not None:
        x, y, w, h = text_result['box']
        cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 3)
    if diagram_result is not None:
        x, y, w, h = diagram_result['box']
        cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 255, 255), 3)
    if seal_result is not None:
        x, y, w, h = seal_result['box']
        cv2.rectangle(marked, (x, y), (x + w, y + h), (255, 0, 255), 3)
    panels = [dat_nhan(resize_preview(marked), 'RAW + ROI'), dat_nhan(resize_preview(real), 'REAL')]
    for result, label in [(text_result, 'FAKE TEXT'), (diagram_result, 'FAKE DIAGRAM'), (seal_result, 'FAKE SEAL')]:
        if result is not None:
            panels.append(dat_nhan(resize_preview(result['img']), label))
    max_h = max((p.shape[0] for p in panels))
    padded = []
    for p in panels:
        diff = max_h - p.shape[0]
        if diff > 0:
            p = cv2.copyMakeBorder(p, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        padded.append(p)
    review = np.hstack(padded)
    luu_anh(REVIEW_DIR / f'review_{index:03d}.jpg', review, jpg=True)

def ghi_metadata(rows):
    fields = ['source_index', 'raw_name', 'real_name', 'output_name', 'label', 'tamper_type', 'tamper_target', 'tamper_note']
    with open(META_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def dinh_dang_thoi_gian(seconds):
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f'{hours:d}h {minutes:02d}m {seconds:02d}s'
    if minutes > 0:
        return f'{minutes:d}m {seconds:02d}s'
    return f'{seconds:d}s'

def main():
    start_all = time.perf_counter()
    print('=' * 64, flush=True)
    print('TUAN 1 - AUTO DATASET', flush=True)
    print('=' * 64, flush=True)
    print('Anonymize PII + tao synthetic tamper + review/metadata', flush=True)
    print('PaddleOCR tren CPU co the xu ly cham hon cac buoc khac.', flush=True)
    print('Neu terminal im lang trong luc OCR, runner van se bao heartbeat.', flush=True)
    print(flush=True)
    print('[KHOI TAO] Chuan bi output...', flush=True)
    tao_thu_muc()
    all_raw = sorted([p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS], key=lambda p: p.name.lower()) if RAW_DIR.exists() else []
    files = lay_ds_anh()
    if not files:
        print('[LOI] Khong co anh trong', RAW_DIR, flush=True)
        return
    total_raw = len(all_raw)
    selected = len(files)
    print(f'[KHOI TAO] Raw hien co      : {total_raw}', flush=True)
    print(f'[KHOI TAO] Se xu ly         : {selected}', flush=True)
    if selected < total_raw:
        print('[LUU Y] Dang chay SUBSET de test nhanh.', flush=True)
        print('        Metric cac tuan sau co the khac bao cao full dataset.', flush=True)
    else:
        print('[KHOI TAO] Che do           : FULL DATASET', flush=True)
    WEEK1_RUN_INFO.parent.mkdir(parents=True, exist_ok=True)
    WEEK1_RUN_INFO.write_text(f'raw_total={total_raw}\nselected={selected}\n', encoding='utf-8')
    print('[KHOI TAO] Dang khoi tao PaddleOCR...', flush=True)
    ocr = khoi_tao_ocr()
    print('[KHOI TAO] PaddleOCR san sang.', flush=True)
    rows = []
    count_real = count_text = count_diagram = count_seal = 0
    for index, path in enumerate(files, start=1):
        image_start = time.perf_counter()
        percent_before = (index - 1) / selected * 100.0
        print('-' * 64, flush=True)
        print(f'[{index}/{selected}] {percent_before:5.1f}% | {path.name}', flush=True)
        print('  [1/5] Doc anh...', flush=True)
        img = doc_anh(path)
        if img is None:
            print('  [LOI] Khong doc duoc anh - bo qua.', flush=True)
            continue
        print('  [2/5] PaddleOCR + tim PII...', flush=True)
        items = chay_ocr(ocr, img)
        pii_boxes = tim_pii_boxes(items, img.shape)
        print(f'        OCR lines: {len(items)} | PII boxes: {len(pii_boxes)}', flush=True)
        print('  [3/5] Tao Real/anonymized...', flush=True)
        real_img = mosaic_boxes(img, pii_boxes)
        real_name = f'anonymized_real_{index:03d}.png'
        luu_anh(REAL_DIR / real_name, real_img)
        count_real += 1
        rows.append({'source_index': index, 'raw_name': path.name, 'real_name': real_name, 'output_name': real_name, 'label': 'real', 'tamper_type': 'none', 'tamper_target': 'none', 'tamper_note': 'anonymized_only'})
        print('  [4/5] Tao synthetic tamper...', flush=True)
        text_result = tao_text_tamper(real_img, items, index)
        if text_result is not None:
            name = f'fake_text_{index:03d}.png'
            luu_anh(TAMPER_DIR / name, text_result['img'])
            count_text += 1
            rows.append({'source_index': index, 'raw_name': path.name, 'real_name': real_name, 'output_name': name, 'label': 'tampered', 'tamper_type': text_result['type'], 'tamper_target': text_result['target'], 'tamper_note': text_result['note']})
        diagram_result = tao_diagram_tamper(real_img, index)
        if diagram_result is not None:
            name = f'fake_diagram_{index:03d}.png'
            luu_anh(TAMPER_DIR / name, diagram_result['img'])
            count_diagram += 1
            rows.append({'source_index': index, 'raw_name': path.name, 'real_name': real_name, 'output_name': name, 'label': 'tampered', 'tamper_type': diagram_result['type'], 'tamper_target': diagram_result['target'], 'tamper_note': diagram_result['note']})
        seal_result = tao_seal_tamper(real_img, index)
        if seal_result is not None:
            name = f'fake_seal_{index:03d}.png'
            luu_anh(TAMPER_DIR / name, seal_result['img'])
            count_seal += 1
            rows.append({'source_index': index, 'raw_name': path.name, 'real_name': real_name, 'output_name': name, 'label': 'tampered', 'tamper_type': seal_result['type'], 'tamper_target': seal_result['target'], 'tamper_note': seal_result['note']})
        print('        text:', 'YES' if text_result is not None else 'NO', '| diagram:', 'YES' if diagram_result is not None else 'NO', '| seal:', 'YES' if seal_result is not None else 'NO', flush=True)
        print('  [5/5] Luu review + metadata...', flush=True)
        tao_review(img, real_img, pii_boxes, text_result, diagram_result, seal_result, index)
        ghi_metadata(rows)
        image_elapsed = time.perf_counter() - image_start
        total_elapsed = time.perf_counter() - start_all
        avg_per_image = total_elapsed / index
        remaining = max(0, selected - index)
        eta = avg_per_image * remaining
        percent_after = index / selected * 100.0
        print(f'  [XONG] {index}/{selected} | {percent_after:5.1f}% | anh nay: {dinh_dang_thoi_gian(image_elapsed)} | ETA uoc tinh: {dinh_dang_thoi_gian(eta)}', flush=True)
    total_tampered = count_text + count_diagram + count_seal
    total_elapsed = time.perf_counter() - start_all
    print('\n' + '=' * 64, flush=True)
    print('HOAN THANH TUAN 1 - AUTO', flush=True)
    print('=' * 64, flush=True)
    print('raw            :', len(files), flush=True)
    print('real           :', count_real, flush=True)
    print('tamper_text    :', count_text, flush=True)
    print('tamper_diagram :', count_diagram, flush=True)
    print('tamper_seal    :', count_seal, flush=True)
    print('tampered_total :', total_tampered, flush=True)
    print('thoi_gian      :', dinh_dang_thoi_gian(total_elapsed), flush=True)
if __name__ == '__main__':
    main()
