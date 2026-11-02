import csv
import os
import re
import shutil
import unicodedata
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.morphology import skeletonize
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = 'False'
from paddleocr import PaddleOCR
REAL_DIAGRAM_DIR = Path('dataset/processed_real/diagram')
FAKE_DIAGRAM_DIR = Path('dataset/processed_tampered/diagram_tamper/diagram')
REAL_TEXT_DIR = Path('dataset/processed_real/text')
FAKE_TEXT_DIR = Path('dataset/processed_tampered/text_tamper/text')
REAL_SEAL_DIR = Path('dataset/processed_real/seal')
FAKE_SEAL_DIR = Path('dataset/processed_tampered/seal_tamper/seal')
OUT_DIR = Path('dataset/tuan4_results')
GEOMETRY_OUT = OUT_DIR / 'geometry'
OCR_OUT = OUT_DIR / 'ocr'
SEAL_OUT = OUT_DIR / 'seal'
GEOMETRY_CSV = OUT_DIR / 'geometry_metrics.csv'
OCR_CSV = OUT_DIR / 'ocr_fields.csv'
SEAL_CSV = OUT_DIR / 'seal_metrics.csv'
SUMMARY_CSV = OUT_DIR / 'pipeline_summary.csv'
CLEAR_OLD_OUTPUT = True
MAX_GEOMETRY_PAIRS = 10
MAX_OCR_IMAGES = 10
MAX_OCR_PAIRS = 10
MAX_SEAL_PAIRS = 10
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
SHORT_LINE_MAX = 50
LONG_LINE_MIN = 100
TOP_HOUGH_LINES = 12
MAX_CORNERS = 20
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
GEOMETRY_WARNING_THR = 0.1
SEAL_WARNING_THR = 0.03
OCR_MIN_SCORE = 0.35

def doc_anh(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def luu_anh(path, img):
    path = Path(path)
    if path.suffix.lower() in ['.jpg', '.jpeg']:
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    if not ok:
        return False
    buf.tofile(str(path))
    return True

def lay_ds_anh(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS], key=lambda p: p.name.lower())

def lay_source_index(path):
    match = re.search('(\\d+)$', Path(path).stem)
    return int(match.group(1)) if match else None

def tao_map_theo_index(folder):
    result = {}
    for path in lay_ds_anh(folder):
        index = lay_source_index(path)
        if index is not None:
            result[index] = path
    return result

def gioi_han(items, max_items):
    return items if max_items is None else items[:max_items]

def tao_output():
    if CLEAR_OLD_OUTPUT and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for folder in [GEOMETRY_OUT, OCR_OUT, SEAL_OUT]:
        folder.mkdir(parents=True, exist_ok=True)

def dat_nhan(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (255, 255, 255), -1)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)
    return out

def resize_cung_chieu_cao(img1, img2):
    h = min(img1.shape[0], img2.shape[0])

    def resize(img):
        scale = h / img.shape[0]
        w = max(1, int(img.shape[1] * scale))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return (resize(img1), resize(img2))

def tach_hinh_hoc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    skeleton = skeletonize(binary > 0)
    skeleton_img = skeleton.astype(np.uint8) * 255
    lines = cv2.HoughLinesP(skeleton_img, 1, np.pi / 180, threshold=40, minLineLength=20, maxLineGap=10)
    return (skeleton_img, lines)

def line_info(line):
    x1, y1, x2, y2 = map(int, line[0])
    dx = x2 - x1
    dy = y2 - y1
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'length': float(np.hypot(dx, dy)), 'angle': float(np.degrees(np.arctan2(dy, dx)))}

def phan_tich_net(lines):
    if lines is None:
        return {'total': 0, 'short': 0, 'long': 0, 'short_ratio': 0.0, 'mean_top_length': 0.0, 'top_lines': []}
    infos = [line_info(line) for line in lines]
    infos.sort(key=lambda x: x['length'], reverse=True)
    total = len(infos)
    short_count = sum((x['length'] < SHORT_LINE_MAX for x in infos))
    long_count = sum((x['length'] >= LONG_LINE_MIN for x in infos))
    top_lines = infos[:TOP_HOUGH_LINES]
    mean_top_length = float(np.mean([x['length'] for x in top_lines])) if top_lines else 0.0
    return {'total': total, 'short': short_count, 'long': long_count, 'short_ratio': short_count / total if total else 0.0, 'mean_top_length': mean_top_length, 'top_lines': top_lines}

def giao_diem_hai_line(a, b):
    x1, y1, x2, y2 = (a['x1'], a['y1'], a['x2'], a['y2'])
    x3, y3, x4, y4 = (b['x1'], b['y1'], b['x2'], b['y2'])
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-06:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (float(px), float(py))

def goc_chenh_lech(a1, a2):
    diff = abs(a1 - a2) % 180
    return min(diff, 180 - diff)

def gan_diem(points, point, min_dist=15):
    return any((np.hypot(point[0] - p[0], point[1] - p[1]) < min_dist for p in points))

def tim_goc_ung_vien(top_lines, img_shape):
    h, w = img_shape[:2]
    corners = []
    for i in range(len(top_lines)):
        for j in range(i + 1, len(top_lines)):
            a = top_lines[i]
            b = top_lines[j]
            if goc_chenh_lech(a['angle'], b['angle']) < 20:
                continue
            p = giao_diem_hai_line(a, b)
            if p is None:
                continue
            x, y = p
            if not (0 <= x < w and 0 <= y < h):
                continue
            point = (int(round(x)), int(round(y)))
            if not gan_diem(corners, point):
                corners.append(point)
            if len(corners) >= MAX_CORNERS:
                return corners
    return corners

def ve_geometry(img, stats, corners):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    for line in stats['top_lines']:
        cv2.line(out, (line['x1'], line['y1']), (line['x2'], line['y2']), (0, 0, 255), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(corners, start=1):
        cv2.circle(out, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(out, str(i), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return out

def geometry_change_score(real_stats, fake_stats):
    d_total = abs(fake_stats['total'] - real_stats['total']) / (real_stats['total'] + 1)
    d_short = abs(fake_stats['short_ratio'] - real_stats['short_ratio'])
    d_long = abs(fake_stats['long'] - real_stats['long']) / (real_stats['long'] + 1)
    d_length = abs(fake_stats['mean_top_length'] - real_stats['mean_top_length']) / (real_stats['mean_top_length'] + 1)
    score = 0.3 * d_total + 0.25 * d_short + 0.25 * d_long + 0.2 * d_length
    return min(1.0, float(score))

def chay_geometry():
    real_map = tao_map_theo_index(REAL_DIAGRAM_DIR)
    fake_map = tao_map_theo_index(FAKE_DIAGRAM_DIR)
    indices = sorted(set(real_map) & set(fake_map))
    indices = gioi_han(indices, MAX_GEOMETRY_PAIRS)
    print(f'[GEOMETRY] so cap: {len(indices)}')
    rows = []
    for index in indices:
        real_img = doc_anh(real_map[index])
        fake_img = doc_anh(fake_map[index])
        if real_img is None or fake_img is None:
            continue
        sk_real, lines_real = tach_hinh_hoc(real_img)
        sk_fake, lines_fake = tach_hinh_hoc(fake_img)
        real_stats = phan_tich_net(lines_real)
        fake_stats = phan_tich_net(lines_fake)
        real_corners = tim_goc_ung_vien(real_stats['top_lines'], real_img.shape)
        fake_corners = tim_goc_ung_vien(fake_stats['top_lines'], fake_img.shape)
        real_view = ve_geometry(sk_real, real_stats, real_corners)
        fake_view = ve_geometry(sk_fake, fake_stats, fake_corners)
        real_view, fake_view = resize_cung_chieu_cao(real_view, fake_view)
        score = geometry_change_score(real_stats, fake_stats)
        warning = score >= GEOMETRY_WARNING_THR
        compare = np.hstack([dat_nhan(real_view, f"REAL | lines={real_stats['total']} corners={len(real_corners)}"), dat_nhan(fake_view, f'TAMPER | score={score:.3f} corners={len(fake_corners)}')])
        luu_anh(GEOMETRY_OUT / f'geometry_{index:03d}.png', compare)
        top_lines_text = '; '.join((f"({x['x1']},{x['y1']})-({x['x2']},{x['y2']}) L={x['length']:.1f} A={x['angle']:.1f}" for x in real_stats['top_lines'][:6]))
        corners_text = '; '.join((f'({x},{y})' for x, y in real_corners))
        rows.append({'source_index': index, 'real_total': real_stats['total'], 'fake_total': fake_stats['total'], 'real_short_ratio': real_stats['short_ratio'], 'fake_short_ratio': fake_stats['short_ratio'], 'real_long': real_stats['long'], 'fake_long': fake_stats['long'], 'geometry_score': score, 'geometry_warning': int(warning), 'main_lines_real': top_lines_text, 'corner_candidates_real': corners_text})
    return rows

def khoi_tao_ocr():
    return PaddleOCR(lang='vi', device='cpu', enable_mkldnn=False, use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)

def parse_paddle_text(result):
    texts_out = []
    if result is None:
        return texts_out
    for res in result:
        try:
            texts = res['rec_texts']
            scores = res['rec_scores']
            for text, score in zip(texts, scores):
                text = str(text).strip()
                try:
                    score = float(score)
                except Exception:
                    score = 1.0
                if text and score >= OCR_MIN_SCORE:
                    texts_out.append(text)
        except Exception:
            continue
    return texts_out

def chay_ocr_crop(ocr, img):
    try:
        result = ocr.predict(img)
        return ' '.join(parse_paddle_text(result)).strip()
    except Exception:
        return ''

def tim_dong_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape[:2]
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 40 or h < 8:
            continue
        if w * h < 250:
            continue
        if h > h_img * 0.2:
            continue
        x1 = max(0, x - 4)
        y1 = max(0, y - 3)
        x2 = min(w_img, x + w + 4)
        y2 = min(h_img, y + h + 3)
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return (boxes, dilated)

def ocr_theo_dong(ocr, img):
    boxes, dilated = tim_dong_text(img)
    lines = []
    for x, y, w, h in boxes:
        crop = img[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        text = chay_ocr_crop(ocr, crop)
        if text:
            lines.append(text)
    if not lines:
        text = chay_ocr_crop(ocr, img)
        if text:
            lines = [text]
    return (lines, boxes, dilated)

def bo_dau(text):
    text = str(text).lower().replace('đ', 'd')
    text = unicodedata.normalize('NFD', text)
    text = ''.join((c for c in text if unicodedata.category(c) != 'Mn'))
    return re.sub('\\s+', ' ', text).strip()

def lay_regex(lines_norm, pattern, group=1):
    for line in lines_norm:
        match = re.search(pattern, line)
        if match:
            value = match.group(group).strip(' ,.;:-')
            if value:
                return value
    return ''

def trich_xuat_truong(lines):
    lines_norm = [bo_dau(line) for line in lines]
    return {'so_thua': lay_regex(lines_norm, '(?:thua dat so|so thua)\\s*[:\\-]?\\s*([0-9a-z./-]+)'), 'to_ban_do': lay_regex(lines_norm, 'to ban do(?: so)?\\s*[:\\-]?\\s*([0-9a-z./-]+)'), 'dien_tich': lay_regex(lines_norm, 'dien tich\\s*[:\\-]?\\s*([0-9]+(?:[.,][0-9]+)?)'), 'hinh_thuc_su_dung': lay_regex(lines_norm, 'hinh thuc su dung\\s*[:\\-]?\\s*(.+)'), 'thoi_han_su_dung': lay_regex(lines_norm, 'thoi han su dung\\s*[:\\-]?\\s*(.+)'), 'ten_chu_so_huu': lay_regex(lines_norm, '(?:ho va ten|nguoi su dung dat)\\s*[:\\-]?\\s*(.+)')}

def ve_ocr_boxes(img, boxes):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 1)
    return out

def fields_khac_nhau(real_fields, fake_fields):
    changed = []
    for key in real_fields:
        real_value = bo_dau(real_fields[key])
        fake_value = bo_dau(fake_fields[key])
        if real_value and fake_value and (real_value != fake_value):
            changed.append(key)
    return changed

def chay_ocr_pipeline(ocr):
    real_paths = gioi_han(lay_ds_anh(REAL_TEXT_DIR), MAX_OCR_IMAGES)
    print(f'[OCR] so anh real: {len(real_paths)}')
    rows = []
    real_fields_map = {}
    for path in real_paths:
        index = lay_source_index(path)
        img = doc_anh(path)
        if img is None or index is None:
            continue
        lines, boxes, _ = ocr_theo_dong(ocr, img)
        fields = trich_xuat_truong(lines)
        real_fields_map[index] = fields
        luu_anh(OCR_OUT / f'ocr_lines_{index:03d}.png', ve_ocr_boxes(img, boxes))
        rows.append({'source_index': index, 'label': 'real', **fields, 'raw_text': ' | '.join(lines)})
    fake_map = tao_map_theo_index(FAKE_TEXT_DIR)
    pair_indices = sorted(set(real_fields_map) & set(fake_map))
    pair_indices = gioi_han(pair_indices, MAX_OCR_PAIRS)
    text_changes = {}
    for index in pair_indices:
        img = doc_anh(fake_map[index])
        if img is None:
            continue
        lines, boxes, _ = ocr_theo_dong(ocr, img)
        fields = trich_xuat_truong(lines)
        changed = fields_khac_nhau(real_fields_map[index], fields)
        text_changes[index] = changed
        luu_anh(OCR_OUT / f'ocr_tamper_{index:03d}.png', ve_ocr_boxes(img, boxes))
        rows.append({'source_index': index, 'label': 'text_tamper', **fields, 'raw_text': ' | '.join(lines)})
    return (rows, text_changes)

def tao_red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 60, 45]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 60, 45]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask1, mask2)

def tim_seal_box(real_img):
    mask = tao_red_mask(real_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    grouped = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    if w < 20 or h < 20:
        return None
    H, W = real_img.shape[:2]
    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return (x1, y1, x2 - x1, y2 - y1)

def tinh_lbp_hist(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist

def lbp_distance(hist1, hist2):
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))

def ve_lbp_hist(hist, width=460, height=260):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    max_value = float(hist.max()) if hist.size else 0.0
    if max_value <= 0:
        return canvas
    bar_w = max(1, width // len(hist))
    for i, value in enumerate(hist):
        bar_h = int(value / max_value * (height - 40))
        x1 = i * bar_w
        x2 = min(width - 1, x1 + bar_w - 2)
        cv2.rectangle(canvas, (x1, height - 20), (x2, height - 20 - bar_h), (0, 0, 0), -1)
    return canvas

def chay_seal():
    real_map = tao_map_theo_index(REAL_SEAL_DIR)
    fake_map = tao_map_theo_index(FAKE_SEAL_DIR)
    indices = sorted(set(real_map) & set(fake_map))
    indices = gioi_han(indices, MAX_SEAL_PAIRS)
    print(f'[SEAL] so cap: {len(indices)}')
    rows = []
    for index in indices:
        real_img = doc_anh(real_map[index])
        fake_img = doc_anh(fake_map[index])
        if real_img is None or fake_img is None:
            continue
        box = tim_seal_box(real_img)
        if box is None:
            continue
        x, y, w, h = box
        real_crop = real_img[y:y + h, x:x + w]
        fake_crop = fake_img[y:y + h, x:x + w]
        if real_crop.size == 0 or fake_crop.size == 0:
            continue
        real_crop = cv2.resize(real_crop, (256, 256))
        fake_crop = cv2.resize(fake_crop, (256, 256))
        hist_real = tinh_lbp_hist(real_crop)
        hist_fake = tinh_lbp_hist(fake_crop)
        distance = lbp_distance(hist_real, hist_fake)
        warning = distance >= SEAL_WARNING_THR
        compare = np.hstack([dat_nhan(ve_lbp_hist(hist_real), 'LBP - REAL SEAL'), dat_nhan(ve_lbp_hist(hist_fake), f'TAMPER | distance={distance:.3f}')])
        luu_anh(SEAL_OUT / f'seal_lbp_{index:03d}.png', compare)
        rows.append({'source_index': index, 'lbp_distance': distance, 'seal_warning': int(warning)})
    return rows

def ghi_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def luu_ket_qua(geometry_rows, ocr_rows, text_changes, seal_rows):
    ghi_csv(GEOMETRY_CSV, geometry_rows)
    ghi_csv(OCR_CSV, ocr_rows)
    ghi_csv(SEAL_CSV, seal_rows)
    geometry_map = {row['source_index']: row for row in geometry_rows}
    seal_map = {row['source_index']: row for row in seal_rows}
    all_indices = sorted(set(geometry_map) | set(seal_map) | set(text_changes))
    summary = []
    for index in all_indices:
        geometry = geometry_map.get(index, {})
        seal = seal_map.get(index, {})
        changed = text_changes.get(index, [])
        warnings = []
        if changed:
            warnings.append('text:' + ','.join(changed))
        if geometry.get('geometry_warning', 0):
            warnings.append('geometry')
        if seal.get('seal_warning', 0):
            warnings.append('seal')
        summary.append({'source_index': index, 'text_changed_fields': ','.join(changed), 'geometry_score': geometry.get('geometry_score', ''), 'lbp_distance': seal.get('lbp_distance', ''), 'warning': ' | '.join(warnings)})
    ghi_csv(SUMMARY_CSV, summary)

def main():
    tao_output()
    print('\n[1] Geometry - Skeletonization + Hough')
    geometry_rows = chay_geometry()
    print('\n[2] OCR - Dilate + PaddleOCR')
    ocr = khoi_tao_ocr()
    ocr_rows, text_changes = chay_ocr_pipeline(ocr)
    print('\n[3] Seal - LBP')
    seal_rows = chay_seal()
    luu_ket_qua(geometry_rows, ocr_rows, text_changes, seal_rows)
    print('\nHOAN THANH TUAN 4')
    print('Ket qua:', OUT_DIR)
if __name__ == '__main__':
    main()
