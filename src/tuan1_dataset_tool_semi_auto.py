import cv2
import csv
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
RAW_DIR = Path('dataset/raw')
REAL_DIR = Path('dataset/real')
TAMPER_DIR = Path('dataset/tampered')
META_FILE = Path('dataset/metadata_tuan1.csv')
OUTPUT_EXT = '.png'
PREVIEW_MAX_W = 1400
PREVIEW_MAX_H = 900
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
DEFAULT_FAKE_TEXT = ['999,9 m2', '888', '25', '1000,0 m2', '777']

def tao_thu_muc():
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    TAMPER_DIR.mkdir(parents=True, exist_ok=True)

def doc_anh(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def luu_png(path, img):
    ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        return False
    buf.tofile(str(path))
    return True

def lay_ds_anh():
    files = []
    for p in RAW_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)
    return sorted(files, key=lambda x: x.name.lower())

def tao_preview(img):
    h, w = img.shape[:2]
    scale = min(PREVIEW_MAX_W / w, PREVIEW_MAX_H / h, 1.0)
    if scale < 1.0:
        preview = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        preview = img.copy()
    return (preview, scale)

def roi_preview_sang_goc(roi, scale, img_shape):
    x, y, w, h = roi
    H, W = img_shape[:2]
    if scale <= 0:
        scale = 1.0
    x = int(round(x / scale))
    y = int(round(y / scale))
    w = int(round(w / scale))
    h = int(round(h / scale))
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))
    return (x, y, w, h)

def chon_mot_roi(img, title):
    preview, scale = tao_preview(img)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, preview.shape[1], preview.shape[0])
    roi = cv2.selectROI(title, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if roi == (0, 0, 0, 0):
        return None
    return roi_preview_sang_goc(roi, scale, img.shape)

def blur_pii(img, rois):
    out = img.copy()
    for x, y, w, h in rois:
        roi = out[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        small_w = max(8, w // 18)
        small_h = max(8, h // 18)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        out[y:y + h, x:x + w] = pixelated
    return out

def tim_font(size):
    candidates = ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/segoeui.ttf', 'C:/Windows/Fonts/times.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()

def ve_text_unicode(img, text, box):
    x, y, w, h = box
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    font_size = max(12, int(h * 0.5))
    font = tim_font(font_size)
    while font_size > 10:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= w - 8 and th <= h - 4:
            break
        font_size -= 2
        font = tim_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x + max(4, (w - tw) // 2)
    ty = y + max(2, (h - th) // 2)
    draw.text((tx, ty), text, font=font, fill=(20, 20, 20))
    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out

def tamper_thay_text(img, box, fake_text):
    out = img.copy()
    x, y, w, h = box
    mask = np.zeros(out.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    clean = cv2.inpaint(out, mask, 5, cv2.INPAINT_TELEA)
    pad = max(3, min(w, h) // 10)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(out.shape[1], x + w + pad)
    y2 = min(out.shape[0], y + h + pad)
    border = clean[y1:y2, x1:x2]
    if border.size > 0:
        bg = np.median(border.reshape(-1, 3), axis=0).astype(np.uint8)
        overlay = clean.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), tuple((int(v) for v in bg)), -1)
        clean = cv2.addWeighted(overlay, 0.55, clean, 0.45, 0)
    clean = ve_text_unicode(clean, fake_text, box)
    return clean

def tamper_copy_move(img, src_box, dst_box):
    out = img.copy()
    sx, sy, sw, sh = src_box
    dx, dy, dw, dh = dst_box
    patch = out[sy:sy + sh, sx:sx + sw].copy()
    if patch.size == 0:
        return out
    patch = cv2.resize(patch, (dw, dh), interpolation=cv2.INTER_CUBIC)
    out[dy:dy + dh, dx:dx + dw] = patch
    return out

def chon_nhieu_roi(img, title):
    preview, scale = tao_preview(img)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, preview.shape[1], preview.shape[0])
    rois_preview = cv2.selectROIs(title, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    rois = []
    for roi in rois_preview:
        roi = tuple(map(int, roi))
        if roi[2] > 0 and roi[3] > 0:
            rois.append(roi_preview_sang_goc(roi, scale, img.shape))
    return rois

def xu_ly_mot_anh(raw_path, index):
    print(f'[{index}] {raw_path.name}')
    img = doc_anh(raw_path)
    if img is None:
        print('  loi doc anh')
        return None
    print('  Chon cac vung PII, ENTER de xac nhan, ESC neu khong co.')
    pii_rois = chon_nhieu_roi(img, f'PII - {raw_path.name}')
    real_img = blur_pii(img, pii_rois)
    real_name = f'anonymized_real_{index:03d}{OUTPUT_EXT}'
    luu_png(REAL_DIR / real_name, real_img)
    print('  Tamper: 1-Text | 2-Copy move | 3-Bo qua')
    mode = input('  Lua chon [1]: ').strip() or '1'
    fake_img = real_img.copy()
    tamper_type = 'none'
    tamper_info = ''
    if mode == '1':
        box = chon_mot_roi(real_img, 'Chon vung can sua text')
        if box is not None:
            default_text = DEFAULT_FAKE_TEXT[(index - 1) % len(DEFAULT_FAKE_TEXT)]
            fake_text = input(f'  Noi dung gia [{default_text}]: ').strip() or default_text
            fake_img = tamper_thay_text(real_img, box, fake_text)
            tamper_type = 'text_replace'
            tamper_info = f'box={box};text={fake_text}'
    elif mode == '2':
        src_box = chon_mot_roi(real_img, 'Chon vung nguon copy-move')
        if src_box is not None:
            dst_box = chon_mot_roi(real_img, 'Chon vung dich copy-move')
            if dst_box is not None:
                fake_img = tamper_copy_move(real_img, src_box, dst_box)
                tamper_type = 'copy_move'
                tamper_info = f'src={src_box};dst={dst_box}'
    fake_name = ''
    if tamper_type != 'none':
        fake_name = f'fake_{index:03d}{OUTPUT_EXT}'
        luu_png(TAMPER_DIR / fake_name, fake_img)
    return {'stt': index, 'raw_name': raw_path.name, 'real_name': real_name, 'tampered_name': fake_name, 'tamper_type': tamper_type, 'pii_boxes': str(pii_rois), 'tamper_info': tamper_info}

def ghi_metadata(rows):
    fields = ['stt', 'raw_name', 'real_name', 'tampered_name', 'tamper_type', 'pii_boxes', 'tamper_info']
    with open(META_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    tao_thu_muc()
    files = lay_ds_anh()
    if not files:
        print('khong co anh trong:', RAW_DIR)
        return
    print('tong so anh raw:', len(files))
    rows = []
    for i, raw_path in enumerate(files, start=1):
        result = xu_ly_mot_anh(raw_path, i)
        if result is not None:
            rows.append(result)
            ghi_metadata(rows)
    tampered = sum((1 for r in rows if r['tampered_name']))
    print('\nHOAN THANH TUAN 1 - SEMI AUTO')
    print('raw      :', len(files))
    print('real     :', len(rows))
    print('tampered :', tampered)
if __name__ == '__main__':
    main()