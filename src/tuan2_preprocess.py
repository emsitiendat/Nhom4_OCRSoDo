import cv2
import math
import shutil
import numpy as np
from pathlib import Path
REAL_INPUT = Path('dataset/real')
TAMPER_INPUT = Path('dataset/tampered')
REAL_OUTPUT = Path('dataset/processed_real')
TAMPER_OUTPUT = Path('dataset/processed_tampered')
CLEAR_OLD_OUTPUT = True
STD_WIDTH = 1200
STD_HEIGHT = 800
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
TAMPER_RULES = {'fake_text_': ('text_tamper', 'text'), 'fake_diagram_': ('diagram_tamper', 'diagram'), 'fake_seal_': ('seal_tamper', 'seal')}

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

def tao_cau_truc_output():
    if CLEAR_OLD_OUTPUT:
        for folder in [REAL_OUTPUT, TAMPER_OUTPUT]:
            if folder.exists():
                shutil.rmtree(folder)
    for sub in ['text', 'diagram', 'seal', 'full_clahe']:
        (REAL_OUTPUT / sub).mkdir(parents=True, exist_ok=True)
    for tamper_type, roi_name in TAMPER_RULES.values():
        (TAMPER_OUTPUT / tamper_type / roi_name).mkdir(parents=True, exist_ok=True)
        (TAMPER_OUTPUT / tamper_type / 'full_clahe').mkdir(parents=True, exist_ok=True)

def tang_tuong_phan(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def xoay_thang_anh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()[:4]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -45 <= angle <= 45:
                angles.append(angle)
    angle = np.median(angles) if angles else 0
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def sap_xep_4_diem(pts):
    rect = np.zeros((4, 2), dtype='float32')
    tong = pts.sum(axis=1)
    hieu = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(tong)]
    rect[1] = pts[np.argmin(hieu)]
    rect[2] = pts[np.argmax(tong)]
    rect[3] = pts[np.argmax(hieu)]
    return rect

def bien_doi_phoi_canh(img):
    img_area = img.shape[0] * img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    khung_giay = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.3 * img_area:
            khung_giay = approx
            break
    if khung_giay is None:
        return None
    rect = sap_xep_4_diem(khung_giay.reshape(4, 2))
    tl, tr, br, bl = rect
    width1 = np.linalg.norm(br - bl)
    width2 = np.linalg.norm(tr - tl)
    height1 = np.linalg.norm(tr - br)
    height2 = np.linalg.norm(tl - bl)
    max_width = max(int(width1), int(width2))
    max_height = max(int(height1), int(height2))
    if max_width < 10 or max_height < 10:
        return None
    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (max_width, max_height))

def preprocess_image(img):
    warped = bien_doi_phoi_canh(img)
    if warped is None:
        warped = xoay_thang_anh(img)
    img_resize = cv2.resize(warped, (STD_WIDTH, STD_HEIGHT))
    gray = tang_tuong_phan(img_resize)
    h, w = gray.shape[:2]
    text_roi = gray[0:int(h * 0.55), 0:int(w * 0.45)]
    diagram_roi = gray[int(h * 0.1):int(h * 0.8), int(w * 0.5):w]
    seal_roi = img_resize[int(h * 0.65):h, 0:int(w * 0.5)]
    return {'text': text_roi, 'diagram': diagram_roi, 'seal': seal_roi, 'full_clahe': gray}

def xu_ly_real():
    files = lay_ds_anh(REAL_INPUT)
    for i, path in enumerate(files, start=1):
        img = doc_anh(path)
        if img is None:
            print('loi doc anh:', path.name)
            continue
        result = preprocess_image(img)
        for roi_name in ['text', 'diagram', 'seal', 'full_clahe']:
            luu_anh(REAL_OUTPUT / roi_name / path.name, result[roi_name])
        print(f'[REAL {i}/{len(files)}] {path.name}')

def lay_loai_tamper(filename):
    name = filename.lower()
    for prefix, rule in TAMPER_RULES.items():
        if name.startswith(prefix):
            return rule
    return None

def xu_ly_tampered():
    files = lay_ds_anh(TAMPER_INPUT)
    for i, path in enumerate(files, start=1):
        rule = lay_loai_tamper(path.name)
        if rule is None:
            print('bo qua file ko ro loai:', path.name)
            continue
        tamper_type, roi_name = rule
        img = doc_anh(path)
        if img is None:
            print('loi doc anh:', path.name)
            continue
        result = preprocess_image(img)
        luu_anh(TAMPER_OUTPUT / tamper_type / roi_name / path.name, result[roi_name])
        luu_anh(TAMPER_OUTPUT / tamper_type / 'full_clahe' / path.name, result['full_clahe'])
        print(f'[TAMPER {i}/{len(files)}] {tamper_type} / {path.name}')

def dem_anh(folder):
    return len(lay_ds_anh(folder))

def kiem_tra_output():
    print('\n--- KIEM TRA OUTPUT ---')
    for roi_name in ['text', 'diagram', 'seal', 'full_clahe']:
        count = dem_anh(REAL_OUTPUT / roi_name)
        print(f'real/{roi_name:<11}: {count}')
    for tamper_type, roi_name in [('text_tamper', 'text'), ('diagram_tamper', 'diagram'), ('seal_tamper', 'seal')]:
        roi_count = dem_anh(TAMPER_OUTPUT / tamper_type / roi_name)
        full_count = dem_anh(TAMPER_OUTPUT / tamper_type / 'full_clahe')
        print(f'\n[{tamper_type}]')
        print(f'  {roi_name:<11}: {roi_count}')
        print(f"  {'full_clahe':<11}: {full_count}")

def main():
    tao_cau_truc_output()
    xu_ly_real()
    xu_ly_tampered()
    kiem_tra_output()
    print('\nHOAN THANH TUAN 2')
if __name__ == '__main__':
    main()