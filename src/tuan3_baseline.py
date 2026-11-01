import cv2
import os
import re
import shutil
import numpy as np
import pytesseract
from pathlib import Path
REAL_SEAL_DIR = Path('dataset/processed_real/seal')
REAL_TEXT_DIR = Path('dataset/processed_real/text')
REAL_FULL_DIR = Path('dataset/processed_real/full_clahe')
TAMPER_DIRS = {'text': Path('dataset/processed_tampered/text_tamper/full_clahe'), 'diagram': Path('dataset/processed_tampered/diagram_tamper/full_clahe'), 'seal': Path('dataset/processed_tampered/seal_tamper/full_clahe')}
SEAL_TAMPER_DIR = Path('dataset/processed_tampered/seal_tamper/seal')
OUT_DIR = Path('dataset/tuan3_results')
HSV_OUT_DIR = OUT_DIR / 'hsv_seal'
FFT_OUT_DIR = OUT_DIR / 'fft'
OCR_OUT_FILE = OUT_DIR / 'ocr_tesseract.txt'
CLEAR_OLD_OUTPUT = True
MAX_FFT_PAIRS_PER_TYPE = 5
MAX_OCR_IMAGES = 5
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_TESSERACT_CMD = PROJECT_ROOT / 'Tesseract' / 'tesseract.exe'
LOCAL_TESSDATA_DIR = PROJECT_ROOT / 'Tesseract' / 'tessdata'
LEGACY_TESSERACT_CMD = Path('D:/Tesseract/tesseract.exe')
LEGACY_TESSDATA_DIR = Path('D:/Tesseract/tessdata')

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

def tao_output():
    if CLEAR_OLD_OUTPUT and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    HSV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    FFT_OUT_DIR.mkdir(parents=True, exist_ok=True)

def tach_dau_do(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 60, 45]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 60, 45]), np.array([180, 255, 255]))
    mask_do = cv2.bitwise_or(mask1, mask2)
    dau_do = cv2.bitwise_and(img, img, mask=mask_do)
    return (dau_do, mask_do)

def dat_nhan(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (255, 255, 255), -1)
    cv2.putText(out, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return out

def resize_cung_chieu_cao(img1, img2):
    h = min(img1.shape[0], img2.shape[0])

    def resize(img):
        scale = h / img.shape[0]
        w = max(1, int(img.shape[1] * scale))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return (resize(img1), resize(img2))

def chay_hsv_seal():
    real_map = tao_map_theo_index(REAL_SEAL_DIR)
    fake_map = tao_map_theo_index(SEAL_TAMPER_DIR)
    common_indices = sorted(set(real_map) & set(fake_map))
    print(f'[HSV] so cap Real/Seal Tamper: {len(common_indices)}')
    for index in common_indices:
        real_img = doc_anh(real_map[index])
        fake_img = doc_anh(fake_map[index])
        if real_img is None or fake_img is None:
            print('loi doc cap seal:', index)
            continue
        real_red, _ = tach_dau_do(real_img)
        fake_red, _ = tach_dau_do(fake_img)
        real_red, fake_red = resize_cung_chieu_cao(real_red, fake_red)
        compare = np.hstack([dat_nhan(real_red, 'HSV - REAL SEAL'), dat_nhan(fake_red, 'HSV - TAMPERED SEAL')])
        luu_anh(HSV_OUT_DIR / f'seal_compare_{index:03d}.png', compare)

def phan_tich_fft(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    f = np.fft.fftshift(np.fft.fft2(gray))
    spectrum = 20 * np.log(np.abs(f) + 1)
    spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return spectrum.astype(np.uint8)

def cat_fft_roi(img, tamper_type):
    h, w = img.shape[:2]
    if tamper_type == 'text':
        return img[0:int(h * 0.55), 0:int(w * 0.45)]
    if tamper_type == 'diagram':
        return img[int(h * 0.1):int(h * 0.8), int(w * 0.5):w]
    if tamper_type == 'seal':
        return img[int(h * 0.65):h, 0:int(w * 0.5)]
    return img

def tao_fft_compare(real_img, fake_img, tamper_type):
    roi_real = cat_fft_roi(real_img, tamper_type)
    roi_fake = cat_fft_roi(fake_img, tamper_type)
    fft_real = phan_tich_fft(roi_real)
    fft_fake = phan_tich_fft(roi_fake)
    fft_real = cv2.cvtColor(fft_real, cv2.COLOR_GRAY2BGR)
    fft_fake = cv2.cvtColor(fft_fake, cv2.COLOR_GRAY2BGR)
    fft_real, fft_fake = resize_cung_chieu_cao(fft_real, fft_fake)
    return np.hstack([dat_nhan(fft_real, 'FFT - REAL'), dat_nhan(fft_fake, f'FFT - {tamper_type.upper()} TAMPER')])

def chay_fft_cho_loai(tamper_type, tamper_folder):
    real_map = tao_map_theo_index(REAL_FULL_DIR)
    fake_map = tao_map_theo_index(tamper_folder)
    common_indices = sorted(set(real_map) & set(fake_map))
    if MAX_FFT_PAIRS_PER_TYPE is not None:
        common_indices = common_indices[:MAX_FFT_PAIRS_PER_TYPE]
    print(f'[FFT-{tamper_type}] so cap: {len(common_indices)}')
    for index in common_indices:
        real_img = doc_anh(real_map[index])
        fake_img = doc_anh(fake_map[index])
        if real_img is None or fake_img is None:
            print('loi doc cap FFT:', tamper_type, index)
            continue
        compare = tao_fft_compare(real_img, fake_img, tamper_type)
        luu_anh(FFT_OUT_DIR / f'fft_{tamper_type}_{index:03d}.png', compare)

def chay_fft():
    for tamper_type, folder in TAMPER_DIRS.items():
        chay_fft_cho_loai(tamper_type, folder)

def cau_hinh_tesseract():
    if LOCAL_TESSERACT_CMD.exists():
        pytesseract.pytesseract.tesseract_cmd = str(LOCAL_TESSERACT_CMD)
        if LOCAL_TESSDATA_DIR.exists():
            os.environ['TESSDATA_PREFIX'] = str(LOCAL_TESSDATA_DIR)
        print('[OCR] Tesseract local:', LOCAL_TESSERACT_CMD)
        vie_file = LOCAL_TESSDATA_DIR / 'vie.traineddata'
        if not vie_file.exists():
            print('[OCR][CANH BAO] Khong thay vie.traineddata trong:', LOCAL_TESSDATA_DIR)
        return True
    path_cmd = shutil.which('tesseract')
    if path_cmd:
        pytesseract.pytesseract.tesseract_cmd = path_cmd
        print('[OCR] Tesseract PATH:', path_cmd)
        return True
    if LEGACY_TESSERACT_CMD.exists():
        pytesseract.pytesseract.tesseract_cmd = str(LEGACY_TESSERACT_CMD)
        if LEGACY_TESSDATA_DIR.exists():
            os.environ['TESSDATA_PREFIX'] = str(LEGACY_TESSDATA_DIR)
        print('[OCR] Tesseract legacy:', LEGACY_TESSERACT_CMD)
        return True
    print('[OCR][LOI] Khong tim thay Tesseract. Can co Tesseract\\tesseract.exe trong thu muc project hoac Tesseract trong PATH.')
    return False

def doc_tesseract(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return pytesseract.image_to_string(thresh, lang='vie', config='--oem 1 --psm 6')

def chay_ocr():
    if not cau_hinh_tesseract():
        return
    text_paths = lay_ds_anh(REAL_TEXT_DIR)
    if not text_paths:
        print('[OCR] khong co anh text')
        return
    if MAX_OCR_IMAGES is not None:
        text_paths = text_paths[:MAX_OCR_IMAGES]
    print(f'[OCR] so anh baseline: {len(text_paths)}')
    results = []
    for path in text_paths:
        img = doc_anh(path)
        if img is None:
            print('loi doc anh OCR:', path.name)
            continue
        try:
            text = doc_tesseract(img).strip()
        except Exception as e:
            text = 'LOI TESSERACT: ' + str(e)
        results.append((path.name, text))
    with open(OCR_OUT_FILE, 'w', encoding='utf-8') as f:
        for name, text in results:
            f.write('=' * 60 + '\n' + name + '\n' + '=' * 60 + '\n' + text + '\n\n')
    if results:
        print('\nOCR mau dau tien:')
        print(results[0][0])
        print(results[0][1])

def main():
    tao_output()
    print('\n[1] HSV - tach dau do')
    chay_hsv_seal()
    print('\n[2] FFT - so sanh Real/Tampered')
    chay_fft()
    print('\n[3] OCR - Tesseract baseline')
    chay_ocr()
    print('\nHOAN THANH TUAN 3')
if __name__ == '__main__':
    main()