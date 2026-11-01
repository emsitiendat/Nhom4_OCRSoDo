import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import io
import re
import os
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher
from skimage.feature import local_binary_pattern
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['FLAGS_enable_pir_api'] = '0'
st.set_page_config(page_title='Phân tích Sổ đỏ', layout='wide', page_icon='📜')

@st.cache_resource
def load_ocr_model():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_textline_orientation=True, lang='vi', enable_mkldnn=False)
try:
    ocr_model = load_ocr_model()
except Exception as e:
    st.error(f'Lỗi khởi tạo OCR: {e}')
    ocr_model = None

def strip_accents(text: str) -> str:
    text = text.replace('đ', 'd').replace('Đ', 'D')
    text = unicodedata.normalize('NFD', text)
    return ''.join((ch for ch in text if unicodedata.category(ch) != 'Mn'))

def normalize_spaces(text: str) -> str:
    text = text.replace('\xa0', ' ')
    text = re.sub('[ \\t]+', ' ', text)
    text = re.sub('\\n{3,}', '\n\n', text)
    return text.strip()

def normalize_for_match(text: str) -> str:
    s = strip_accents(text).lower()
    s = s.replace('ș', 's').replace('ş', 's')
    s = s.replace('ı', 'i')
    s = s.replace('²', '2')
    s = re.sub('[^a-z0-9.,:/()\\-\\s]', ' ', s)
    s = re.sub('\\s+', ' ', s)
    return s.strip()

def clean_known_labels(text: str) -> str:
    observed_label_errors = {'Dięn tich': 'Diện tích', 'Dien tich': 'Diện tích', 'Hinh thic súr dung': 'Hình thức sử dụng', 'Hinh thic sur dung': 'Hình thức sử dụng', 'Hinh thirc sir dung': 'Hình thức sử dụng', 'Muc dich sù dung': 'Mục đích sử dụng', 'Muc dich su dung': 'Mục đích sử dụng', 'Mue dich su dung': 'Mục đích sử dụng', 'Thời hạn sú dung': 'Thời hạn sử dụng', 'Thói han sú dung': 'Thời hạn sử dụng', 'Thoi han su dung': 'Thời hạn sử dụng', 'Thi han su dung': 'Thời hạn sử dụng', 'Nguồn gốc súù dung': 'Nguồn gốc sử dụng', 'Nguón góc súù dung': 'Nguồn gốc sử dụng', 'Nguon goc suu dung': 'Nguồn gốc sử dụng', 'Nguon goc su dung': 'Nguồn gốc sử dụng', 'Tờ bản đồ sóe': 'Tờ bản đồ số', 'To ban do soe': 'Tờ bản đồ số', 'Nhà o': 'Nhà ở', 'Nha o': 'Nhà ở'}
    out = text
    for wrong, right in observed_label_errors.items():
        out = re.sub(re.escape(wrong), right, out, flags=re.IGNORECASE)
    replacements = {'\\bth[iìíỉĩị]?a\\s+d[aáàảãạăắằẳẵặâấầẩẫậ]?t\\b': 'Thửa đất', '\\bthua\\s+dat\\b': 'Thửa đất', '\\bthra\\s+dat\\b': 'Thửa đất', '\\bt[oòóỏõọơớờởỡợ]?\\s+b[aáàảãạăắằẳẵặâấầẩẫậ]?n\\s+d[oòóỏõọôốồổỗộơớờởỡợ]?\\b': 'Tờ bản đồ', '\\bto\\s+ban\\s+do\\b': 'Tờ bản đồ', '\\bdia\\s+chi\\b': 'Địa chỉ', '\\bdi[a-z]*n\\s+t[iíìỉĩị]?[a-z]*\\b': 'Diện tích', '\\bdien\\s+tich\\b': 'Diện tích', '\\bhinh\\s+th\\w*c\\s+s\\w*\\s+dung\\b': 'Hình thức sử dụng', '\\bm\\w*c\\s+d\\w*ch\\s+s\\w*\\s+dung\\b': 'Mục đích sử dụng', '\\bth\\w*i\\s+h\\w*n\\s+s\\w*\\s+dung\\b': 'Thời hạn sử dụng', '\\bngu\\w*n\\s+g\\w*c\\s+s\\w*\\s+dung\\b': 'Nguồn gốc sử dụng'}
    for pattern, replacement in replacements.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    fixed_phrases = {'Lau dài': 'Lâu dài', 'Lau dai': 'Lâu dài', 'Dát o tai nóng thón': 'Đất ở tại nông thôn', 'Dat o tai nong thon': 'Đất ở tại nông thôn', 'Dát vuón': 'Đất vườn', 'Dat vuon': 'Đất vườn', 'Dát è tai nông thon': 'Đất ở tại nông thôn', 'Dat o tai nong thon': 'Đất ở tại nông thôn', 'Dát o tai do thi': 'Đất ở tại đô thị', 'Dat o tai do thi': 'Đất ở tại đô thị', 'Sù dung rién': 'Sử dụng riêng', 'Su dung rieng': 'Sử dụng riêng', 'Sử dung riêng': 'Sử dụng riêng', 'Chua ching nhán quyén sà hu': 'Chưa chứng nhận quyền sở hữu', 'Chua chung nhan quyen so huu': 'Chưa chứng nhận quyền sở hữu', 'Chua ching nhán quyèn sò hitu': 'Chưa chứng nhận quyền sở hữu', 'Chua ching nhun quyén sò hu': 'Chưa chứng nhận quyền sở hữu', 'Chura ching nhán qiyèn só hu': 'Chưa chứng nhận quyền sở hữu'}
    for wrong, right in fixed_phrases.items():
        out = re.sub(re.escape(wrong), right, out, flags=re.IGNORECASE)
    return normalize_spaces(out)

def parse_ocr_result_detailed(result):
    entries = []
    if not result:
        return entries
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        for line in result[0]:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            box = line[0]
            rec = line[1]
            if not isinstance(rec, (list, tuple)) or len(rec) < 1:
                continue
            txt = str(rec[0]).strip()
            if not txt:
                continue
            try:
                score = float(rec[1])
            except Exception:
                score = 0.0
            try:
                box = np.asarray(box, dtype=np.float32).reshape(-1, 2).tolist()
            except Exception:
                box = None
            entries.append({'text': txt, 'score': score, 'box': box})
        return entries
    results = result if isinstance(result, list) else [result]
    for res in results:
        rec_texts = None
        rec_scores = None
        polys = None
        if hasattr(res, 'rec_texts'):
            rec_texts = res.rec_texts
            rec_scores = getattr(res, 'rec_scores', None)
            polys = getattr(res, 'dt_polys', None)
            if polys is None:
                polys = getattr(res, 'rec_polys', None)
        elif isinstance(res, dict):
            rec_texts = res.get('rec_texts')
            rec_scores = res.get('rec_scores')
            polys = res.get('dt_polys')
            if polys is None:
                polys = res.get('rec_polys')
        if rec_texts is None:
            continue
        for i, txt in enumerate(rec_texts):
            txt = str(txt).strip()
            if not txt:
                continue
            try:
                score = float(rec_scores[i])
            except Exception:
                score = 0.0
            box = None
            try:
                if polys is not None and i < len(polys):
                    box = np.asarray(polys[i], dtype=np.float32).reshape(-1, 2).tolist()
            except Exception:
                box = None
            entries.append({'text': txt, 'score': score, 'box': box})
    return entries

def resize_for_fullpage_ocr(image):
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest < 1100:
        scale = min(2.0, 1900.0 / max(longest, 1))
    elif longest < 1800:
        scale = min(1.35, 2000.0 / longest)
    elif longest > 2400:
        scale = 2200.0 / longest
    else:
        scale = 1.0
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA)
    return (resized, scale)

def build_fast_fullpage_variants(image):
    resized, scale = resize_for_fullpage_ocr(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, _, r = cv2.split(resized)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    red_clahe = clahe.apply(r)
    blur = cv2.GaussianBlur(gray_clahe, (0, 0), 0.9)
    gray_sharp = cv2.addWeighted(gray_clahe, 1.55, blur, -0.55, 0)
    red_pass = cv2.medianBlur(red_clahe, 3)
    return [('FAST Gray + CLAHE + Sharpen', cv2.cvtColor(gray_sharp, cv2.COLOR_GRAY2BGR), scale), ('FAST Red channel + CLAHE', cv2.cvtColor(red_pass, cv2.COLOR_GRAY2BGR), scale)]

def _run_one_ocr(img_variant, pass_name, scale=1.0, roi_origin=(0, 0), roi_kind=None, forced_fields=None):
    if ocr_model is None:
        return None
    try:
        result = ocr_model.ocr(img_variant)
    except Exception:
        try:
            result = list(ocr_model.predict(img_variant))
        except Exception:
            return None
    entries = parse_ocr_result_detailed(result)
    if not entries:
        return None
    texts = [e['text'] for e in entries]
    scores = [e['score'] for e in entries]
    raw_text = '\n'.join(texts)
    cleaned_text = clean_known_labels(raw_text)
    valid_scores = [s for s in scores if 0.0 <= s <= 1.0]
    avg_conf = float(np.mean(valid_scores)) if valid_scores else 0.0
    return {'pass': pass_name, 'raw_text': raw_text, 'cleaned_text': cleaned_text, 'avg_conf': avg_conf, 'quality': avg_conf + min(len(texts), 30) * 0.002, 'num_lines': len(texts), 'entries': entries, 'scale': float(scale), 'roi_origin': roi_origin, 'roi_kind': roi_kind, 'forced_fields': forced_fields or {}}

def first_group(patterns, text, flags=re.IGNORECASE):
    for pattern in patterns:
        m = re.search(pattern, text, flags)
        if m:
            return m.group(1).strip(' \t\n:;,.|')
    return None

def parse_number(value):
    if value is None:
        return None
    s = str(value)
    s = s.replace(' ', '').replace(',', '.')
    s = s.replace('O', '0').replace('o', '0')
    s = s.replace('I', '1').replace('l', '1')
    s = s.replace('S', '5').replace('s', '5')
    m = re.search('\\d+(?:\\.\\d+)?', s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None

def extract_between(text, start_patterns, end_patterns):
    start_match = None
    for pattern in start_patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m and (start_match is None or m.start() < start_match.start()):
            start_match = m
    if start_match is None:
        return None
    start_pos = start_match.end()
    remainder = text[start_pos:]
    end_positions = []
    for pattern in end_patterns:
        m = re.search(pattern, remainder, re.IGNORECASE | re.DOTALL)
        if m:
            end_positions.append(m.start())
    end_pos = min(end_positions) if end_positions else len(remainder)
    value = remainder[:end_pos]
    value = re.sub('^[\\s:;,.\\-]+', '', value)
    value = normalize_spaces(value)
    return value if value else None

def extract_area_robust(original, match_text):
    patterns = ['dien\\s*tich\\s*[:.]?\\s*([0-9OoIlSs]{1,8}(?:[.,][0-9OoIlSs]{1,4})?)\\s*m\\s*(?:2|\\^2)?', 'dien.{0,10}?tich[^0-9]{0,20}([0-9OoIlSs]{1,8}(?:[.,][0-9OoIlSs]{1,4})?)', '(?:^|\\s)c\\)?\\s*[^0-9\\n]{0,25}([0-9OoIlSs]{2,8}(?:[.,][0-9OoIlSs]{1,4})?)\\s*m\\s*(?:2)?']
    raw = first_group(patterns, match_text)
    value = parse_number(raw)
    if value is not None:
        return value
    for line in original.splitlines():
        ml = normalize_for_match(line)
        if 'dien' in ml and 'tich' in ml:
            nums = re.findall('([0-9]{1,8}(?:[.,][0-9]{1,4})?)', ml)
            for n in nums:
                x = parse_number(n)
                if x is not None and x > 0:
                    return x
    return None

def extract_contract_date_semantic(text):
    lines = [normalize_spaces(x) for x in str(text).splitlines() if normalize_spaces(x)]
    positive = ['hop dong', 'hdcn', 'chuyen nhuong', 'cong chung', 'chung thuc', 'chung thue']
    negative = ['quyet dinh', 'qd ubnd', 'qd-ubnd', 'quy hoach', 'ghi chu']
    for i, line in enumerate(lines):
        context_lines = lines[max(0, i - 1):min(len(lines), i + 2)]
        context = ' '.join(context_lines)
        nm = normalize_for_match(context)
        if not any((k in nm for k in positive)):
            continue
        dates = re.findall('([0-3]?\\d/[01]?\\d/[12]\\d{3})', context)
        if not dates:
            continue
        for d in dates:
            date_line = None
            for z in context_lines:
                if d in z:
                    date_line = normalize_for_match(z)
                    break
            if date_line and any((k in date_line for k in negative)):
                continue
            return d
    return None

def canonicalize_hinh_thuc(value):
    if not value:
        return None
    raw = normalize_spaces(value)
    nm = normalize_for_match(raw)
    has_rieng = 'su dung rieng' in nm or 'rieng' in nm
    has_chung = 'su dung chung' in nm or 'chung' in nm
    if has_rieng and (not has_chung):
        return 'Sử dụng riêng'
    if has_chung and (not has_rieng):
        return 'Sử dụng chung'
    cut_tokens = ['muc dich su dung', 'thoi han su dung', 'nguon goc su dung', '2. nha o', '3. cong trinh']
    cut = len(raw)
    nraw = normalize_for_match(raw)
    for token in cut_tokens:
        p = nraw.find(token)
        if p >= 0:
            cut = min(cut, p)
    return normalize_spaces(raw[:cut])[:120] or None

def canonicalize_muc_dich(value):
    if not value:
        return None
    raw = normalize_spaces(value)
    nm = normalize_for_match(raw)
    known = []
    mapping = [('dat o tai nong thon', 'Đất ở tại nông thôn'), ('dat o tai do thi', 'Đất ở tại đô thị'), ('dat vuon', 'Đất vườn'), ('dat trong cay lau nam', 'Đất trồng cây lâu năm'), ('dat trong cay hang nam', 'Đất trồng cây hàng năm'), ('dat nong nghiep', 'Đất nông nghiệp')]
    for key, label in mapping:
        if key in nm and label not in known:
            known.append(label)
    if re.search('\\d+(?:[.,]\\d+)?\\s*m\\s*[²2]?', raw, re.IGNORECASE):
        return raw[:180]
    if known:
        return '; '.join(known)
    return raw[:180] or None

def field_is_plausible(field_name, value):
    if value in (None, '', []):
        return False
    s = normalize_spaces(str(value))
    nm = normalize_for_match(s)
    if field_name == 'so_thua':
        return re.fullmatch('\\d{1,8}', s) is not None
    if field_name == 'to_ban_do':
        return re.fullmatch('\\d{1,8}', s) is not None
    if field_name == 'dien_tich_m2':
        try:
            x = float(value)
            return 0 < x < 10000000
        except Exception:
            return False
    if field_name == 'dia_chi':
        if len(s) > 220:
            return False
        bad = ['dien tich', 'hinh thuc su dung', 'muc dich su dung', 'thoi han su dung']
        return not any((x in nm for x in bad))
    if field_name == 'hinh_thuc_su_dung':
        if len(s) > 120:
            return False
        bad = ['muc dich su dung', 'thoi han su dung', 'nguon goc su dung']
        if any((x in nm for x in bad)):
            return False
        return 'rieng' in nm or 'chung' in nm or 'su dung' in nm
    if field_name == 'muc_dich_su_dung':
        if len(s) > 200:
            return False
        bad = ['hinh thuc su dung', 'thoi han su dung', 'nguon goc su dung']
        if any((x in nm for x in bad)):
            return False
        return 'dat' in nm
    if field_name == 'thoi_han_su_dung':
        if len(s) > 100:
            return False
        bad = ['nguon goc su dung', 'muc dich su dung', 'hinh thuc su dung']
        return not any((x in nm for x in bad))
    if field_name == 'nguon_goc_su_dung':
        if len(s) > 350:
            return False
        bad = ['2. nha o', '3. cong trinh', '4. rung', '5. cay lau nam', '6. ghi chu']
        return not any((x in nm for x in bad))
    if field_name == 'ngay_hop_dong':
        return re.fullmatch('[0-3]?\\d/[01]?\\d/[12]\\d{3}', s) is not None
    return True

def clean_final_fields(fields):
    out = dict(fields)
    if out.get('hinh_thuc_su_dung'):
        out['hinh_thuc_su_dung'] = canonicalize_hinh_thuc(out['hinh_thuc_su_dung'])
    if out.get('muc_dich_su_dung'):
        out['muc_dich_su_dung'] = canonicalize_muc_dich(out['muc_dich_su_dung'])
    if out.get('nguon_goc_su_dung'):
        s = out['nguon_goc_su_dung']
        nm = normalize_for_match(s)
        cut = len(s)
        for token in ['2. nha', '3. cong trinh', '4. rung', '5. cay', '6. ghi chu']:
            p = nm.find(token)
            if p >= 0:
                cut = min(cut, p)
        out['nguon_goc_su_dung'] = normalize_spaces(s[:cut]).strip(' -:;,.') or None
    return out

def extract_land_fields(text):
    original = normalize_spaces(text)
    match_text = normalize_for_match(original)
    fields = {'so_thua': None, 'to_ban_do': None, 'dia_chi': None, 'dien_tich_m2': None, 'hinh_thuc_su_dung': None, 'muc_dich_su_dung': None, 'thoi_han_su_dung': None, 'nguon_goc_su_dung': None, 'so_hop_dong': None, 'ngay_hop_dong': None, 'muc_dich_chi_tiet': []}
    fields['so_thua'] = first_group(['thua\\s+dat\\s+so\\s*[:.]?\\s*([0-9]+)', 'thua\\s+dat[^0-9]{0,12}([0-9]{1,8})'], match_text)
    fields['to_ban_do'] = first_group(['to\\s+ban\\s+do\\s+so(?:e)?\\s*[:.]?\\s*([0-9]+)', 'ban\\s+do\\s+so(?:e)?\\s*[:.]?\\s*([0-9]+)', 'to\\s+ban\\s+do[^0-9]{0,12}([0-9]{1,6})'], match_text)
    fields['dien_tich_m2'] = extract_area_robust(original, match_text)
    fields['dia_chi'] = extract_between(original, ['Địa chỉ\\s*[:.]?', 'Dia chi\\s*[:.]?'], ['\\bc\\)\\s*Diện tích', '\\bDiện tích\\s*[:.]?', '\\bd\\)\\s*Hình thức', '\\bHình thức sử dụng'])
    fields['hinh_thuc_su_dung'] = extract_between(original, ['Hình thức sử dụng\\s*[:.]?'], ['\\b[dđ]\\)\\s*Mục đích sử dụng', '\\bMục đích sử dụng\\s*[:.]?', '\\be\\)\\s*Thời hạn sử dụng'])
    fields['muc_dich_su_dung'] = extract_between(original, ['Mục đích sử dụng\\s*[:.]?'], ['\\be\\)\\s*Thời hạn sử dụng', '\\bThời hạn sử dụng\\s*[:.]?', '\\bg\\)\\s*Nguồn gốc sử dụng'])
    fields['thoi_han_su_dung'] = extract_between(original, ['Thời hạn sử dụng\\s*[:.]?'], ['\\bg\\)\\s*Nguồn gốc sử dụng', '\\bNguồn gốc sử dụng\\s*[:.]?'])
    fields['nguon_goc_su_dung'] = extract_between(original, ['Nguồn gốc sử dụng\\s*[:.]?'], ['(?:^|\\n)\\s*2\\.\\s*Nhà', '(?:^|\\n)\\s*3\\.\\s*Công trình', '(?:^|\\n)\\s*4\\.\\s*Rừng', '(?:^|\\n)\\s*5\\.\\s*Cây', '(?:^|\\n)\\s*6\\.\\s*Ghi chú', '\\bNhà ở\\s*[:.]?', '\\bCông trình xây dựng khác'])
    fields['so_hop_dong'] = first_group(['(?:so|s[oóố])\\s*[:.]?\\s*([0-9]{2,6}/[0-9]{4}/[A-Za-zĐđ]+)', '([0-9]{2,6}/[0-9]{4}/[A-Za-zĐđ]+)'], original)
    fields['ngay_hop_dong'] = extract_contract_date_semantic(original)
    if fields['muc_dich_su_dung']:
        pairs = re.findall('([^,;()]{2,80}?)\\s*\\(\\s*([0-9]+(?:[.,][0-9]+)?)\\s*m\\s*[²2]?\\s*\\)', fields['muc_dich_su_dung'], flags=re.IGNORECASE)
        for name, area in pairs:
            fields['muc_dich_chi_tiet'].append({'loai': normalize_spaces(name).strip(' ,;:-'), 'dien_tich_m2': parse_number(area)})
    return clean_final_fields(fields)

def field_completeness_score(fields):
    weights = {'so_thua': 2, 'to_ban_do': 2, 'dia_chi': 1, 'dien_tich_m2': 4, 'hinh_thuc_su_dung': 1, 'muc_dich_su_dung': 2, 'thoi_han_su_dung': 1, 'nguon_goc_su_dung': 1, 'so_hop_dong': 1, 'ngay_hop_dong': 1}
    return sum((weight for key, weight in weights.items() if field_is_plausible(key, fields.get(key))))

def _text_quality(value):
    if value in (None, '', []):
        return -999.0
    s = normalize_spaces(str(value))
    if not s:
        return -999.0
    score = 0.0
    n = len(s)
    if 2 <= n <= 220:
        score += 0.25
    if 3 <= n <= 120:
        score += 0.15
    if re.search('[A-Za-zÀ-ỹ0-9]', s):
        score += 0.2
    weird = len(re.findall('[^A-Za-zÀ-ỹ0-9\\s.,:/()\\-²]', s))
    if weird > max(3, n * 0.08):
        score -= 0.2
    return score

def _field_value_score(field_name, value, candidate, fields):
    if not field_is_plausible(field_name, value):
        return -999.0
    score = float(candidate.get('avg_conf', 0.0)) * 0.6
    score += min(float(candidate.get('num_lines', 0)), 35.0) * 0.002
    if candidate.get('roi_kind') == field_name:
        score += 0.25
    if field_name == 'so_thua':
        s = str(value).strip()
        if re.fullmatch('\\d{1,8}', s):
            score += 0.85
            if len(s) <= 6:
                score += 0.1
        else:
            score -= 0.5
    elif field_name == 'to_ban_do':
        s = str(value).strip()
        if re.fullmatch('\\d{1,8}', s):
            score += 0.85
            if len(s) <= 5:
                score += 0.1
        else:
            score -= 0.5
    elif field_name == 'dien_tich_m2':
        try:
            x = float(value)
            if 0 < x < 10000000:
                score += 1.05
            else:
                score -= 0.6
        except Exception:
            score -= 0.8
    elif field_name == 'ngay_hop_dong':
        if re.fullmatch('[0-3]?\\d/[01]?\\d/[12]\\d{3}', str(value).strip()):
            score += 0.8
        else:
            score -= 0.3
    elif field_name == 'so_hop_dong':
        if re.search('\\d{2,6}/\\d{4}/[A-Za-zĐđ]+', str(value)):
            score += 0.75
        else:
            score += _text_quality(value)
    else:
        score += _text_quality(value)
    return score

def fuse_fields_across_passes(candidates):
    empty = {'so_thua': None, 'to_ban_do': None, 'dia_chi': None, 'dien_tich_m2': None, 'hinh_thuc_su_dung': None, 'muc_dich_su_dung': None, 'thoi_han_su_dung': None, 'nguon_goc_su_dung': None, 'so_hop_dong': None, 'ngay_hop_dong': None, 'muc_dich_chi_tiet': []}
    if not candidates:
        return (empty, {}, [])
    field_names = ['so_thua', 'to_ban_do', 'dia_chi', 'dien_tich_m2', 'hinh_thuc_su_dung', 'muc_dich_su_dung', 'thoi_han_su_dung', 'nguon_goc_su_dung', 'so_hop_dong', 'ngay_hop_dong']
    per_pass = []
    for c in candidates:
        f = extract_land_fields(c['cleaned_text'])
        for k, v in (c.get('forced_fields') or {}).items():
            if v not in (None, '', []):
                f[k] = v
        f = clean_final_fields(f)
        per_pass.append({'candidate': c, 'fields': f, 'completeness': field_completeness_score(f)})
    fused = empty.copy()
    field_sources = {}
    for field_name in field_names:
        best_item = None
        best_score = -999.0
        for item in per_pass:
            c = item['candidate']
            f = item['fields']
            value = f.get(field_name)
            score = _field_value_score(field_name, value, c, f)
            if score > best_score:
                best_score = score
                best_item = item
        if best_item is not None and best_score > -900:
            fused[field_name] = best_item['fields'].get(field_name)
            field_sources[field_name] = {'pass': best_item['candidate']['pass'], 'confidence': round(best_item['candidate']['avg_conf'], 4), 'score': round(best_score, 4)}
    purpose_source = field_sources.get('muc_dich_su_dung', {}).get('pass')
    if purpose_source:
        for item in per_pass:
            if item['candidate']['pass'] == purpose_source:
                fused['muc_dich_chi_tiet'] = item['fields'].get('muc_dich_chi_tiet') or []
                break
    if fused.get('dien_tich_m2') is None and fused.get('muc_dich_chi_tiet'):
        vals = [p.get('dien_tich_m2') for p in fused['muc_dich_chi_tiet'] if p.get('dien_tich_m2') is not None]
        if len(vals) >= 2:
            inferred = round(sum(vals), 4)
            fused['dien_tich_m2'] = inferred
            field_sources['dien_tich_m2'] = {'pass': 'Cross-check từ mục đích sử dụng', 'confidence': None, 'score': None}
    return (fused, field_sources, per_pass)

def choose_best_overall_candidate(candidates):
    if not candidates:
        return None
    enriched = []
    for c in candidates:
        f = extract_land_fields(c['cleaned_text'])
        completeness = field_completeness_score(f)
        total_score = c['quality'] + completeness * 0.08
        enriched.append((total_score, c))
    return max(enriched, key=lambda x: x[0])[1]
FIELD_LABELS = {'so_thua': ['thua dat so', 'thua dat'], 'to_ban_do': ['to ban do', 'ban do so'], 'dia_chi': ['dia chi'], 'dien_tich_m2': ['dien tich'], 'hinh_thuc_su_dung': ['hinh thuc su dung'], 'muc_dich_su_dung': ['muc dich su dung'], 'thoi_han_su_dung': ['thoi han su dung'], 'nguon_goc_su_dung': ['nguon goc su dung']}
NEXT_FIELD = {'dia_chi': 'dien_tich_m2', 'dien_tich_m2': 'hinh_thuc_su_dung', 'hinh_thuc_su_dung': 'muc_dich_su_dung', 'muc_dich_su_dung': 'thoi_han_su_dung', 'thoi_han_su_dung': 'nguon_goc_su_dung'}

def _label_match_score(text, field_name):
    t = normalize_for_match(text)
    best = 0.0
    for label in FIELD_LABELS.get(field_name, []):
        ln = normalize_for_match(label)
        tokens = [x for x in ln.split() if len(x) >= 2]
        token_hit = sum((tok in t for tok in tokens)) / max(len(tokens), 1)
        seq = SequenceMatcher(None, t, ln).ratio()
        best = max(best, token_hit, seq)
    return best

def find_best_label_entry(candidates, field_name):
    best = None
    best_score = 0.0
    for c in candidates:
        if c.get('roi_kind') is not None:
            continue
        for e in c.get('entries', []):
            if not e.get('box'):
                continue
            s = _label_match_score(e.get('text', ''), field_name)
            s = s * 0.85 + float(e.get('score', 0.0)) * 0.15
            if s > best_score:
                best_score = s
                best = (c, e)
    if best_score < 0.5:
        return None
    return best

def _entry_rect_original(candidate, entry):
    box = np.asarray(entry['box'], dtype=np.float32)
    scale = max(float(candidate.get('scale', 1.0)), 1e-06)
    box = box / scale
    ox, oy = candidate.get('roi_origin', (0, 0))
    box[:, 0] += float(ox)
    box[:, 1] += float(oy)
    xs = box[:, 0]
    ys = box[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

def crop_field_roi_between_labels(image, candidates, field_name, return_origin=False):
    found = find_best_label_entry(candidates, field_name)
    if not found:
        return None
    cand, entry = found
    x0, y0, x1, y1 = _entry_rect_original(cand, entry)
    h, w = image.shape[:2]
    line_h = max(12.0, y1 - y0)
    cx0 = max(0, int(x0 - 0.025 * w))
    cx1 = min(w, int(0.97 * w))
    cy0 = max(0, int(y0 - 0.65 * line_h))
    next_name = NEXT_FIELD.get(field_name)
    cy1 = None
    if next_name:
        next_found = find_best_label_entry(candidates, next_name)
        if next_found:
            nc, ne = next_found
            _, ny0, _, _ = _entry_rect_original(nc, ne)
            if ny0 > y0 and ny0 - y0 < 12 * line_h:
                cy1 = int(ny0 - 0.2 * line_h)
    if cy1 is None:
        if field_name == 'nguon_goc_su_dung':
            cy1 = int(y1 + 2.4 * line_h)
        elif field_name == 'dia_chi':
            cy1 = int(y1 + 2.0 * line_h)
        else:
            cy1 = int(y1 + 1.35 * line_h)
    cy1 = min(h, max(cy0 + 4, cy1))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    crop = image[cy0:cy1, cx0:cx1].copy()
    if return_origin:
        return (crop, (cx0, cy0))
    return crop

def crop_land_header_row(image, candidates, return_origin=False):
    found = find_best_label_entry(candidates, 'so_thua')
    if not found:
        found = find_best_label_entry(candidates, 'to_ban_do')
    if not found:
        return None
    cand, entry = found
    x0, y0, x1, y1 = _entry_rect_original(cand, entry)
    h, w = image.shape[:2]
    line_h = max(12.0, y1 - y0)
    cx0 = max(0, int(x0 - 0.04 * w))
    cx1 = min(w, int(0.98 * w))
    cy0 = max(0, int(y0 - 0.8 * line_h))
    cy1 = min(h, int(y1 + 1.3 * line_h))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    crop = image[cy0:cy1, cx0:cx1].copy()
    if return_origin:
        return (crop, (cx0, cy0))
    return crop

def prepare_local_variant(crop, method='gray', return_scale=False):
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None
    scale_h = 160.0 / max(h, 1)
    scale = min(3.2, max(1.4, scale_h))
    scale = min(scale, 2200.0 / max(w, 1))
    scale = max(scale, 1.0)
    rs = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(rs, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    if method == 'otsu':
        _, out_img = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        blur = cv2.GaussianBlur(g, (0, 0), 0.7)
        out_img = cv2.addWeighted(g, 1.7, blur, -0.7, 0)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
    if return_scale:
        return (out_img, float(scale))
    return out_img

def parse_land_header_special(text):
    nm = normalize_for_match(text)
    out = {}
    so_thua = first_group(['thua\\s+dat\\s+so\\s*[:.]?\\s*([0-9]{1,8})', 'thua\\s+dat[^0-9]{0,16}([0-9]{1,8})'], nm)
    to_ban_do = first_group(['to\\s+ban\\s+do\\s+so\\s*[:.]?\\s*([0-9]{1,8})', 'ban\\s+do\\s+so[^0-9]{0,12}([0-9]{1,8})'], nm)
    if so_thua:
        out['so_thua'] = so_thua
    if to_ban_do:
        out['to_ban_do'] = to_ban_do
    return out

def parse_field_special(field_name, text):
    cleaned = clean_known_labels(text)
    nm = normalize_for_match(cleaned)
    out = {}
    if field_name == 'dien_tich_m2':
        v = extract_area_robust(cleaned, nm)
        if v is None:
            m = re.search('([0-9]{1,8}(?:[.,][0-9]{1,4})?)\\s*m\\s*(?:2)?', nm)
            if m:
                v = parse_number(m.group(1))
        if v is not None:
            out[field_name] = v
        return out
    if field_name == 'hinh_thuc_su_dung':
        if 'rieng' in nm:
            out[field_name] = 'Sử dụng riêng'
        elif 'chung' in nm:
            out[field_name] = 'Sử dụng chung'
        return out
    if field_name == 'muc_dich_su_dung':
        value = extract_between(cleaned, ['Mục đích sử dụng\\s*[:.]?'], ['Thời hạn sử dụng\\s*[:.]?', 'Nguồn gốc sử dụng\\s*[:.]?'])
        if value is None:
            value = cleaned
        value = canonicalize_muc_dich(value)
        if value and field_is_plausible(field_name, value):
            out[field_name] = value
        return out
    if field_name == 'thoi_han_su_dung':
        if 'lau dai' in nm:
            out[field_name] = 'Lâu dài'
            return out
        value = extract_between(cleaned, ['Thời hạn sử dụng\\s*[:.]?'], ['Nguồn gốc sử dụng\\s*[:.]?'])
        if value and field_is_plausible(field_name, value):
            out[field_name] = value[:100]
        return out
    if field_name == 'dia_chi':
        value = extract_between(cleaned, ['Địa chỉ\\s*[:.]?', 'Dia chi\\s*[:.]?'], ['Diện tích\\s*[:.]?', 'Hình thức sử dụng\\s*[:.]?'])
        if value is None:
            value = cleaned
        value = re.sub('(?i)^\\s*(?:b\\)?\\s*)?(?:địa chỉ|dia chi)\\s*[:.]?\\s*', '', value)
        value = normalize_spaces(value)
        if value and field_is_plausible(field_name, value):
            out[field_name] = value[:220]
        return out
    if field_name == 'nguon_goc_su_dung':
        value = extract_between(cleaned, ['Nguồn gốc sử dụng\\s*[:.]?'], ['(?:^|\\n)\\s*2\\.\\s*Nhà', '(?:^|\\n)\\s*3\\.\\s*Công trình', '(?:^|\\n)\\s*4\\.\\s*Rừng', '(?:^|\\n)\\s*5\\.\\s*Cây', '(?:^|\\n)\\s*6\\.\\s*Ghi chú'])
        if value is None:
            value = cleaned
        value = clean_final_fields({'nguon_goc_su_dung': value}).get('nguon_goc_su_dung')
        if value and field_is_plausible(field_name, value):
            out[field_name] = value
        return out
    return out

def run_land_header_fallback(image, candidates):
    crop_info = crop_land_header_row(image, candidates, return_origin=True)
    if crop_info is None:
        return []
    crop, roi_origin = crop_info
    output = []
    for method_name, method in [('ROI Land Header Gray', 'gray'), ('ROI Land Header Otsu', 'otsu')]:
        prepared = prepare_local_variant(crop, method, return_scale=True)
        if prepared is None:
            continue
        variant, local_scale = prepared
        c = _run_one_ocr(variant, method_name, scale=local_scale, roi_origin=roi_origin, roi_kind='land_header')
        if c is None:
            continue
        forced = parse_land_header_special(c['cleaned_text'])
        c['forced_fields'] = forced
        output.append(c)
        if forced.get('so_thua') and forced.get('to_ban_do'):
            break
    return output

def run_targeted_field_fallback(image, field_name, candidates):
    crop_info = crop_field_roi_between_labels(image, candidates, field_name, return_origin=True)
    if crop_info is None:
        return []
    crop, roi_origin = crop_info
    output = []
    for method_name, method in [('ROI Field Gray', 'gray'), ('ROI Field Otsu', 'otsu')]:
        prepared = prepare_local_variant(crop, method, return_scale=True)
        if prepared is None:
            continue
        variant, local_scale = prepared
        c = _run_one_ocr(variant, f'{method_name} [{field_name}]', scale=local_scale, roi_origin=roi_origin, roi_kind=field_name)
        if c is None:
            continue
        forced = parse_field_special(field_name, c['cleaned_text'])
        c['forced_fields'] = forced
        output.append(c)
        v = forced.get(field_name)
        if field_is_plausible(field_name, v):
            break
    return output

def run_textblock_fallback(image):
    h, w = image.shape[:2]
    crop = image[0:int(h * 0.68), 0:int(w * 0.93)].copy()
    variant = prepare_local_variant(crop, method='gray')
    if variant is None:
        return None
    return _run_one_ocr(variant, 'ROI Text Block - Gray+Sharpen', roi_kind='text_block')

def field_missing_or_bad(fields, field_name):
    return not field_is_plausible(field_name, fields.get(field_name))

def run_smart_fast_ocr(image):
    candidates = []
    variants = build_fast_fullpage_variants(image)
    name, variant, scale = variants[0]
    c = _run_one_ocr(variant, name, scale=scale)
    if c is not None:
        candidates.append(c)
    fields, sources, debug = fuse_fields_across_passes(candidates)
    fields = clean_final_fields(fields)
    core_keys = ['so_thua', 'to_ban_do', 'dien_tich_m2', 'hinh_thuc_su_dung', 'muc_dich_su_dung', 'thoi_han_su_dung']
    core_found = sum((not field_missing_or_bad(fields, k) for k in core_keys))
    if core_found < 5:
        name, variant, scale = variants[1]
        c = _run_one_ocr(variant, name, scale=scale)
        if c is not None:
            candidates.append(c)
        fields, sources, debug = fuse_fields_across_passes(candidates)
        fields = clean_final_fields(fields)
    if field_missing_or_bad(fields, 'so_thua') or field_missing_or_bad(fields, 'to_ban_do'):
        extra = run_land_header_fallback(image, candidates)
        if extra:
            candidates.extend(extra)
            fields, sources, debug = fuse_fields_across_passes(candidates)
            fields = clean_final_fields(fields)
    if field_missing_or_bad(fields, 'dien_tich_m2'):
        extra = run_targeted_field_fallback(image, 'dien_tich_m2', candidates)
        if extra:
            candidates.extend(extra)
            fields, sources, debug = fuse_fields_across_passes(candidates)
            fields = clean_final_fields(fields)
    text_keys = ['dia_chi', 'hinh_thuc_su_dung', 'muc_dich_su_dung', 'thoi_han_su_dung', 'nguon_goc_su_dung']
    bad_text = [k for k in text_keys if field_missing_or_bad(fields, k)]
    if len(bad_text) >= 3:
        block = run_textblock_fallback(image)
        if block is not None:
            candidates.append(block)
            fields, sources, debug = fuse_fields_across_passes(candidates)
            fields = clean_final_fields(fields)
    for field_name in text_keys:
        if field_missing_or_bad(fields, field_name):
            extra = run_targeted_field_fallback(image, field_name, candidates)
            if extra:
                candidates.extend(extra)
                fields, sources, debug = fuse_fields_across_passes(candidates)
                fields = clean_final_fields(fields)
    semantic_dates = []
    for cand in candidates:
        d = extract_contract_date_semantic(cand.get('cleaned_text', ''))
        if d:
            semantic_dates.append((float(cand.get('avg_conf', 0.0)), d, cand.get('pass', '')))
    if semantic_dates:
        semantic_dates.sort(reverse=True)
        _, d, p = semantic_dates[0]
        fields['ngay_hop_dong'] = d
        sources['ngay_hop_dong'] = {'pass': f'Semantic context: {p}', 'confidence': semantic_dates[0][0], 'score': None}
    else:
        fields['ngay_hop_dong'] = None
        sources.pop('ngay_hop_dong', None)
    fields = clean_final_fields(fields)
    best = choose_best_overall_candidate(candidates)
    stats = {'so_lan_ocr': len(candidates), 'fullpage_passes': sum((1 for c in candidates if c.get('roi_kind') is None)), 'roi_passes': sum((1 for c in candidates if c.get('roi_kind') is not None)), 'version': 'V5 Field-ROI + Semantic Validation'}
    return (candidates, best, fields, sources, debug, stats)

def validate_extracted_fields(fields):
    checks = []
    area = fields.get('dien_tich_m2')
    purposes = fields.get('muc_dich_chi_tiet') or []
    purpose_areas = [p.get('dien_tich_m2') for p in purposes if p.get('dien_tich_m2') is not None]
    if area is not None and len(purpose_areas) >= 2:
        total = sum(purpose_areas)
        diff = abs(total - area)
        if diff <= max(0.5, area * 0.01):
            checks.append({'type': 'success', 'text': f'Tổng diện tích theo mục đích sử dụng ({total:g} m²) khớp diện tích thửa ({area:g} m²).'})
        else:
            checks.append({'type': 'warning', 'text': f'Tổng diện tích thành phần ({total:g} m²) lệch diện tích thửa ({area:g} m²). Cần kiểm tra OCR.'})
    if fields.get('so_thua') is None:
        checks.append({'type': 'warning', 'text': 'Chưa trích xuất được Số thửa.'})
    if fields.get('to_ban_do') is None:
        checks.append({'type': 'warning', 'text': 'Chưa trích xuất được Tờ bản đồ.'})
    if fields.get('dien_tich_m2') is None:
        checks.append({'type': 'warning', 'text': 'Chưa trích xuất được Diện tích.'})
    return checks
V8_TAMPER_VERSION = 'V8_HYBRID_LOCAL_COPYMOVE'

def _decode_image_bytes(file_bytes):
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _encode_png(img):
    if img is None:
        return None
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes() if ok else None

def _resize_for_tamper(image, target_long=1400):
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= 0:
        return (image.copy(), 1.0)
    scale = min(1.0, target_long / float(longest))
    if longest < 900:
        scale = min(1.35, 1200.0 / float(longest))
    new_w = max(64, int(round(w * scale)))
    new_h = max(64, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return (cv2.resize(image, (new_w, new_h), interpolation=interp), scale)

def _robust_z(v, center, mad):
    denom = max(1.4826 * mad, 1e-06)
    return (v - center) / denom

def _image_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def detect_blur_pii_mask(gray, block=32):
    h, w = gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    lap_vals = []
    grad_vals = []
    coords = []
    for y in range(0, h, block):
        for x in range(0, w, block):
            y2 = min(h, y + block)
            x2 = min(w, x + block)
            if y2 - y < block // 2 or x2 - x < block // 2:
                continue
            lv = float(np.var(lap[y:y2, x:x2]))
            gv = float(np.mean(grad[y:y2, x:x2]))
            lap_vals.append(lv)
            grad_vals.append(gv)
            coords.append((x, y, x2, y2, lv, gv))
    if not coords:
        return mask
    lap_thr = max(8.0, float(np.percentile(lap_vals, 18)))
    grad_thr = max(5.0, float(np.percentile(grad_vals, 22)))
    for x, y, x2, y2, lv, gv in coords:
        if lv <= lap_thr and gv <= grad_thr:
            mask[y:y2, x:x2] = 255
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask

def compute_patch_anomaly(image, blur_mask, patch=64, stride=32):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    entries = []
    for y in range(0, max(1, h - patch + 1), stride):
        for x in range(0, max(1, w - patch + 1), stride):
            roi = gray[y:y + patch, x:x + patch]
            if roi.shape[0] < patch or roi.shape[1] < patch:
                continue
            bm = blur_mask[y:y + patch, x:x + patch]
            blur_fraction = float(np.mean(bm > 0))
            edges = cv2.Canny(roi, 50, 150)
            edge_density = float(np.mean(edges > 0))
            lap_var = float(np.var(cv2.Laplacian(roi, cv2.CV_64F)))
            std = float(np.std(roi))
            mean = float(np.mean(roi))
            entropy = _image_entropy(roi)
            f = roi.astype(np.float32)
            f -= float(np.mean(f))
            spec = np.fft.fftshift(np.fft.fft2(f))
            power = np.abs(spec) ** 2
            yy, xx = np.ogrid[:patch, :patch]
            cy = patch / 2.0
            cx = patch / 2.0
            rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            hf = rr >= 0.28 * float(rr.max())
            fft_high = float(np.sum(power[hf]) / (np.sum(power) + 1e-12))
            entries.append({'x': x, 'y': y, 'mean': mean, 'std': std, 'entropy': entropy, 'edge_density': edge_density, 'lap_log': float(np.log1p(lap_var)), 'fft_high': fft_high, 'blur_fraction': blur_fraction})
    if not entries:
        return (np.zeros((h, w), dtype=np.float32), [], 0.0)
    baseline = [e for e in entries if e['blur_fraction'] < 0.2]
    if len(baseline) < 8:
        baseline = entries
    feat_names = ['std', 'entropy', 'edge_density', 'lap_log', 'fft_high']
    stats = {}
    for k in feat_names:
        vals = np.asarray([e[k] for e in baseline], dtype=float)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        stats[k] = (med, mad)
    amap = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    scores = []
    for e in entries:
        if e['blur_fraction'] >= 0.45:
            score = 0.0
        else:
            zs = []
            for k in feat_names:
                med, mad = stats[k]
                z = abs(_robust_z(e[k], med, mad))
                zs.append(min(z, 8.0))
            score = 0.15 * zs[0] + 0.15 * zs[1] + 0.25 * zs[2] + 0.25 * zs[3] + 0.2 * zs[4]
            if e['mean'] > 205 and e['std'] < 45 and (e['edge_density'] > 0.035):
                score += 1.5
        e['anomaly_score'] = float(score)
        scores.append(score)
        x, y = (e['x'], e['y'])
        amap[y:y + patch, x:x + patch] += score
        count[y:y + patch, x:x + patch] += 1.0
    amap = np.divide(amap, np.maximum(count, 1.0), out=np.zeros_like(amap), where=count > 0)
    non_blur_scores = [e['anomaly_score'] for e in entries if e['blur_fraction'] < 0.45]
    local_peak = float(np.percentile(non_blur_scores, 95)) if non_blur_scores else 0.0
    return (amap, entries, local_peak)

def detect_rectangular_paste(image, blur_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    img_area = float(h * w)
    edges = cv2.Canny(gray, 60, 170)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 0.003 * img_area or area > 0.4 * img_area:
            continue
        if bw < 40 or bh < 18:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) < 4 or len(approx) > 8:
            continue
        rect_area = float(max(bw * bh, 1))
        extent = float(cv2.contourArea(c) / rect_area)
        pad = max(2, int(min(bw, bh) * 0.08))
        x1, y1 = (x + pad, y + pad)
        x2, y2 = (x + bw - pad, y + bh - pad)
        if x2 <= x1 or y2 <= y1:
            continue
        inner = gray[y1:y2, x1:x2]
        ox1 = max(0, x - pad * 2)
        oy1 = max(0, y - pad * 2)
        ox2 = min(w, x + bw + pad * 2)
        oy2 = min(h, y + bh + pad * 2)
        outer = gray[oy1:oy2, ox1:ox2]
        if outer.size == 0 or inner.size == 0:
            continue
        blur_frac = float(np.mean(blur_mask[y:y + bh, x:x + bw] > 0))
        if blur_frac > 0.45:
            continue
        inner_mean = float(np.mean(inner))
        inner_std = float(np.std(inner))
        outer_mean = float(np.mean(outer))
        outer_std = float(np.std(outer))
        contrast = abs(inner_mean - outer_mean) / 80.0
        texture_jump = abs(inner_std - outer_std) / 60.0
        white_box_bonus = 0.0
        if inner_mean > 205 and inner_std < 55:
            white_box_bonus = 0.65
        score = 0.3 * min(extent / 0.55, 1.0) + 0.3 * min(contrast, 1.0) + 0.25 * min(texture_jump, 1.0) + white_box_bonus
        score = float(min(score, 1.0))
        if score >= 0.42:
            boxes.append((x, y, bw, bh, score))
    boxes = sorted(boxes, key=lambda z: z[4], reverse=True)
    kept = []
    for b in boxes:
        x, y, bw, bh, s = b
        accept = True
        for k in kept:
            kx, ky, kw, kh, ks = k
            ix1 = max(x, kx)
            iy1 = max(y, ky)
            ix2 = min(x + bw, kx + kw)
            iy2 = min(y + bh, ky + kh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = bw * bh + kw * kh - inter
            if union > 0 and inter / union > 0.35:
                accept = False
                break
        if accept:
            kept.append(b)
        if len(kept) >= 6:
            break
    rect_score = max([b[4] for b in kept], default=0.0)
    return (float(rect_score), kept)

def detect_copy_move(image, blur_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    diag = max(float(np.hypot(h, w)), 1.0)
    detector_name = 'ORB'
    try:
        detector = cv2.SIFT_create(nfeatures=2200, contrastThreshold=0.035, edgeThreshold=12)
        norm_type = cv2.NORM_L2
        detector_name = 'SIFT'
    except Exception:
        detector = cv2.ORB_create(nfeatures=2600, scaleFactor=1.2, nlevels=8, fastThreshold=20)
        norm_type = cv2.NORM_HAMMING
    kps, desc = detector.detectAndCompute(gray, None)
    if desc is None or len(kps) < 20:
        return {'score': 0.0, 'match_count': 0, 'cluster_size': 0, 'detector': detector_name, 'pairs': []}
    matcher = cv2.BFMatcher(norm_type)
    knn = matcher.knnMatch(desc, desc, k=4)
    pairs = []
    for neigh in knn:
        if len(neigh) < 2:
            continue
        candidates = [m for m in neigh if m.queryIdx != m.trainIdx]
        if not candidates:
            continue
        m = candidates[0]
        p1 = np.asarray(kps[m.queryIdx].pt, dtype=float)
        p2 = np.asarray(kps[m.trainIdx].pt, dtype=float)
        spatial = float(np.linalg.norm(p2 - p1))
        if spatial < 0.08 * diag:
            continue
        x1, y1 = (int(round(p1[0])), int(round(p1[1])))
        x2, y2 = (int(round(p2[0])), int(round(p2[1])))
        if 0 <= x1 < w and 0 <= y1 < h and (0 <= x2 < w) and (0 <= y2 < h):
            if blur_mask[y1, x1] > 0 or blur_mask[y2, x2] > 0:
                continue
        if detector_name == 'SIFT':
            if m.distance > 180:
                continue
        elif m.distance > 55:
            continue
        dx, dy = p2 - p1
        if dx < 0 or (abs(dx) < 1e-06 and dy < 0):
            dx, dy = (-dx, -dy)
            p1, p2 = (p2, p1)
        pairs.append({'p1': p1, 'p2': p2, 'dx': float(dx), 'dy': float(dy), 'distance': float(m.distance)})
    if len(pairs) < 6:
        return {'score': 0.0, 'match_count': len(pairs), 'cluster_size': 0, 'detector': detector_name, 'pairs': []}
    bin_size = max(10.0, 0.012 * diag)
    clusters = {}
    for p in pairs:
        key = (int(round(p['dx'] / bin_size)), int(round(p['dy'] / bin_size)))
        clusters.setdefault(key, []).append(p)
    best_cluster = max(clusters.values(), key=len)
    cluster_size = len(best_cluster)
    pts = []
    for p in best_cluster:
        pts.append(p['p1'])
        pts.append(p['p2'])
    pts = np.asarray(pts, dtype=float)
    if len(pts):
        minxy = np.min(pts, axis=0)
        maxxy = np.max(pts, axis=0)
        coverage = float((maxxy[0] - minxy[0]) * (maxxy[1] - minxy[1]) / max(h * w, 1))
    else:
        coverage = 0.0
    cluster_term = min(cluster_size / 18.0, 1.0)
    pair_term = min(len(pairs) / 45.0, 1.0)
    if coverage > 0.65:
        coverage_term = 0.35
    else:
        coverage_term = min(coverage / 0.18, 1.0)
    score = 0.6 * cluster_term + 0.2 * pair_term + 0.2 * coverage_term
    return {'score': float(min(score, 1.0)), 'match_count': int(len(pairs)), 'cluster_size': int(cluster_size), 'coverage': float(coverage), 'detector': detector_name, 'pairs': best_cluster[:80]}

def detect_seam_score(image, blur_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    diag = max(float(np.hypot(h, w)), 1.0)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, strong = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strong[blur_mask > 0] = 0
    lines = cv2.HoughLinesP(strong, 1, np.pi / 180, threshold=50, minLineLength=max(35, int(0.08 * diag)), maxLineGap=6)
    if lines is None:
        return (0.0, [])
    scored = []
    for line in lines[:250]:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 0.08 * diag:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = min(angle, abs(180 - angle))
        axis = min(abs(angle), abs(90 - angle))
        axis_bonus = max(0.0, 1.0 - axis / 20.0)
        score = min(length / (0.25 * diag), 1.0)
        score *= 0.55 + 0.45 * axis_bonus
        scored.append((x1, y1, x2, y2, float(score)))
    scored = sorted(scored, key=lambda z: z[4], reverse=True)[:20]
    if not scored:
        return (0.0, [])
    top = [x[4] for x in scored[:5]]
    seam_score = float(np.mean(top))
    return (float(min(seam_score, 1.0)), scored)

def _morphological_skeleton(binary):
    img = (binary > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    max_iter = max(img.shape[:2]) + 10
    for _ in range(max_iter):
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel

def _segment_angle_deg(x1, y1, x2, y2):
    a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    a = a % 180.0
    return a

def _angle_diff_180(a, b):
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)

def _point_line_distance(px, py, x1, y1, x2, y2):
    vx = float(x2 - x1)
    vy = float(y2 - y1)
    denom = max(np.hypot(vx, vy), 1e-06)
    return abs(vy * px - vx * py + x2 * y1 - y2 * x1) / denom

def _projection_overlap_ratio(seg_a, seg_b):
    ax1, ay1, ax2, ay2 = seg_a[:4]
    bx1, by1, bx2, by2 = seg_b[:4]
    v = np.asarray([ax2 - ax1, ay2 - ay1], dtype=float)
    nv = float(np.linalg.norm(v))
    if nv < 1e-06:
        return 0.0
    u = v / nv
    origin = np.asarray([ax1, ay1], dtype=float)
    a0 = 0.0
    a1 = nv
    b0 = float(np.dot(np.asarray([bx1, by1], dtype=float) - origin, u))
    b1 = float(np.dot(np.asarray([bx2, by2], dtype=float) - origin, u))
    if b0 > b1:
        b0, b1 = (b1, b0)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = max(min(nv, abs(b1 - b0)), 1e-06)
    return float(inter / shorter)

def _line_darkness(gray, x1, y1, x2, y2, thickness=3):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness)
    vals = gray[mask > 0]
    if vals.size == 0:
        return 0.0
    return float(1.0 - np.mean(vals) / 255.0)

def _estimate_diagram_bottom(gray):
    h, w = gray.shape[:2]
    default_bottom = int(round(0.72 * h))
    edge = cv2.Canny(gray, 50, 150)
    row_density = np.mean(edge > 0, axis=1)
    y_start = int(round(0.48 * h))
    y_end = int(round(0.9 * h))
    candidate_rows = [y for y in range(y_start, y_end) if row_density[y] >= 0.27]
    if candidate_rows:
        window = max(55, int(round(0.13 * h)))
        best_rows = []
        for y0 in candidate_rows:
            rows = [y for y in candidate_rows if y0 <= y <= y0 + window]
            if len(rows) > len(best_rows):
                best_rows = rows
        if len(best_rows) >= 6:
            detected = int(round(min(best_rows) - max(5, 0.01 * h)))
            return int(np.clip(detected, int(0.46 * h), default_bottom))
    lines = cv2.HoughLinesP(edge, 1, np.pi / 180.0, threshold=max(40, int(0.07 * w)), minLineLength=max(70, int(0.55 * w)), maxLineGap=max(8, int(0.025 * w)))
    ys = []
    if lines is not None:
        for item in lines[:300]:
            x1, y1, x2, y2 = map(int, item[0])
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < 0.55 * w:
                continue
            angle = _segment_angle_deg(x1, y1, x2, y2)
            axis = min(abs(angle), abs(180.0 - angle))
            if axis > 4.0:
                continue
            ymid = 0.5 * (y1 + y2)
            if 0.48 * h <= ymid <= 0.88 * h:
                ys.append(ymid)
    if ys:
        detected = int(round(min(ys) - max(6, 0.012 * h)))
        return int(np.clip(detected, int(0.46 * h), default_bottom))
    return default_bottom

def _line_local_contrast(gray, x1, y1, x2, y2):
    h, w = gray.shape[:2]
    vx = float(x2 - x1)
    vy = float(y2 - y1)
    length = float(np.hypot(vx, vy))
    if length < 2:
        return 0.0
    nx = -vy / length
    ny = vx / length
    samples = max(12, int(length / 4.0))
    ts = np.linspace(0.05, 0.95, samples)
    center_vals = []
    side_vals = []
    for t in ts:
        px = x1 + t * vx
        py = y1 + t * vy
        xi = int(round(px))
        yi = int(round(py))
        if 1 <= xi < w - 1 and 1 <= yi < h - 1:
            center_vals.append(float(gray[yi, xi]))
        for off in (5.0, -5.0):
            sx = int(round(px + off * nx))
            sy = int(round(py + off * ny))
            if 0 <= sx < w and 0 <= sy < h:
                side_vals.append(float(gray[sy, sx]))
    if not center_vals or not side_vals:
        return 0.0
    center = float(np.median(center_vals))
    side = float(np.median(side_vals))
    return float(np.clip((side - center) / 90.0, 0.0, 1.0))

def _parallel_neighbor_count(seg, segments, max_angle=5.0, max_sep=55.0):
    count = 0
    for other in segments:
        if other is seg:
            continue
        if _angle_diff_180(seg['angle'], other['angle']) > max_angle:
            continue
        sep = _point_line_distance(other['mx'], other['my'], seg['x1'], seg['y1'], seg['x2'], seg['y2'])
        if 3.0 <= sep <= max_sep:
            overlap = _projection_overlap_ratio((seg['x1'], seg['y1'], seg['x2'], seg['y2']), (other['x1'], other['y1'], other['x2'], other['y2']))
            if overlap >= 0.35:
                count += 1
    return count

def _segment_structural_support(seg, segments, radius=18.0):
    support = 0
    x1, y1, x2, y2 = (seg['x1'], seg['y1'], seg['x2'], seg['y2'])
    for other in segments:
        if other is seg:
            continue
        ad = _angle_diff_180(seg['angle'], other['angle'])
        if ad < 12.0 or ad > 168.0:
            continue
        d1 = _point_line_distance(other['x1'], other['y1'], x1, y1, x2, y2)
        d2 = _point_line_distance(other['x2'], other['y2'], x1, y1, x2, y2)
        d3 = _point_line_distance(x1, y1, other['x1'], other['y1'], other['x2'], other['y2'])
        d4 = _point_line_distance(x2, y2, other['x1'], other['y1'], other['x2'], other['y2'])
        if min(d1, d2, d3, d4) <= radius:
            ax0, ax1 = sorted((x1, x2))
            ay0, ay1 = sorted((y1, y2))
            bx0, bx1 = sorted((other['x1'], other['x2']))
            by0, by1 = sorted((other['y1'], other['y2']))
            if not (ax1 + radius < bx0 or bx1 + radius < ax0 or ay1 + radius < by0 or (by1 + radius < ay0)):
                support += 1
                if support >= 4:
                    break
    return support

def _resize_for_geometry(image, target_long=1100):
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= 0:
        return (image.copy(), 1.0)
    scale = min(1.0, target_long / float(longest))
    if abs(scale - 1.0) < 1e-06:
        return (image.copy(), 1.0)
    nw = max(64, int(round(w * scale)))
    nh = max(64, int(round(h * scale)))
    return (cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA), scale)

def _scale_geometry_candidates(candidates, factor):
    if abs(float(factor) - 1.0) < 1e-09:
        return [dict(c) for c in candidates]
    out = []
    f = float(factor)
    for c in candidates:
        d = dict(c)
        if 'line' in d:
            x1, y1, x2, y2 = d['line']
            d['line'] = (x1 * f, y1 * f, x2 * f, y2 * f)
        if 'other_line' in d:
            x1, y1, x2, y2 = d['other_line']
            d['other_line'] = (x1 * f, y1 * f, x2 * f, y2 * f)
        if 'gap' in d:
            x1, y1, x2, y2 = d['gap']
            d['gap'] = (x1 * f, y1 * f, x2 * f, y2 * f)
        if 'center' in d:
            x, y = d['center']
            d['center'] = (x * f, y * f)
        out.append(d)
    return out

def detect_diagram_geometry_locator(image, blur_mask, heat_norm=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    diag = max(float(np.hypot(h, w)), 1.0)
    diagram_y_max = _estimate_diagram_bottom(gray)
    if diagram_y_max < 80:
        diagram_y_max = h
    geometry_y_max = max(80, diagram_y_max - max(28, int(round(0.14 * h))))
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    bw[blur_mask > 0] = 0
    zone = np.zeros_like(bw)
    margin_x = max(4, int(0.02 * w))
    margin_y = max(4, int(0.02 * h))
    zone[margin_y:geometry_y_max, margin_x:max(margin_x + 1, w - margin_x)] = 255
    bw = cv2.bitwise_and(bw, zone)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    skel = _morphological_skeleton(bw)
    lines = cv2.HoughLinesP(skel, 1, np.pi / 180.0, threshold=max(18, int(0.02 * diag)), minLineLength=max(16, int(0.02 * diag)), maxLineGap=max(4, int(0.01 * diag)))
    if lines is None:
        return ([], skel)
    segments = []
    for item in lines[:500]:
        x1, y1, x2, y2 = map(int, item[0])
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 0.02 * diag or length > 0.48 * diag:
            continue
        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)
        if my >= geometry_y_max:
            continue
        angle = _segment_angle_deg(x1, y1, x2, y2)
        darkness = _line_darkness(gray, x1, y1, x2, y2, thickness=3)
        local_contrast = _line_local_contrast(gray, x1, y1, x2, y2)
        if darkness < 0.18:
            continue
        segments.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'length': length, 'angle': angle, 'darkness': darkness, 'local_contrast': local_contrast, 'mx': mx, 'my': my})
    if not segments:
        return ([], skel)
    support_radius = max(14.0, 0.02 * diag)
    for seg in segments:
        seg['struct_support'] = _segment_structural_support(seg, segments, radius=support_radius)
    candidates = []
    for s in segments:
        if not 0.025 * diag <= s['length'] <= 0.095 * diag:
            continue
        if s['my'] >= geometry_y_max:
            continue
        if s.get('struct_support', 0) < 1:
            continue
        axis = min(abs(s['angle'] - 0.0), abs(s['angle'] - 90.0), abs(s['angle'] - 180.0))
        axis_penalty = 0.12 if axis < 3.0 and s['my'] > 0.52 * h else 0.0
        heat_bonus = 0.0
        length_term = 1.0 - abs(s['length'] - 0.055 * diag) / max(0.055 * diag, 1.0)
        length_term = float(np.clip(length_term, 0.0, 1.0))
        score = 0.34 * min(s['darkness'] / 0.6, 1.0) + 0.28 * min(s['local_contrast'] / 0.75, 1.0) + 0.24 * length_term + heat_bonus - axis_penalty
        if score >= 0.58:
            candidates.append({'type': 'short_line_add', 'score': float(np.clip(score, 0.0, 1.0)), 'line': (s['x1'], s['y1'], s['x2'], s['y2']), 'center': (s['mx'], s['my'])})
    min_sep = max(5.0, 0.006 * diag)
    max_sep = max(24.0, 0.04 * diag)
    for i in range(len(segments)):
        a = segments[i]
        if a['length'] < 0.05 * diag:
            continue
        for j in range(i + 1, len(segments)):
            b = segments[j]
            if b['length'] < 0.05 * diag:
                continue
            adiff = _angle_diff_180(a['angle'], b['angle'])
            if adiff > 5.0:
                continue
            sep = _point_line_distance(b['mx'], b['my'], a['x1'], a['y1'], a['x2'], a['y2'])
            if not min_sep <= sep <= max_sep:
                continue
            overlap = _projection_overlap_ratio((a['x1'], a['y1'], a['x2'], a['y2']), (b['x1'], b['y1'], b['x2'], b['y2']))
            if overlap < 0.45:
                continue
            if a['length'] <= b['length']:
                cand, other = (a, b)
            else:
                cand, other = (b, a)
            if cand['my'] >= geometry_y_max:
                continue
            if max(cand.get('struct_support', 0), other.get('struct_support', 0)) < 1:
                continue
            darkness_term = min(cand['darkness'] / 0.6, 1.0)
            contrast_term = min(cand['local_contrast'] / 0.75, 1.0)
            sep_term = 1.0 - abs(sep - 0.02 * diag) / max(0.02 * diag, 1.0)
            sep_term = float(np.clip(sep_term, 0.0, 1.0))
            heat_bonus = 0.0
            neighbor_count = _parallel_neighbor_count(cand, segments)
            density_penalty = min(max(neighbor_count - 2, 0) * 0.07, 0.28)
            if neighbor_count >= 4 and cand.get('struct_support', 0) >= 3:
                density_penalty += 0.12
            score = 0.27 * overlap + 0.23 * darkness_term + 0.23 * contrast_term + 0.15 * sep_term + heat_bonus - density_penalty
            if score >= 0.62:
                candidates.append({'type': 'parallel_line_add', 'score': float(np.clip(score, 0.0, 1.0)), 'line': (cand['x1'], cand['y1'], cand['x2'], cand['y2']), 'other_line': (other['x1'], other['y1'], other['x2'], other['y2']), 'center': (cand['mx'], cand['my'])})
    gap_min = max(4.0, 0.004 * diag)
    gap_max = max(22.0, 0.032 * diag)
    for i in range(len(segments)):
        a = segments[i]
        if a['length'] < 0.035 * diag:
            continue
        for j in range(i + 1, len(segments)):
            b = segments[j]
            if b['length'] < 0.035 * diag:
                continue
            if _angle_diff_180(a['angle'], b['angle']) > 7.0:
                continue
            endpoints_a = [(a['x1'], a['y1']), (a['x2'], a['y2'])]
            endpoints_b = [(b['x1'], b['y1']), (b['x2'], b['y2'])]
            best_pair = None
            best_gap = 1000000000.0
            for pa in endpoints_a:
                for pb in endpoints_b:
                    gap = float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
                    if gap < best_gap:
                        best_gap = gap
                        best_pair = (pa, pb)
            if best_pair is None or not gap_min <= best_gap <= gap_max:
                continue
            pa, pb = best_pair
            gap_cy = 0.5 * (pa[1] + pb[1])
            if gap_cy >= geometry_y_max:
                continue
            if max(a.get('struct_support', 0), b.get('struct_support', 0)) < 1:
                continue
            bridge_angle = _segment_angle_deg(pa[0], pa[1], pb[0], pb[1])
            align = min(_angle_diff_180(bridge_angle, a['angle']), _angle_diff_180(bridge_angle, b['angle']))
            if align > 12.0:
                continue
            col_a = _point_line_distance(b['mx'], b['my'], a['x1'], a['y1'], a['x2'], a['y2'])
            col_b = _point_line_distance(a['mx'], a['my'], b['x1'], b['y1'], b['x2'], b['y2'])
            if max(col_a, col_b) > max(8.0, 0.012 * diag):
                continue
            gap_term = 1.0 - (best_gap - gap_min) / max(gap_max - gap_min, 1.0)
            align_term = 1.0 - align / 12.0
            darkness_term = min((a['darkness'] + b['darkness']) / 1.0, 1.0)
            contrast_term = min((a['local_contrast'] + b['local_contrast']) / 1.0, 1.0)
            heat_bonus = 0.0
            score = 0.32 * gap_term + 0.28 * align_term + 0.18 * darkness_term + 0.14 * contrast_term + heat_bonus
            if score >= 0.62:
                candidates.append({'type': 'line_break', 'score': float(np.clip(score, 0.0, 1.0)), 'gap': (pa[0], pa[1], pb[0], pb[1]), 'center': ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)})
    candidates = sorted(candidates, key=lambda c: c['score'], reverse=True)
    kept = []
    min_center_dist = max(18.0, 0.028 * diag)
    for c in candidates:
        cx, cy = c['center']
        duplicate = False
        for k in kept:
            kx, ky = k['center']
            if np.hypot(cx - kx, cy - ky) < min_center_dist:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(c)
        if len(kept) >= 4:
            break
    return (kept, skel)

def draw_diagram_geometry_candidates(overlay, candidates):
    if overlay is None or not candidates:
        return overlay
    out = overlay
    c = candidates[0]
    typ = c.get('type')
    score = float(c.get('score', 0.0))
    if typ == 'line_break':
        x1, y1, x2, y2 = c['gap']
        cv2.circle(out, (int(x1), int(y1)), 8, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (int(x2), int(y2)), 8, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3, cv2.LINE_AA)
        tx, ty = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    else:
        x1, y1, x2, y2 = c['line']
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 5, cv2.LINE_AA)
        tx, ty = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    cv2.putText(out, f'geometry {score:.2f}', (max(4, tx + 6), max(18, ty - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 0, 255), 2, cv2.LINE_AA)
    return out

def _geometry_display_map(shape, candidates):
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=np.float32)
    if not candidates:
        return out
    c = candidates[0]
    score = float(np.clip(c.get('score', 0.0), 0.0, 1.0))
    mask = np.zeros((h, w), dtype=np.uint8)
    if c.get('type') == 'line_break':
        x1, y1, x2, y2 = c['gap']
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 10, cv2.LINE_AA)
        cv2.circle(mask, (int(x1), int(y1)), 12, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, (int(x2), int(y2)), 12, 255, -1, cv2.LINE_AA)
    else:
        x1, y1, x2, y2 = c['line']
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 12, cv2.LINE_AA)
    sigma = max(3.0, 0.006 * max(h, w))
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigma)
    if np.max(soft) > 0:
        soft /= float(np.max(soft))
    out = soft * max(0.75, score)
    return out.astype(np.float32)

def analyze_tamper_hybrid(image):
    work, scale = _resize_for_tamper(image, target_long=1400)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur_mask = detect_blur_pii_mask(gray)
    amap, patch_entries, local_peak = compute_patch_anomaly(work, blur_mask, patch=64, stride=32)
    local_score = float(1.0 / (1.0 + np.exp(-1.15 * (local_peak - 3.4))))
    rect_score, rect_boxes = detect_rectangular_paste(work, blur_mask)
    cm = detect_copy_move(work, blur_mask)
    copy_score = float(cm['score'])
    seam_score, seam_lines = detect_seam_score(work, blur_mask)
    strong_override = max(rect_score, copy_score)
    score = 0.34 * local_score + 0.28 * rect_score + 0.28 * copy_score + 0.1 * seam_score
    if strong_override >= 0.82:
        score = max(score, 0.72)
    elif strong_override >= 0.68:
        score = max(score, 0.6)
    score = float(np.clip(score, 0.0, 1.0))
    if score >= 0.62:
        status = 'TAMPERED'
    elif score >= 0.4:
        status = 'UNCERTAIN'
    else:
        status = 'REAL'
    vis_map, _, _ = compute_patch_anomaly(work, blur_mask, patch=48, stride=16)
    heat = vis_map.copy()
    positive_vals = heat[heat > 0]
    if positive_vals.size:
        low = float(np.percentile(positive_vals, 85))
        high = float(np.percentile(positive_vals, 98))
        if high <= low:
            high = low + 1e-06
        heat_norm = np.clip((heat - low) / (high - low), 0, 1)
    else:
        heat_norm = np.zeros_like(heat, dtype=np.float32)
    heat_norm = heat_norm.astype(np.float32)
    heat_norm[blur_mask > 0] = 0.0
    geometry_image, geometry_scale = _resize_for_geometry(image, target_long=1100)
    geometry_gray = cv2.cvtColor(geometry_image, cv2.COLOR_BGR2GRAY)
    geometry_blur_mask = detect_blur_pii_mask(geometry_gray)
    geometry_raw, geometry_skeleton = detect_diagram_geometry_locator(geometry_image, geometry_blur_mask, heat_norm=None)
    geometry_factor = float(scale) / max(float(geometry_scale), 1e-06)
    geometry_candidates = _scale_geometry_candidates(geometry_raw, geometry_factor)
    geometry_map = _geometry_display_map(work.shape, geometry_candidates)
    display_heat_norm = np.maximum(0.72 * heat_norm, geometry_map)
    display_heat_norm[blur_mask > 0] = 0.0
    heat_u8 = np.uint8(np.clip(display_heat_norm, 0, 1) * 255)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(work, 0.76, heat_color, 0.24, 0)
    vis_table_top = _estimate_diagram_bottom(gray)
    vis_geometry_y_max = max(80, vis_table_top - max(28, int(round(0.14 * work.shape[0]))))
    blur_contours, _ = cv2.findContours(blur_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in blur_contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < 0.004 * work.shape[0] * work.shape[1]:
            continue
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (80, 200, 80), 2)
    for x, y, bw, bh, s in rect_boxes:
        if s < 0.65:
            continue
        cy = y + 0.5 * bh
        if cy >= vis_geometry_y_max:
            continue
        yy1 = max(0, y)
        yy2 = min(heat_norm.shape[0], y + bh)
        xx1 = max(0, x)
        xx2 = min(heat_norm.shape[1], x + bw)
        region_heat = float(np.max(heat_norm[yy1:yy2, xx1:xx2])) if yy2 > yy1 and xx2 > xx1 else 0.0
        near_geom = False
        if geometry_candidates:
            gx, gy = geometry_candidates[0]['center']
            ex = max(18.0, 0.05 * work.shape[1])
            ey = max(18.0, 0.05 * work.shape[0])
            near_geom = x - ex <= gx <= x + bw + ex and y - ey <= gy <= y + bh + ey
        if geometry_candidates:
            if not near_geom:
                continue
        elif region_heat < 0.28:
            continue
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
        cv2.putText(overlay, f'paste {s:.2f}', (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    for p in cm.get('pairs', [])[:35]:
        p1_arr = np.round(p['p1']).astype(int)
        p2_arr = np.round(p['p2']).astype(int)
        p1 = tuple(p1_arr)
        p2 = tuple(p2_arr)
        x1, y1 = (int(p1_arr[0]), int(p1_arr[1]))
        x2, y2 = (int(p2_arr[0]), int(p2_arr[1]))
        a1 = 0 <= y1 < heat_norm.shape[0] and 0 <= x1 < heat_norm.shape[1] and (heat_norm[y1, x1] >= 0.35)
        a2 = 0 <= y2 < heat_norm.shape[0] and 0 <= x2 < heat_norm.shape[1] and (heat_norm[y2, x2] >= 0.35)
        if not (a1 or a2):
            continue
        cv2.line(overlay, p1, p2, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, p1, 3, (0, 215, 255), -1)
        cv2.circle(overlay, p2, 3, (0, 215, 255), -1)
    for x1, y1, x2, y2, s in seam_lines[:20]:
        if s < 0.75:
            continue
        mx = int(round((x1 + x2) / 2.0))
        my = int(round((y1 + y2) / 2.0))
        if not (0 <= my < heat_norm.shape[0] and 0 <= mx < heat_norm.shape[1] and (heat_norm[my, mx] >= 0.45)):
            continue
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 120, 0), 2, cv2.LINE_AA)
    overlay = draw_diagram_geometry_candidates(overlay, geometry_candidates)
    evidence = []
    if geometry_candidates:
        type_counts = {}
        for gc in geometry_candidates:
            t = gc.get('type', 'geometry')
            type_counts[t] = type_counts.get(t, 0) + 1
        summary = ', '.join((f'{k}:{v}' for k, v in sorted(type_counts.items())))
        evidence.append(f'Geometry locator đánh dấu {len(geometry_candidates)} ứng viên nét cần kiểm tra ({summary}); chỉ dùng để định vị, không cộng vào Hybrid Score')
    if rect_score >= 0.55:
        evidence.append(f'Phát hiện vùng paste/text-box bất thường (score {rect_score:.2f})')
    if copy_score >= 0.55:
        evidence.append(f"Phát hiện dấu hiệu copy-move ({cm['cluster_size']} match cùng displacement)")
    if local_score >= 0.6:
        evidence.append(f'Texture cục bộ lệch nền mạnh (local score {local_score:.2f})')
    if seam_score >= 0.65:
        evidence.append(f'Biên/seam cục bộ mạnh (score {seam_score:.2f})')
    blur_fraction = float(np.mean(blur_mask > 0))
    if blur_fraction > 0.02:
        evidence.append(f'Đã nhận diện {blur_fraction:.1%} vùng che/chi tiết thấp PII và giảm ảnh hưởng khi tính điểm')
    return {'version': V8_TAMPER_VERSION, 'status': status, 'score': score, 'local_score': local_score, 'local_peak_raw': float(local_peak), 'rect_score': float(rect_score), 'copy_move_score': copy_score, 'seam_score': float(seam_score), 'blur_fraction': blur_fraction, 'copy_move_matches': int(cm['match_count']), 'copy_move_cluster': int(cm['cluster_size']), 'copy_move_detector': cm['detector'], 'copy_move_coverage': float(cm.get('coverage', 0.0)), 'geometry_candidate_count': int(len(geometry_candidates)), 'geometry_candidate_types': [c.get('type') for c in geometry_candidates], 'evidence': evidence, 'overlay_bytes': _encode_png(overlay), 'blur_mask_bytes': _encode_png(blur_mask), 'heatmap_bytes': _encode_png(heat_color), 'work_scale': float(scale)}
SEAL_REAL_DIR = Path('dataset/processed_real/seal')
SEAL_LBP_POINTS = 8
SEAL_LBP_RADIUS = 1
SEAL_LBP_BINS = SEAL_LBP_POINTS + 2

def _ocr_field_inconsistency(candidates):
    if not candidates:
        return 0.0
    values = {'so_thua': [], 'to_ban_do': [], 'dien_tich_m2': []}
    for c in candidates:
        if c.get('roi_kind') is not None:
            continue
        fields = extract_land_fields(c.get('cleaned_text', ''))
        for key in values:
            v = fields.get(key)
            if field_is_plausible(key, v):
                if key == 'dien_tich_m2':
                    try:
                        v = f'{float(v):.4f}'
                    except Exception:
                        pass
                values[key].append(str(v).strip())
    compared = 0
    disagree = 0
    for vals in values.values():
        if len(vals) < 2:
            continue
        compared += 1
        if len(set(vals)) > 1:
            disagree += 1
    if compared == 0:
        return 0.0
    return float(disagree / compared)
CORE_TEXT_FIELDS = {'so_thua': {'label': 'Số thửa', 'label_tokens': ['thua dat', 'thua dat so']}, 'to_ban_do': {'label': 'Tờ bản đồ', 'label_tokens': ['to ban do', 'ban do so']}, 'dien_tich_m2': {'label': 'Diện tích', 'label_tokens': ['dien tich']}}
OVERLAY_FIELD_LABELS = {'so_thua': 'So thua', 'to_ban_do': 'To ban do', 'dien_tich_m2': 'Dien tich'}
SYNTHETIC_TEXT_BGR = np.array([220.0, 70.0, 20.0], dtype=np.float32)

def _numeric_equal(a, b, field_name):
    try:
        va = parse_number(a)
        vb = parse_number(b)
        if va is None or vb is None:
            return False
        if field_name in ('so_thua', 'to_ban_do'):
            return int(round(va)) == int(round(vb))
        return abs(float(va) - float(vb)) <= 0.03
    except Exception:
        return False

def _target_value_string(fields, field_name):
    if not fields:
        return None
    value = fields.get(field_name)
    if value in (None, '', []):
        return None
    if field_name == 'dien_tich_m2':
        try:
            return f'{float(value):g}'
        except Exception:
            return str(value)
    return str(value).strip()

def _entry_rect_work(candidate, entry, work_scale):
    try:
        x0, y0, x1, y1 = _entry_rect_original(candidate, entry)
    except Exception:
        return None
    x0 *= work_scale
    x1 *= work_scale
    y0 *= work_scale
    y1 *= work_scale
    if x1 <= x0 or y1 <= y0:
        return None
    return (float(x0), float(y0), float(x1), float(y1))

def _token_bbox_inside_entry(entry_rect, text, token_start, token_end, image_shape):
    h_img, w_img = image_shape[:2]
    x0, y0, x1, y1 = entry_rect
    text_len = max(len(text), 1)
    line_w = max(x1 - x0, 1.0)
    line_h = max(y1 - y0, 10.0)
    sx = x0 + token_start / text_len * line_w
    ex = x0 + token_end / text_len * line_w
    pad_x = max(0.35 * line_h, 3.0)
    pad_y = max(0.22 * line_h, 2.0)
    rx0 = max(0, int(round(sx - pad_x)))
    rx1 = min(w_img, int(round(ex + pad_x)))
    ry0 = max(0, int(round(y0 - pad_y)))
    ry1 = min(h_img, int(round(y1 + pad_y)))
    if rx1 - rx0 < 8 or ry1 - ry0 < 8:
        return None
    return (rx0, ry0, rx1, ry1)

def _field_label_anchor_rect(ocr_candidates, field_name, work_scale):
    found = find_best_label_entry(ocr_candidates, field_name)
    if not found:
        return None
    cand, entry = found
    return _entry_rect_work(cand, entry, work_scale)

def _tighten_numeric_bbox(image_bgr, bbox):
    if bbox is None:
        return None
    h_img, w_img = image_bgr.shape[:2]
    x0, y0, x1, y1 = map(int, bbox)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w_img, x1)
    y1 = min(h_img, y1)
    if x1 <= x0 or y1 <= y0:
        return bbox
    roi = image_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return bbox
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    H, W = gray.shape[:2]
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 3:
            continue
        if h < max(3, int(0.18 * H)):
            continue
        if h > 0.95 * H:
            continue
        if w > 0.95 * W and h > 0.75 * H:
            continue
        comps.append((x, y, w, h, area))
    if not comps:
        return bbox
    cx = W / 2.0
    comps.sort(key=lambda z: abs(z[0] + z[2] / 2.0 - cx))
    seed = comps[0]
    sx0, sy0, sw, sh, _ = seed
    sx1 = sx0 + sw
    sy1 = sy0 + sh
    char_h = max(sh, 4)
    chosen = [seed]
    for c in comps[1:]:
        x, y, w, h, _ = c
        c0 = x
        c1 = x + w
        horizontal_gap = max(0, max(sx0, c0) - min(sx1, c1))
        vertical_overlap = max(0, min(sy1, y + h) - max(sy0, y))
        if horizontal_gap <= 1.5 * char_h and vertical_overlap >= 0.2 * min(sh, h):
            chosen.append(c)
            sx0 = min(sx0, c0)
            sy0 = min(sy0, y)
            sx1 = max(sx1, c1)
            sy1 = max(sy1, y + h)
    pad = max(2, int(round(0.12 * max(sy1 - sy0, 4))))
    return (max(0, x0 + sx0 - pad), max(0, y0 + sy0 - pad), min(w_img, x0 + sx1 + pad), min(h_img, y0 + sy1 + pad))

def _collect_mapped_ocr_entries(ocr_candidates, work_scale):
    allowed_roi_kinds = {None, 'land_header', 'so_thua', 'to_ban_do', 'dien_tich_m2'}
    items = []
    for cand in ocr_candidates or []:
        roi_kind = cand.get('roi_kind')
        if roi_kind not in allowed_roi_kinds:
            continue
        for entry in cand.get('entries', []):
            if not entry.get('box'):
                continue
            rect = _entry_rect_work(cand, entry, work_scale)
            if rect is None:
                continue
            raw = str(entry.get('text', '')).strip()
            if not raw:
                continue
            items.append({'raw': raw, 'norm': normalize_for_match(raw), 'rect': rect, 'ocr_score': float(entry.get('score', 0.0)), 'roi_kind': roi_kind})
    return items

def _header_pair_from_same_line(items, fields, work_shape, work_image=None):
    target_so = _target_value_string(fields, 'so_thua')
    target_to = _target_value_string(fields, 'to_ban_do')
    if target_so is None and target_to is None:
        return {}
    numeric_pattern = re.compile('\\d+(?:[.,]\\d+)?')
    candidates = []
    for item in items:
        raw = item['raw']
        norm = item['norm']
        has_so_label = 'thua dat' in norm
        has_to_label = 'to ban do' in norm or 'ban do so' in norm
        if not (has_so_label or has_to_label):
            continue
        matches = list(numeric_pattern.finditer(raw))
        if not matches:
            continue
        row_score = float(item['ocr_score'])
        if has_so_label:
            row_score += 0.7
        if has_to_label:
            row_score += 0.7
        if item.get('roi_kind') == 'land_header':
            row_score += 0.5
        candidates.append((row_score, item, matches))
    if not candidates:
        return {}
    candidates.sort(key=lambda z: z[0], reverse=True)
    output = {}
    for _, item, matches in candidates:
        raw = item['raw']
        rect = item['rect']
        found_so = []
        found_to = []
        for m in matches:
            token = m.group(0)
            token_box = _token_bbox_inside_entry(rect, raw, m.start(), m.end(), work_shape)
            if token_box is None:
                continue
            if work_image is not None:
                token_box = _tighten_numeric_bbox(work_image, token_box)
            if target_so is not None and _numeric_equal(token, target_so, 'so_thua'):
                found_so.append((m.start(), token_box, token))
            if target_to is not None and _numeric_equal(token, target_to, 'to_ban_do'):
                found_to.append((m.start(), token_box, token))
        if found_so and found_to:
            so_item = min(found_so, key=lambda z: z[0])
            to_item = max(found_to, key=lambda z: z[0])
            output['so_thua'] = {'field': 'so_thua', 'label': CORE_TEXT_FIELDS['so_thua']['label'], 'value': target_so, 'bbox': so_item[1], 'locator_score': 10.0, 'source_text': raw, 'header_locked': True}
            output['to_ban_do'] = {'field': 'to_ban_do', 'label': CORE_TEXT_FIELDS['to_ban_do']['label'], 'value': target_to, 'bbox': to_item[1], 'locator_score': 10.0, 'source_text': raw, 'header_locked': True}
            return output
        if found_so and 'so_thua' not in output:
            so_item = min(found_so, key=lambda z: z[0])
            output['so_thua'] = {'field': 'so_thua', 'label': CORE_TEXT_FIELDS['so_thua']['label'], 'value': target_so, 'bbox': so_item[1], 'locator_score': 8.0, 'source_text': raw, 'header_locked': True}
        if found_to and 'to_ban_do' not in output:
            to_item = max(found_to, key=lambda z: z[0])
            output['to_ban_do'] = {'field': 'to_ban_do', 'label': CORE_TEXT_FIELDS['to_ban_do']['label'], 'value': target_to, 'bbox': to_item[1], 'locator_score': 8.0, 'source_text': raw, 'header_locked': True}
    return output

def _header_pair_by_row_geometry(items, fields, work_shape, work_image=None):
    target_so = _target_value_string(fields, 'so_thua')
    target_to = _target_value_string(fields, 'to_ban_do')
    if target_so is None and target_to is None:
        return {}
    numeric_pattern = re.compile('\\d+(?:[.,]\\d+)?')
    header_rows = []
    for item in items:
        norm = item['norm']
        if 'thua dat' in norm or 'to ban do' in norm or 'ban do so' in norm:
            x0, y0, x1, y1 = item['rect']
            cy = 0.5 * (y0 + y1)
            h = max(y1 - y0, 8.0)
            header_rows.append((float(item['ocr_score']), cy, h, item))
    if not header_rows:
        return {}
    header_rows.sort(key=lambda z: z[0], reverse=True)
    _, row_cy, row_h, _ = header_rows[0]
    token_candidates = []
    for item in items:
        raw = item['raw']
        rect = item['rect']
        x0, y0, x1, y1 = rect
        cy = 0.5 * (y0 + y1)
        h = max(y1 - y0, 8.0)
        if abs(cy - row_cy) > 2.0 * max(row_h, h):
            continue
        matches = list(numeric_pattern.finditer(raw))
        for m in matches:
            token = m.group(0)
            matches_so = target_so is not None and _numeric_equal(token, target_so, 'so_thua')
            matches_to = target_to is not None and _numeric_equal(token, target_to, 'to_ban_do')
            if not (matches_so or matches_to):
                continue
            token_box = _token_bbox_inside_entry(rect, raw, m.start(), m.end(), work_shape)
            if token_box is None:
                continue
            if work_image is not None:
                token_box = _tighten_numeric_bbox(work_image, token_box)
            bx0, by0, bx1, by1 = token_box
            cx = 0.5 * (bx0 + bx1)
            token_candidates.append({'token': token, 'bbox': token_box, 'cx': cx, 'matches_so': matches_so, 'matches_to': matches_to, 'ocr_score': float(item['ocr_score']), 'raw': raw})
    output = {}
    so_candidates = [c for c in token_candidates if c['matches_so']]
    to_candidates = [c for c in token_candidates if c['matches_to']]
    if so_candidates:
        c = min(so_candidates, key=lambda z: z['cx'])
        output['so_thua'] = {'field': 'so_thua', 'label': CORE_TEXT_FIELDS['so_thua']['label'], 'value': target_so, 'bbox': c['bbox'], 'locator_score': 7.0, 'source_text': c['raw'], 'header_locked': True}
    if to_candidates:
        c = max(to_candidates, key=lambda z: z['cx'])
        output['to_ban_do'] = {'field': 'to_ban_do', 'label': CORE_TEXT_FIELDS['to_ban_do']['label'], 'value': target_to, 'bbox': c['bbox'], 'locator_score': 7.0, 'source_text': c['raw'], 'header_locked': True}
    return output

def locate_core_field_boxes(ocr_candidates, fields, work_scale, work_shape, work_image=None):
    h_img, w_img = work_shape[:2]
    all_entries = _collect_mapped_ocr_entries(ocr_candidates, work_scale)
    output = {}
    header = _header_pair_from_same_line(all_entries, fields, work_shape, work_image=work_image)
    output.update(header)
    if 'so_thua' not in output or 'to_ban_do' not in output:
        geom = _header_pair_by_row_geometry(all_entries, fields, work_shape, work_image=work_image)
        for k, v in geom.items():
            output.setdefault(k, v)
    field_name = 'dien_tich_m2'
    meta = CORE_TEXT_FIELDS[field_name]
    target = _target_value_string(fields, field_name)
    if target is not None:
        anchor_rect = _field_label_anchor_rect(ocr_candidates or [], field_name, work_scale)
        anchor_cy = None
        anchor_h = None
        if anchor_rect is not None:
            _, ay0, _, ay1 = anchor_rect
            anchor_cy = 0.5 * (ay0 + ay1)
            anchor_h = max(ay1 - ay0, 8.0)
        numeric_pattern = re.compile('\\d+(?:[.,]\\d+)?')
        found = []
        for item in all_entries:
            raw = item['raw']
            rect = item['rect']
            matches = list(numeric_pattern.finditer(raw))
            if not matches:
                continue
            _, y0, _, y1 = rect
            cy = 0.5 * (y0 + y1)
            line_h = max(y1 - y0, 8.0)
            anchor_bonus = 0.0
            if anchor_cy is not None:
                dy = abs(cy - anchor_cy)
                tolerance = 1.8 * max(anchor_h, line_h)
                if dy > tolerance:
                    continue
                anchor_bonus = 0.9 * max(0.0, 1.0 - dy / max(tolerance, 1e-06))
            for m in matches:
                token = m.group(0)
                if not _numeric_equal(token, target, field_name):
                    continue
                token_box = _token_bbox_inside_entry(rect, raw, m.start(), m.end(), work_shape)
                if token_box is None:
                    continue
                if work_image is not None:
                    token_box = _tighten_numeric_bbox(work_image, token_box)
                label_bonus = 1.0 if 'dien tich' in item['norm'] else 0.0
                roi_bonus = 0.5 if item.get('roi_kind') == 'dien_tich_m2' else 0.0
                locator_score = float(item['ocr_score']) + anchor_bonus + label_bonus + roi_bonus
                found.append({'field': field_name, 'label': meta['label'], 'value': target, 'bbox': token_box, 'locator_score': float(locator_score), 'source_text': raw, 'anchor_rect': anchor_rect})
        if found:
            found.sort(key=lambda z: z['locator_score'], reverse=True)
            output[field_name] = found[0]
    return output

def _nearest_field_for_bbox(bbox, locations, page_shape):
    if bbox is None or not locations:
        return (None, None)
    h, w = page_shape[:2]
    diag = max(float(np.hypot(h, w)), 1.0)
    best_name = None
    best_score = float('inf')
    bx0, by0, bx1, by1 = bbox
    bcx = 0.5 * (bx0 + bx1)
    bcy = 0.5 * (by0 + by1)
    for name, info in locations.items():
        x0, y0, x1, y1 = info['bbox']
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        dx = abs(bcx - cx) / diag
        dy = abs(bcy - cy) / diag
        d = dx + 2.2 * dy
        if d < best_score:
            best_score = d
            best_name = name
    return (best_name, best_score)

def _synthetic_blue_mask(image_bgr):
    pix = image_bgr.astype(np.float32)
    dist = np.linalg.norm(pix - SYNTHETIC_TEXT_BGR.reshape(1, 1, 3), axis=2)
    b = pix[:, :, 0]
    g = pix[:, :, 1]
    r = pix[:, :, 2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    near_exact = (dist <= 85.0) & (b >= 105) & (b - g >= 35) & (b - r >= 65)
    blue_dominant = (b >= 115) & (b - g >= 55) & (b - r >= 85) & (sat >= 75)
    mask = np.uint8(near_exact | blue_dominant) * 255
    return (mask, dist)

def _expand_bbox(bbox, shape, x_pad_ratio=0.55, y_pad_ratio=0.35):
    h_img, w_img = shape[:2]
    x0, y0, x1, y1 = map(int, bbox)
    bh = max(y1 - y0, 8)
    bw = max(x1 - x0, 8)
    px = max(4, int(round(x_pad_ratio * bh)))
    py = max(3, int(round(y_pad_ratio * bh)))
    return (max(0, x0 - px), max(0, y0 - py), min(w_img, x1 + px), min(h_img, y1 + py))

def _gray_text_metrics(gray):
    if gray is None or gray.size == 0:
        return None
    gray = gray.astype(np.uint8)
    lap = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
    bg = float(np.percentile(gray, 80))
    thr = max(60.0, min(190.0, bg - 35.0))
    dark = (gray < thr).astype(np.uint8) * 255
    d1 = cv2.dilate(dark, np.ones((3, 3), np.uint8), iterations=1)
    d2 = cv2.dilate(dark, np.ones((7, 7), np.uint8), iterations=1)
    ring = (d2 > 0) & (d1 == 0)
    halo = float(np.mean(gray[ring] > min(245.0, bg + 20.0))) if np.any(ring) else 0.0
    dark_fraction = float(np.mean(dark > 0))
    return {'lap': lap, 'halo': halo, 'dark_fraction': dark_fraction}

def _left_text_reference_bbox(field_bbox, image_shape):
    h_img, w_img = image_shape[:2]
    x0, y0, x1, y1 = map(int, field_bbox)
    h = max(8, y1 - y0)
    w = max(8, x1 - x0)
    gap = max(2, int(round(0.15 * h)))
    rx1 = max(0, x0 - gap)
    ref_width = max(int(round(4.0 * w)), int(round(5.0 * h)))
    rx0 = max(0, rx1 - ref_width)
    py = max(2, int(round(0.25 * h)))
    ry0 = max(0, y0 - py)
    ry1 = min(h_img, y1 + py)
    if rx1 - rx0 < 12 or ry1 - ry0 < 8:
        return None
    return (rx0, ry0, rx1, ry1)

def _grayscale_field_anomaly(image_bgr, field_bbox):
    h_img, w_img = image_bgr.shape[:2]
    x0, y0, x1, y1 = map(int, field_bbox)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w_img, x1)
    y1 = min(h_img, y1)
    if x1 <= x0 or y1 <= y0:
        return (0.0, None, {})
    ref_bbox = _left_text_reference_bbox((x0, y0, x1, y1), image_bgr.shape)
    if ref_bbox is None:
        return (0.0, None, {})
    rx0, ry0, rx1, ry1 = ref_bbox
    target = image_bgr[y0:y1, x0:x1]
    reference = image_bgr[ry0:ry1, rx0:rx1]
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    tm = _gray_text_metrics(target_gray)
    rm = _gray_text_metrics(ref_gray)
    if tm is None or rm is None:
        return (0.0, None, {})
    lap_ratio = tm['lap'] / max(rm['lap'], 1e-06)
    halo_delta = tm['halo'] - rm['halo']
    dark_delta = tm['dark_fraction'] - rm['dark_fraction']
    smooth_term = float(np.clip((0.82 - lap_ratio) / 0.52, 0.0, 1.0))
    halo_term = float(np.clip((halo_delta - 0.015) / 0.14, 0.0, 1.0))
    sharp_term = float(np.clip((lap_ratio - 1.65) / 1.35, 0.0, 1.0))
    dark_positive = float(np.clip((dark_delta - 0.04) / 0.14, 0.0, 1.0))
    dark_negative = float(np.clip((-dark_delta - 0.035) / 0.14, 0.0, 1.0))
    smooth_score = 0.72 * smooth_term + 0.18 * halo_term + 0.1 * dark_negative
    sharp_support = max(halo_term, dark_positive)
    sharp_score = 0.55 * sharp_term + 0.3 * halo_term + 0.15 * dark_positive
    if sharp_support < 0.2:
        sharp_score *= 0.3
    score = max(smooth_score, sharp_score)
    if lap_ratio <= 0.52:
        score = max(score, 0.78)
    elif lap_ratio <= 0.68:
        score = max(score, 0.66)
    elif lap_ratio <= 0.82 and (halo_delta >= 0.025 or dark_delta <= -0.04):
        score = max(score, 0.52)
    if lap_ratio >= 2.1 and halo_delta >= 0.08:
        score = max(score, 0.68)
    elif lap_ratio >= 1.8 and halo_delta >= 0.05:
        score = max(score, 0.54)
    debug = {'lap_ratio': float(lap_ratio), 'halo_delta': float(halo_delta), 'dark_delta': float(dark_delta), 'smooth_term': float(smooth_term), 'sharp_term': float(sharp_term), 'reference_bbox': ref_bbox}
    return (float(np.clip(score, 0.0, 1.0)), (x0, y0, x1, y1), debug)

def _field_synthetic_cue(image_bgr, field_bbox):
    h_img, w_img = image_bgr.shape[:2]
    ex0, ey0, ex1, ey1 = _expand_bbox(field_bbox, image_bgr.shape, x_pad_ratio=0.65, y_pad_ratio=0.45)
    roi = image_bgr[ey0:ey1, ex0:ex1]
    if roi.size == 0:
        return (0.0, None, {})
    mask, dist = _synthetic_blue_mask(roi)
    raw = mask > 0
    raw_count = int(np.sum(raw))
    raw_fraction = float(np.mean(raw))
    if raw_count > 0:
        dist_values = dist[raw]
        median_dist = float(np.median(dist_values))
        pix = roi.astype(np.float32)
        b = pix[:, :, 0]
        g = pix[:, :, 1]
        r = pix[:, :, 2]
        dominance = b - np.maximum(g, r)
        median_dominance = float(np.median(dominance[raw]))
    else:
        median_dist = 999.0
        median_dominance = 0.0
    grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)), iterations=1)
    grouped = cv2.dilate(grouped, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)), iterations=1)
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = int(bw * bh)
        if area < 6:
            continue
        if bh < 3:
            continue
        if bw > 0.95 * roi.shape[1] and bh > 0.8 * roi.shape[0]:
            continue
        valid.append((x, y, bw, bh, area))
    cue_bbox = None
    largest_component_area = 0
    if valid:
        xs0 = [v[0] for v in valid]
        ys0 = [v[1] for v in valid]
        xs1 = [v[0] + v[2] for v in valid]
        ys1 = [v[1] + v[3] for v in valid]
        bx0 = min(xs0)
        by0 = min(ys0)
        bx1 = max(xs1)
        by1 = max(ys1)
        largest_component_area = max((v[4] for v in valid))
        pad = 3
        cue_bbox = (max(0, ex0 + bx0 - pad), max(0, ey0 + by0 - pad), min(w_img, ex0 + bx1 + pad), min(h_img, ey0 + by1 + pad))
    count_term = float(np.clip((raw_count - 2) / 18.0, 0.0, 1.0))
    fraction_term = float(np.clip((raw_fraction - 0.0015) / 0.035, 0.0, 1.0))
    color_term = float(np.clip((110.0 - median_dist) / 90.0, 0.0, 1.0))
    dominance_term = float(np.clip((median_dominance - 35.0) / 90.0, 0.0, 1.0))
    component_term = float(np.clip((largest_component_area - 3) / 45.0, 0.0, 1.0))
    score = 0.3 * count_term + 0.2 * fraction_term + 0.2 * color_term + 0.15 * dominance_term + 0.15 * component_term
    if raw_count >= 8 and median_dist <= 80 and (median_dominance >= 60) and (cue_bbox is not None):
        score = max(score, 0.72)
    elif raw_count >= 5 and median_dist <= 95 and (cue_bbox is not None):
        score = max(score, 0.52)
    debug = {'raw_blue_pixels': raw_count, 'raw_blue_fraction': raw_fraction, 'median_color_distance': median_dist, 'median_blue_dominance': median_dominance, 'largest_component_area': largest_component_area, 'expanded_bbox': (ex0, ey0, ex1, ey1)}
    return (float(np.clip(score, 0.0, 1.0)), cue_bbox, debug)

def _global_synthetic_blue_candidate(image_bgr, max_y_ratio=0.58):
    h, w = image_bgr.shape[:2]
    y_end = max(1, min(h, int(round(max_y_ratio * h))))
    zone = image_bgr[:y_end].copy()
    mask, dist = _synthetic_blue_mask(zone)
    grouped = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)), iterations=1)
    grouped = cv2.dilate(grouped, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)), iterations=1)
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < 8:
            continue
        if bh < 3:
            continue
        raw_roi = mask[y:y + bh, x:x + bw]
        n = int(np.sum(raw_roi > 0))
        if n < 5:
            continue
        dist_roi = dist[y:y + bh, x:x + bw]
        vals = dist_roi[raw_roi > 0]
        md = float(np.median(vals)) if vals.size else 999.0
        score = 0.6 * float(np.clip((n - 3) / 18.0, 0.0, 1.0)) + 0.4 * float(np.clip((110.0 - md) / 90.0, 0.0, 1.0))
        if n >= 8 and md <= 80:
            score = max(score, 0.68)
        candidates.append({'score': float(np.clip(score, 0.0, 1.0)), 'bbox': (max(0, x - 3), max(0, y - 3), min(w, x + bw + 3), min(h, y + bh + 3)), 'pixel_count': n, 'median_dist': md})
    if not candidates:
        return None
    candidates.sort(key=lambda z: z['score'], reverse=True)
    return candidates[0]

def _snap_display_bbox_to_field_row(image_bgr, bbox, anchor_rect):
    if bbox is None:
        return None
    h_img, w_img = image_bgr.shape[:2]
    x0, y0, x1, y1 = map(int, bbox)
    if anchor_rect is None:
        return (max(0, x0), max(0, y0), min(w_img, x1), min(h_img, y1))
    ax0, ay0, ax1, ay1 = map(float, anchor_rect)
    anchor_h = max(ay1 - ay0, 8.0)
    anchor_cy = 0.5 * (ay0 + ay1)
    sx0 = max(0, int(x0 - 0.55 * anchor_h))
    sx1 = min(w_img, int(x1 + 0.55 * anchor_h))
    sy0 = max(0, int(anchor_cy - 0.95 * anchor_h))
    sy1 = min(h_img, int(anchor_cy + 1.05 * anchor_h))
    roi = image_bgr[sy0:sy1, sx0:sx1]
    if roi.size == 0:
        return (max(0, x0), max(0, int(anchor_cy - 0.55 * anchor_h)), min(w_img, x1), min(h_img, int(anchor_cy + 0.55 * anchor_h)))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    target_x0 = x0 - sx0
    target_x1 = x1 - sx0
    target_cx = 0.5 * (target_x0 + target_x1)
    valid = []
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if area < 3:
            continue
        if ch < 0.28 * anchor_h:
            continue
        if ch > 1.65 * anchor_h:
            continue
        comp_cx = cx + cw / 2.0
        comp_cy = cy + ch / 2.0
        if comp_cx < target_x0 - 0.55 * anchor_h or comp_cx > target_x1 + 0.55 * anchor_h:
            continue
        expected_cy = anchor_cy - sy0
        if abs(comp_cy - expected_cy) > 0.75 * anchor_h:
            continue
        valid.append((cx, cy, cw, ch, area))
    if valid:
        xs0 = [v[0] for v in valid]
        ys0 = [v[1] for v in valid]
        xs1 = [v[0] + v[2] for v in valid]
        ys1 = [v[1] + v[3] for v in valid]
        bx0 = min(xs0)
        by0 = min(ys0)
        bx1 = max(xs1)
        by1 = max(ys1)
        pad = max(2, int(round(0.14 * anchor_h)))
        return (max(0, sx0 + bx0 - pad), max(0, sy0 + by0 - pad), min(w_img, sx0 + bx1 + pad), min(h_img, sy0 + by1 + pad))
    display_h = max(y1 - y0, int(round(0.95 * anchor_h)))
    ny0 = int(round(anchor_cy - 0.52 * display_h))
    ny1 = int(round(anchor_cy + 0.52 * display_h))
    return (max(0, x0), max(0, ny0), min(w_img, x1), min(h_img, ny1))

def _display_bbox_from_ocr(ocr_candidates, fields, field_name, work_scale, work_shape, work_image=None):
    target = _target_value_string(fields or {}, field_name)
    if target is None:
        return None
    items = _collect_mapped_ocr_entries(ocr_candidates or [], work_scale)
    numeric_pattern = re.compile('\\d+(?:[.,]\\d+)?')
    anchor_rect = _field_label_anchor_rect(ocr_candidates or [], field_name, work_scale)
    anchor_cy = None
    anchor_h = None
    if anchor_rect is not None:
        _, ay0, _, ay1 = anchor_rect
        anchor_cy = 0.5 * (ay0 + ay1)
        anchor_h = max(ay1 - ay0, 8.0)
    exact = []
    for item in items:
        raw = item['raw'].strip()
        if re.fullmatch('\\d+(?:[.,]\\d+)?', raw):
            if not _numeric_equal(raw, target, field_name):
                continue
            x0, y0, x1, y1 = item['rect']
            cy = 0.5 * (y0 + y1)
            row_bonus = 0.0
            if anchor_cy is not None:
                dy = abs(cy - anchor_cy)
                tol = 2.2 * max(anchor_h, y1 - y0, 8.0)
                if dy > tol:
                    continue
                row_bonus = max(0.0, 1.0 - dy / max(tol, 1e-06))
            roi_bonus = 0.35 if item.get('roi_kind') in {'land_header', field_name} else 0.0
            exact.append({'bbox': (int(x0), int(y0), int(x1), int(y1)), 'cx': 0.5 * (x0 + x1), 'score': float(item['ocr_score']) + row_bonus + roi_bonus})
    if exact:
        if field_name == 'so_thua':
            chosen = min(exact, key=lambda z: z['cx'])
        elif field_name == 'to_ban_do':
            chosen = max(exact, key=lambda z: z['cx'])
        else:
            chosen = max(exact, key=lambda z: z['score'])
        x0, y0, x1, y1 = chosen['bbox']
        pad = max(2, int(round(0.18 * max(y1 - y0, 8))))
        h, w = work_shape[:2]
        display_bbox = (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))
        if work_image is not None:
            display_bbox = _snap_display_bbox_to_field_row(work_image, display_bbox, anchor_rect)
        return display_bbox
    fallback = []
    for item in items:
        raw = item['raw']
        rect = item['rect']
        for m in numeric_pattern.finditer(raw):
            token = m.group(0)
            if not _numeric_equal(token, target, field_name):
                continue
            bbox = _token_bbox_inside_entry(rect, raw, m.start(), m.end(), work_shape)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            cy = 0.5 * (y0 + y1)
            row_bonus = 0.0
            if anchor_cy is not None:
                dy = abs(cy - anchor_cy)
                tol = 2.2 * max(anchor_h, y1 - y0, 8.0)
                if dy > tol:
                    continue
                row_bonus = max(0.0, 1.0 - dy / max(tol, 1e-06))
            fallback.append({'bbox': bbox, 'cx': 0.5 * (x0 + x1), 'score': float(item['ocr_score']) + row_bonus})
    if not fallback:
        return None
    if field_name == 'so_thua':
        chosen = min(fallback, key=lambda z: z['cx'])
    elif field_name == 'to_ban_do':
        chosen = max(fallback, key=lambda z: z['cx'])
    else:
        chosen = max(fallback, key=lambda z: z['score'])
    display_bbox = tuple(map(int, chosen['bbox']))
    if work_image is not None:
        display_bbox = _snap_display_bbox_to_field_row(work_image, display_bbox, anchor_rect)
    return display_bbox

def _draw_text_tamper_v5(image_bgr, bbox, field_label, score):
    out = image_bgr.copy()
    x0, y0, x1, y1 = map(int, bbox)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 3)
    text = f'{field_label}: {score:.2f}' if field_label else f'text tamper: {score:.2f}'
    cv2.putText(out, text, (x0, max(20, y0 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    return out

def _shift_header_display_bbox_down(bbox, field_name, image_shape):
    if bbox is None:
        return None
    if field_name not in {'so_thua', 'to_ban_do'}:
        return bbox
    h_img, w_img = image_shape[:2]
    x0, y0, x1, y1 = map(int, bbox)
    box_h = max(y1 - y0, 8)
    dy = int(round(1.05 * box_h))
    ny0 = y0 + dy
    ny1 = y1 + dy
    if ny1 > h_img:
        ny1 = h_img
        ny0 = max(0, ny1 - box_h)
    return (max(0, x0), max(0, ny0), min(w_img, x1), min(h_img, ny1))

def analyze_text_tamper(image, ocr_candidates=None, fields=None):
    work, scale = _resize_for_tamper(image, target_long=1400)
    locations = locate_core_field_boxes(ocr_candidates or [], fields or {}, float(scale), work.shape, work_image=work)
    per_field = {'so_thua': None, 'to_ban_do': None, 'dien_tich_m2': None}
    best_field = None
    best_score = 0.0
    best_bbox = None
    for source_field in ['so_thua', 'to_ban_do', 'dien_tich_m2']:
        info = locations.get(source_field)
        if info is None:
            continue
        bbox = info['bbox']
        color_score, color_bbox, _ = _field_synthetic_cue(work, bbox)
        gray_score, gray_bbox, _ = _grayscale_field_anomaly(work, bbox)
        field_score = float(max(color_score, gray_score))
        cue_bbox = gray_bbox if gray_score >= color_score else color_bbox
        assigned_field = source_field
        old_score = per_field.get(assigned_field)
        if old_score is None or field_score > old_score:
            per_field[assigned_field] = field_score
        if field_score > best_score:
            best_score = field_score
            best_field = assigned_field
            best_bbox = cue_bbox
    score = float(best_score)
    global_candidate = None
    if score < 0.4:
        global_candidate = _global_synthetic_blue_candidate(work, max_y_ratio=0.58)
        if global_candidate is not None and global_candidate['score'] >= 0.62:
            gb = global_candidate['bbox']
            if locations:
                nearest_field, nearest_dist = _nearest_field_for_bbox(gb, locations, work.shape)
                if nearest_field is not None and nearest_dist is not None and (nearest_dist <= 0.08):
                    score = float(global_candidate['score'])
                    best_field = nearest_field
                    best_bbox = gb
                    previous = per_field.get(nearest_field)
                    if previous is None or score > previous:
                        per_field[nearest_field] = score
            elif global_candidate['score'] >= 0.72:
                score = float(global_candidate['score'])
                best_bbox = gb
    score = float(np.clip(score, 0.0, 1.0))
    core_keys = ['so_thua', 'to_ban_do', 'dien_tich_m2']
    core_found = sum(((fields or {}).get(k) not in (None, '', []) for k in core_keys))
    located_count = len(locations)
    enough_information = core_found >= 2 and located_count >= 2
    if score >= 0.62:
        status = 'TAMPERED'
    elif score >= 0.4:
        status = 'UNCERTAIN'
    elif not enough_information:
        status = 'UNCERTAIN'
    else:
        status = 'REAL'
    evidence = []
    if status == 'UNCERTAIN' and (not enough_information) and (score < 0.4):
        evidence.append(f'Chưa đủ cơ sở đánh giá Text Tamper: OCR đọc được {core_found}/3 trường lõi, detector định vị được {located_count}/3 trường.')
    elif status != 'REAL':
        if best_field in CORE_TEXT_FIELDS:
            label = CORE_TEXT_FIELDS[best_field]['label']
            evidence.append(f'Phát hiện dấu hiệu bất thường trong trường {label} (score {score:.2f}).')
        else:
            evidence.append(f'Phát hiện dấu hiệu bất thường trong vùng text (score {score:.2f}).')
        evidence.append('TamperScore là chỉ số bất thường phục vụ sàng lọc, không phải xác suất gian lận.')
    overlay_bytes = None
    if status != 'REAL':
        display_bbox = None
        if best_field in CORE_TEXT_FIELDS:
            display_bbox = _display_bbox_from_ocr(ocr_candidates or [], fields or {}, best_field, float(scale), work.shape, work_image=work)
        if display_bbox is None and best_field in locations:
            display_bbox = locations[best_field].get('bbox')
        if display_bbox is None:
            display_bbox = best_bbox
        if display_bbox is not None:
            display_bbox = _shift_header_display_bbox_down(display_bbox, best_field, work.shape)
            label = OVERLAY_FIELD_LABELS.get(best_field, 'Text tamper') if best_field else 'Text tamper'
            overlay = _draw_text_tamper_v5(work, display_bbox, label, score)
            overlay_bytes = _encode_png(overlay)
    return {'version': 'TEXT_TAMPER_V5_BOX_SHIFT_DOWN_FIXED', 'status': status, 'score': score, 'assessment_ready': enough_information or score >= 0.4, 'core_found': core_found, 'located_count': located_count, 'field_scores': per_field, 'best_field': best_field, 'evidence': evidence, 'overlay_bytes': overlay_bytes, 'heatmap_bytes': None, 'local_score': 0.0, 'rect_score': 0.0, 'replacement_score': score, 'seam_score': 0.0, 'ocr_inconsistency': float(_ocr_field_inconsistency(ocr_candidates or []))}

def _red_mask_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 60, 45], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array([165, 60, 45], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

def _find_red_seal_crop(image):
    mask = _red_mask_hsv(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (None, None, None)
    h, w = image.shape[:2]
    image_area = float(h * w)
    candidates = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 0.00015 * image_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        candidates.append((area, x, y, bw, bh))
    if not candidates:
        return (None, None, None)
    _, x, y, bw, bh = max(candidates, key=lambda z: z[0])
    pad_x = max(8, int(0.25 * bw))
    pad_y = max(8, int(0.25 * bh))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)
    crop = image[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return (None, None, None)
    return (crop, mask_crop, (x0, y0, x1, y1))

def _seal_lbp_hist(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    lbp = local_binary_pattern(gray, SEAL_LBP_POINTS, SEAL_LBP_RADIUS, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=SEAL_LBP_BINS, range=(0, SEAL_LBP_BINS))
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist

def _read_image_path(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

@st.cache_resource
def load_seal_reference_profile():
    if not SEAL_REAL_DIR.exists():
        return None
    paths = sorted([p for p in SEAL_REAL_DIR.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}])
    if not paths:
        return None
    hists = []
    red_ratios = []
    for path in paths[:80]:
        img = _read_image_path(path)
        if img is None:
            continue
        crop, mask, _ = _find_red_seal_crop(img)
        if crop is None:
            continue
        hists.append(_seal_lbp_hist(crop))
        red_ratios.append(float(np.mean(mask > 0)))
    if len(hists) < 3:
        return None
    stack = np.vstack(hists).astype(np.float32)
    ref_hist = np.median(stack, axis=0).astype(np.float32)
    ref_hist /= max(float(ref_hist.sum()), 1e-09)
    distances = np.array([cv2.compareHist(h.astype(np.float32), ref_hist, cv2.HISTCMP_BHATTACHARYYA) for h in stack], dtype=float)
    d_med = float(np.median(distances))
    d_mad = float(np.median(np.abs(distances - d_med)))
    d_sigma = max(1.4826 * d_mad, 0.0001)
    uncertain_thr = max(0.015, d_med + 2.5 * d_sigma)
    suspicious_thr = max(0.03, d_med + 4.0 * d_sigma, uncertain_thr + 0.01)
    red_ratios = np.array(red_ratios, dtype=float)
    r_med = float(np.median(red_ratios))
    r_mad = float(np.median(np.abs(red_ratios - r_med)))
    r_sigma = max(1.4826 * r_mad, 0.005)
    return {'count': len(hists), 'hist': ref_hist, 'distance_median': d_med, 'distance_uncertain': float(uncertain_thr), 'distance_suspicious': float(suspicious_thr), 'red_ratio_median': r_med, 'red_ratio_sigma': r_sigma}

def analyze_seal_tamper(image):
    profile = load_seal_reference_profile()
    crop, mask, bbox = _find_red_seal_crop(image)
    if crop is None:
        return {'status': 'NO_SEAL', 'score': None, 'lbp_distance': None, 'red_ratio': None, 'reference_count': 0 if profile is None else profile['count'], 'uncertain_threshold': None if profile is None else profile['distance_uncertain'], 'suspicious_threshold': None if profile is None else profile['distance_suspicious'], 'evidence': ['Không tìm thấy vùng dấu đỏ đủ rõ để đánh giá.'], 'seal_crop_bytes': None, 'seal_mask_bytes': None, 'overlay_bytes': None}
    overlay = image.copy()
    x0, y0, x1, y1 = bbox
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 3)
    red_ratio = float(np.mean(mask > 0))
    if profile is None:
        return {'status': 'NO_REFERENCE', 'score': None, 'lbp_distance': None, 'red_ratio': red_ratio, 'reference_count': 0, 'uncertain_threshold': None, 'suspicious_threshold': None, 'evidence': ['Đã tìm thấy vùng dấu đỏ nhưng chưa có đủ mẫu Real trong dataset/processed_real/seal để tạo profile.'], 'seal_crop_bytes': _encode_png(crop), 'seal_mask_bytes': _encode_png(mask), 'overlay_bytes': _encode_png(overlay)}
    hist = _seal_lbp_hist(crop)
    distance = float(cv2.compareHist(hist.astype(np.float32), profile['hist'].astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))
    lbp_score = float(np.clip(distance / max(profile['distance_suspicious'], 1e-06), 0.0, 1.0))
    red_z = abs(red_ratio - profile['red_ratio_median']) / max(profile['red_ratio_sigma'], 1e-06)
    color_score = float(np.clip((red_z - 1.5) / 3.5, 0.0, 1.0))
    score = float(np.clip(0.72 * lbp_score + 0.28 * color_score, 0.0, 1.0))
    if distance >= profile['distance_suspicious'] or score >= 0.65:
        status = 'SUSPICIOUS'
    elif distance >= profile['distance_uncertain'] or score >= 0.4:
        status = 'UNCERTAIN'
    else:
        status = 'OK'
    evidence = []
    if distance >= profile['distance_suspicious']:
        evidence.append(f'Texture LBP của dấu lệch mạnh profile Real (distance {distance:.3f}).')
    elif distance >= profile['distance_uncertain']:
        evidence.append(f'Texture LBP của dấu lệch mức cần kiểm tra (distance {distance:.3f}).')
    if red_z >= 3.0:
        evidence.append('Tỷ lệ pixel đỏ của vùng dấu lệch đáng kể so với mẫu Real.')
    if not evidence:
        evidence.append('Chưa thấy khác biệt rõ của vùng dấu so với profile Real.')
    return {'status': status, 'score': score, 'lbp_distance': distance, 'red_ratio': red_ratio, 'red_z': float(red_z), 'reference_count': int(profile['count']), 'uncertain_threshold': float(profile['distance_uncertain']), 'suspicious_threshold': float(profile['distance_suspicious']), 'evidence': evidence, 'seal_crop_bytes': _encode_png(crop), 'seal_mask_bytes': _encode_png(mask), 'overlay_bytes': _encode_png(overlay)}

def _empty_fields():
    return {'so_thua': None, 'to_ban_do': None, 'dia_chi': None, 'dien_tich_m2': None, 'hinh_thuc_su_dung': None, 'muc_dich_su_dung': None, 'thoi_han_su_dung': None, 'nguon_goc_su_dung': None, 'so_hop_dong': None, 'ngay_hop_dong': None, 'muc_dich_chi_tiet': []}

def analyze_one_file(file_name, file_bytes, page_type):
    img_bgr = _decode_image_bytes(file_bytes)
    if img_bgr is None:
        return {'file_name': file_name, 'page_type': page_type, 'error': 'Không thể đọc ảnh.', 'original_bytes': file_bytes}
    result = {'file_name': file_name, 'page_type': page_type, 'error': None, 'original_bytes': file_bytes, 'fields': _empty_fields(), 'field_sources': {}, 'per_pass_debug': [], 'smart_stats': {'so_lan_ocr': 0, 'fullpage_passes': 0, 'roi_passes': 0}, 'ocr_pass': None, 'ocr_confidence': None, 'ocr_raw': None, 'ocr_cleaned': None, 'checks': [], 'tamper': None, 'text_tamper': None, 'seal': None}
    if page_type == 'Trang 3 (Phần Sơ đồ)':
        result['tamper'] = analyze_tamper_hybrid(img_bgr)
        result['seal'] = analyze_seal_tamper(img_bgr)
        return result
    candidates, best_candidate, fields, field_sources, per_pass_debug, smart_stats = run_smart_fast_ocr(img_bgr)
    result['fields'] = fields
    result['field_sources'] = field_sources
    result['per_pass_debug'] = per_pass_debug
    result['smart_stats'] = smart_stats
    result['checks'] = validate_extracted_fields(fields)
    if best_candidate is not None:
        result['ocr_pass'] = best_candidate.get('pass')
        result['ocr_confidence'] = float(best_candidate.get('avg_conf', 0.0))
        result['ocr_raw'] = best_candidate.get('raw_text')
        result['ocr_cleaned'] = best_candidate.get('cleaned_text')
    result['text_tamper'] = analyze_text_tamper(img_bgr, candidates, fields)
    return result

def batch_status(result):
    if result.get('error'):
        return 'Lỗi ảnh'
    if result['page_type'] == 'Trang 3 (Phần Sơ đồ)':
        t = result.get('tamper') or {}
        seal = result.get('seal') or {}
        hybrid_status = t.get('status', 'UNCERTAIN')
        seal_status = seal.get('status')
        if hybrid_status == 'TAMPERED' or seal_status == 'SUSPICIOUS':
            return 'Nghi vấn chỉnh sửa'
        if hybrid_status == 'UNCERTAIN' or seal_status == 'UNCERTAIN':
            return 'Cần kiểm tra'
        return 'Chưa phát hiện chỉnh sửa'
    text_tamper = result.get('text_tamper') or {}
    text_status = text_tamper.get('status')
    if text_status == 'TAMPERED':
        return 'Nghi vấn chỉnh sửa chữ/số'
    if text_status == 'UNCERTAIN':
        return 'Cần kiểm tra text'
    f = result.get('fields', {})
    core_keys = ['so_thua', 'to_ban_do', 'dien_tich_m2']
    found = sum((f.get(k) not in (None, '', []) for k in core_keys))
    if found == 3:
        return 'Đủ 3 trường lõi'
    if found >= 1:
        return f'Thiếu {3 - found}/3 trường lõi'
    return 'OCR yếu / thiếu trường lõi'

def result_to_export_row(result):
    fields = result.get('fields', _empty_fields())
    tamper = result.get('tamper') or {}
    text_tamper = result.get('text_tamper') or {}
    seal = result.get('seal') or {}
    return {'Ten_File': result.get('file_name'), 'Loai_Trang': result.get('page_type'), 'Trang_Thai': batch_status(result), 'So_Thua': fields.get('so_thua'), 'To_Ban_Do': fields.get('to_ban_do'), 'Dia_Chi': fields.get('dia_chi'), 'Dien_Tich_m2': fields.get('dien_tich_m2'), 'Hinh_Thuc_Su_Dung': fields.get('hinh_thuc_su_dung'), 'Muc_Dich_Su_Dung': fields.get('muc_dich_su_dung'), 'Thoi_Han_Su_Dung': fields.get('thoi_han_su_dung'), 'Nguon_Goc_Su_Dung': fields.get('nguon_goc_su_dung'), 'So_Hop_Dong': fields.get('so_hop_dong'), 'Ngay_Hop_Dong': fields.get('ngay_hop_dong'), 'OCR_Pass': result.get('ocr_pass'), 'OCR_Avg_Confidence': result.get('ocr_confidence'), 'TextTamper_Status': text_tamper.get('status'), 'TextTamper_Score': text_tamper.get('score'), 'TextTamper_AssessmentReady': text_tamper.get('assessment_ready'), 'TextTamper_CoreFound': text_tamper.get('core_found'), 'TextTamper_LocatedCount': text_tamper.get('located_count'), 'TextTamper_Local': text_tamper.get('local_score'), 'TextTamper_Paste': text_tamper.get('rect_score'), 'TextTamper_Replacement': text_tamper.get('replacement_score'), 'TextTamper_Seam': text_tamper.get('seam_score'), 'TextTamper_OCR_Inconsistency': text_tamper.get('ocr_inconsistency'), 'TextTamper_Evidence': ' | '.join(text_tamper.get('evidence', [])), 'Tamper_Version': tamper.get('version'), 'Tamper_Status': tamper.get('status'), 'Tamper_Score': tamper.get('score'), 'Local_Score': tamper.get('local_score'), 'Rect_Paste_Score': tamper.get('rect_score'), 'CopyMove_Score': tamper.get('copy_move_score'), 'Seam_Score': tamper.get('seam_score'), 'Blur_PII_Fraction': tamper.get('blur_fraction'), 'CopyMove_Matches': tamper.get('copy_move_matches'), 'CopyMove_Cluster': tamper.get('copy_move_cluster'), 'CopyMove_Detector': tamper.get('copy_move_detector'), 'Tamper_Evidence': ' | '.join(tamper.get('evidence', [])), 'Seal_Status': seal.get('status'), 'Seal_Score': seal.get('score'), 'Seal_LBP_Distance': seal.get('lbp_distance'), 'Seal_Red_Ratio': seal.get('red_ratio'), 'Seal_Reference_Count': seal.get('reference_count'), 'Seal_Uncertain_Threshold': seal.get('uncertain_threshold'), 'Seal_Suspicious_Threshold': seal.get('suspicious_threshold'), 'Seal_Evidence': ' | '.join(seal.get('evidence', []))}

def batch_excel_bytes(results):
    rows = [result_to_export_row(r) for r in results]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name='Tong_hop')
        debug_rows = []
        for r in results:
            if r.get('ocr_raw') is None:
                continue
            debug_rows.append({'Ten_File': r['file_name'], 'OCR_Pass': r.get('ocr_pass'), 'OCR_Avg_Confidence': r.get('ocr_confidence'), 'OCR_Raw': r.get('ocr_raw'), 'OCR_Cleaned': r.get('ocr_cleaned')})
        if debug_rows:
            pd.DataFrame(debug_rows).to_excel(writer, index=False, sheet_name='OCR_Debug')
    return buf.getvalue()
st.title('📜 Phân tích Sổ đỏ')
st.caption('OCR Trang 2 + kiểm tra text tamper; Trang 3 kiểm tra diagram/copy-move và vùng con dấu.')
uploaded_files = st.file_uploader('Chọn một hoặc nhiều ảnh', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key='v9_integrated_batch_files')
batch_mode = st.radio('Loại ảnh', ['Tất cả là Trang 2 (Phần văn bản)', 'Tất cả là Trang 3 (Phần sơ đồ)', 'Chọn riêng từng ảnh'], horizontal=True)
page_type_map = {}
if uploaded_files:
    if batch_mode == 'Tất cả là Trang 2 (Phần văn bản)':
        for f in uploaded_files:
            page_type_map[f.name] = 'Trang 2 (Phần Văn bản)'
    elif batch_mode == 'Tất cả là Trang 3 (Phần sơ đồ)':
        for f in uploaded_files:
            page_type_map[f.name] = 'Trang 3 (Phần Sơ đồ)'
    else:
        st.subheader('Chọn loại trang cho từng ảnh')
        for idx, f in enumerate(uploaded_files):
            page_type_map[f.name] = st.selectbox(f'{idx + 1}. {f.name}', ['Trang 2 (Phần Văn bản)', 'Trang 3 (Phần Sơ đồ)'], key=f'page_{idx}_{f.name}')
    c1, c2, _ = st.columns([1.2, 1.0, 3.8])
    with c1:
        run_batch = st.button(f'Phân tích {len(uploaded_files)} ảnh', type='primary', use_container_width=True)
    with c2:
        clear_batch = st.button('Xóa kết quả', use_container_width=True)
    if clear_batch:
        st.session_state.pop('v9_integrated_batch_results', None)
        st.rerun()
    if run_batch:
        results = []
        total = len(uploaded_files)
        progress = st.progress(0.0, text='Đang chuẩn bị...')
        status_box = st.empty()
        for i, f in enumerate(uploaded_files, start=1):
            page_type = page_type_map[f.name]
            status_box.info(f'Đang xử lý {i}/{total}: {f.name}')
            try:
                result = analyze_one_file(f.name, f.getvalue(), page_type)
            except Exception as e:
                result = {'file_name': f.name, 'page_type': page_type, 'error': str(e), 'original_bytes': f.getvalue()}
            results.append(result)
            progress.progress(i / total, text=f'Đã xử lý {i}/{total} ảnh')
        st.session_state['v9_integrated_batch_results'] = results
        progress.empty()
        status_box.success('Phân tích hoàn tất.')
results = st.session_state.get('v9_integrated_batch_results', [])
if results:
    st.divider()
    st.header('Kết quả phân tích')
    summary = []
    for r in results:
        if r.get('error'):
            summary.append({'Tên file': r['file_name'], 'Kết quả': 'Lỗi xử lý'})
            continue
        if r['page_type'] == 'Trang 3 (Phần Sơ đồ)':
            t = r['tamper']
            seal = r.get('seal') or {}
            summary.append({'Tên file': r['file_name'], 'Kết quả': batch_status(r), 'Điểm hybrid': round(t['score'], 3), 'Chèn/dán': round(t['rect_score'], 3), 'Sao chép vùng': round(t['copy_move_score'], 3), 'Seal': {'OK': 'Không nghi vấn', 'UNCERTAIN': 'Cần kiểm tra', 'SUSPICIOUS': 'Nghi vấn', 'NO_SEAL': 'Không tìm thấy dấu', 'NO_REFERENCE': 'Thiếu mẫu tham chiếu'}.get(seal.get('status'), seal.get('status')), 'LBP seal': round(seal['lbp_distance'], 3) if seal.get('lbp_distance') is not None else None})
        else:
            f = r['fields']
            tt = r.get('text_tamper') or {}
            summary.append({'Tên file': r['file_name'], 'Kết quả': batch_status(r), 'Text tamper': {'REAL': 'Chưa phát hiện', 'UNCERTAIN': 'Cần kiểm tra', 'TAMPERED': 'Nghi vấn'}.get(tt.get('status'), tt.get('status')), 'Text score': round(tt['score'], 3) if tt.get('assessment_ready', True) and tt.get('score') is not None else None, 'Số thửa': f.get('so_thua'), 'Tờ bản đồ': f.get('to_ban_do'), 'Diện tích (m²)': f.get('dien_tich_m2'), 'Mục đích sử dụng': f.get('muc_dich_su_dung'), 'Thời hạn sử dụng': f.get('thoi_han_su_dung')})
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    diagram_results = [r for r in results if not r.get('error') and r['page_type'] == 'Trang 3 (Phần Sơ đồ)']
    if diagram_results:
        n_real = sum((batch_status(r) == 'Chưa phát hiện chỉnh sửa' for r in diagram_results))
        n_uncertain = sum((batch_status(r) == 'Cần kiểm tra' for r in diagram_results))
        n_tampered = sum((batch_status(r) == 'Nghi vấn chỉnh sửa' for r in diagram_results))
        a, b, c = st.columns(3)
        a.metric('Chưa phát hiện chỉnh sửa', n_real)
        b.metric('Cần kiểm tra', n_uncertain)
        c.metric('Nghi vấn chỉnh sửa', n_tampered)
        with st.expander('Giải thích các chỉ số'):
            st.markdown('\n- **Điểm tổng:** điểm kết hợp các dấu hiệu chỉnh sửa. Điểm càng cao thì ảnh càng đáng nghi.\n- **Bất thường cục bộ:** mức khác biệt về texture, biên và tần số ở từng vùng nhỏ của ảnh.\n- **Chèn/dán:** mức nghi vấn có vùng hình chữ nhật hoặc vùng nền khác biệt giống thao tác chèn nội dung.\n- **Sao chép vùng:** mức nghi vấn một phần hoa văn được sao chép sang vị trí khác trong cùng ảnh.\n- **Biên bất thường:** mức xuất hiện các đường nối hoặc mép cắt dán không tự nhiên.\n- **Vùng làm mờ PII:** tỷ lệ vùng được nhận dạng là làm mờ thông tin cá nhân. Vùng này được loại bớt khỏi điểm giả mạo.\n\n**Ngưỡng Hybrid Trang 3:** dưới `0.40` là chưa phát hiện dấu hiệu rõ, từ `0.40` đến dưới `0.62` là cần kiểm tra, từ `0.62` trở lên là nghi vấn chỉnh sửa. Seal LBP được đánh giá thêm bằng profile các mẫu Real.\n\nCác điểm trên là chỉ số của thuật toán, không phải xác suất chính xác tuyệt đối.\n                ')
    text_results = [r for r in results if not r.get('error') and r['page_type'] == 'Trang 2 (Phần Văn bản)']
    if text_results:
        n_text_real = sum(((r.get('text_tamper') or {}).get('status') == 'REAL' for r in text_results))
        n_text_uncertain = sum(((r.get('text_tamper') or {}).get('status') == 'UNCERTAIN' for r in text_results))
        n_text_tampered = sum(((r.get('text_tamper') or {}).get('status') == 'TAMPERED' for r in text_results))
        st.markdown('**Tổng hợp Text Tamper - Trang 2**')
        a, b, c = st.columns(3)
        a.metric('Chưa phát hiện', n_text_real)
        b.metric('Cần kiểm tra', n_text_uncertain)
        c.metric('Nghi vấn chữ/số', n_text_tampered)
    st.subheader('Chi tiết từng ảnh')
    for idx, r in enumerate(results, start=1):
        with st.expander(f"{idx}. {r['file_name']} — {batch_status(r)}", expanded=idx == 1):
            if r.get('error'):
                st.error('Không xử lý được ảnh: ' + r['error'])
                continue
            left, right = st.columns([1, 1.35])
            with left:
                st.image(r['original_bytes'], caption=r['file_name'], use_container_width=True)
            with right:
                if r['page_type'] == 'Trang 3 (Phần Sơ đồ)':
                    t = r['tamper']
                    overall_status = batch_status(r)
                    if overall_status == 'Nghi vấn chỉnh sửa':
                        st.error('🚨 Phát hiện dấu hiệu cần kiểm tra')
                    elif overall_status == 'Cần kiểm tra':
                        st.warning('⚠️ Kết quả chưa rõ, nên kiểm tra thủ công')
                    else:
                        st.success('✅ Chưa phát hiện dấu hiệu chỉnh sửa rõ ràng')
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric('Điểm Hybrid', f"{t['score']:.2f}")
                    m2.metric('Chèn/dán', f"{t['rect_score']:.2f}")
                    m3.metric('Sao chép vùng', f"{t['copy_move_score']:.2f}")
                    m4.metric('Bất thường cục bộ', f"{t['local_score']:.2f}")
                    st.caption(f"Vùng làm mờ PII: {t['blur_fraction']:.1%}")
                    seal = r.get('seal') or {}
                    st.markdown('**Kiểm tra vùng con dấu**')
                    s1, s2, s3 = st.columns(3)
                    s1.metric('Seal status', seal.get('status') or 'Không có dữ liệu')
                    s2.metric('LBP distance', f"{seal['lbp_distance']:.3f}" if seal.get('lbp_distance') is not None else 'N/A')
                    s3.metric('Mẫu Real tham chiếu', seal.get('reference_count', 0))
                    if seal.get('evidence'):
                        for e in seal['evidence']:
                            st.write('•', e)
                    if seal.get('seal_crop_bytes'):
                        with st.expander('Vùng dấu được phân tích'):
                            st.image(seal['seal_crop_bytes'], use_container_width=True)
                    if t['evidence']:
                        st.markdown('**Dấu hiệu tìm thấy:**')
                        for e in t['evidence']:
                            st.write('•', e)
                    if t.get('overlay_bytes'):
                        st.markdown('**Vùng nghi vấn trên ảnh**')
                        st.image(t['overlay_bytes'], use_container_width=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        with st.expander('Bản đồ vùng bất thường'):
                            if t.get('heatmap_bytes'):
                                st.image(t['heatmap_bytes'], use_container_width=True)
                    with c2:
                        with st.expander('Vùng làm mờ thông tin cá nhân'):
                            if t.get('blur_mask_bytes'):
                                st.image(t['blur_mask_bytes'], use_container_width=True)
                else:
                    fields = r['fields']
                    tt = r.get('text_tamper') or {}
                    if tt.get('status') == 'TAMPERED':
                        st.error('🚨 Nghi vấn chỉnh sửa chữ/số')
                    elif tt.get('status') == 'UNCERTAIN':
                        st.warning('⚠️ Chưa đủ bằng chứng, cần kiểm tra thêm')
                    elif tt:
                        st.success('✅ Chưa phát hiện dấu hiệu text tamper rõ ràng')
                    if tt:
                        field_scores = tt.get('field_scores', {})
                        m1, m2, m3, m4 = st.columns(4)
                        score_display = f"{tt.get('score', 0.0):.2f}" if tt.get('assessment_ready', True) else 'N/A'
                        m1.metric('Text Score', score_display)
                        parcel_score = field_scores.get('so_thua')
                        map_score = field_scores.get('to_ban_do')
                        area_score = field_scores.get('dien_tich_m2')
                        m2.metric('Số thửa', f'{parcel_score:.2f}' if parcel_score is not None else 'N/A')
                        m3.metric('Tờ bản đồ', f'{map_score:.2f}' if map_score is not None else 'N/A')
                        m4.metric('Diện tích', f'{area_score:.2f}' if area_score is not None else 'N/A')
                        if tt.get('evidence'):
                            with st.expander('Dấu hiệu Text Tamper'):
                                for e in tt['evidence']:
                                    st.write('•', e)
                        if tt.get('status') != 'REAL' and tt.get('overlay_bytes'):
                            with st.expander('Vùng text nghi vấn', expanded=True):
                                st.image(tt['overlay_bytes'], use_container_width=True)
                    a, b, c = st.columns(3)
                    a.metric('Số thửa', fields.get('so_thua') or 'Chưa đọc được')
                    b.metric('Tờ bản đồ', fields.get('to_ban_do') or 'Chưa đọc được')
                    area = fields.get('dien_tich_m2')
                    c.metric('Diện tích', f'{area:g} m²' if area is not None else 'Chưa đọc được')
                    rows = [('Số thửa', fields.get('so_thua')), ('Tờ bản đồ', fields.get('to_ban_do')), ('Địa chỉ', fields.get('dia_chi')), ('Diện tích', f'{area:g} m²' if area is not None else None), ('Hình thức sử dụng', fields.get('hinh_thuc_su_dung')), ('Mục đích sử dụng', fields.get('muc_dich_su_dung')), ('Thời hạn sử dụng', fields.get('thoi_han_su_dung')), ('Nguồn gốc sử dụng', fields.get('nguon_goc_su_dung'))]
                    df = pd.DataFrame(rows, columns=['Trường', 'Kết quả'])
                    df['Kết quả'] = df['Kết quả'].fillna('Chưa trích xuất được')
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    if r.get('ocr_confidence') is not None:
                        stats = r.get('smart_stats', {})
                        st.caption(f"Độ tin cậy OCR: {r['ocr_confidence']:.3f} | Số lượt OCR: {stats.get('so_lan_ocr', 0)}")
                        with st.expander('Giải thích chỉ số OCR'):
                            st.markdown('\n- **Độ tin cậy OCR:** mức tin cậy trung bình do mô hình OCR trả về, giá trị gần `1.0` thường tốt hơn.\n- **Số lượt OCR:** số lần hệ thống phải đọc ảnh hoặc từng vùng nhỏ để tìm đủ thông tin.\n- Kết quả OCR vẫn cần đối chiếu ảnh gốc nếu trường dữ liệu quan trọng bị thiếu hoặc sai dấu.\n                                ')
                    with st.expander('Văn bản OCR gốc'):
                        st.text(r.get('ocr_raw') or '')
    st.divider()
    st.subheader('Xuất kết quả')
    export_rows = [result_to_export_row(r) for r in results]
    json_str = json.dumps(export_rows, ensure_ascii=False, indent=4)
    excel_data = batch_excel_bytes(results)
    d1, d2, _ = st.columns([1.2, 1.2, 3.6])
    with d1:
        st.download_button('Tải JSON', data=json_str, file_name='ket_qua_phan_tich.json', mime='application/json', use_container_width=True)
    with d2:
        st.download_button('Tải Excel', data=excel_data, file_name='ket_qua_phan_tich.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
