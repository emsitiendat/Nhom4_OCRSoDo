import csv
import math
import os
import re
import shutil
import unicodedata
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from skimage.morphology import skeletonize
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
    from sklearn.model_selection import GroupKFold
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = 'False'
from paddleocr import PaddleOCR
REAL_DIR = Path('dataset/processed_real/full_clahe')
TEXT_TAMPER_DIR = Path('dataset/processed_tampered/text_tamper/full_clahe')
OUT_DIR = Path('dataset/tuan5_results')
BENCHMARK_DIR = Path('dataset/tuan5_auto_benchmark')
OCR_BENCH_DIR = BENCHMARK_DIR / 'ocr'
GEOMETRY_BENCH_DIR = BENCHMARK_DIR / 'geometry'
CLEAR_OLD_OUTPUT = True
RANDOM_SEED = 42
N_FOLDS = 5
OCR_TEST_COUNT = 30
GEOMETRY_TEST_COUNT = 40
OCR_MIN_SCORE = 0.3
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_BINS = LBP_POINTS + 2
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

def tao_output():
    if CLEAR_OLD_OUTPUT:
        for folder in [OUT_DIR, BENCHMARK_DIR]:
            if folder.exists():
                shutil.rmtree(folder)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OCR_BENCH_DIR.mkdir(parents=True, exist_ok=True)
    GEOMETRY_BENCH_DIR.mkdir(parents=True, exist_ok=True)

def doc_anh(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def luu_anh(path, img):
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

def ghi_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def bo_dau(text):
    text = str(text).lower().replace('đ', 'd')
    text = unicodedata.normalize('NFD', text)
    text = ''.join((c for c in text if unicodedata.category(c) != 'Mn'))
    return re.sub('\\s+', ' ', text).strip()

def fft_high_ratio(gray):
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fftshift(np.fft.fft2(small))
    power = np.abs(spectrum) ** 2
    h, w = power.shape
    cy, cx = (h // 2, w // 2)
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low = float(power[radius <= 25].sum())
    high = float(power[radius >= 60].sum())
    return high / (low + high + 1e-09)

def lbp_hist_feature(gray):
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    lbp = local_binary_pattern(small, LBP_POINTS, LBP_RADIUS, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS, range=(0, LBP_BINS))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist

def patch_std_features(gray):
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    values = []
    for y in range(0, 256, 64):
        for x in range(0, 256, 64):
            patch = small[y:y + 64, x:x + 64]
            values.append(float(np.std(patch)))
    values = np.array(values, dtype=float)
    return [float(np.mean(values)), float(np.std(values)), float(np.max(values)), float(np.percentile(values, 75))]

def trich_feature_tamper(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    mean = float(np.mean(small))
    std = float(np.std(small))
    lap_var = float(cv2.Laplacian(small, cv2.CV_64F).var())
    edges = cv2.Canny(small, 60, 160)
    edge_ratio = float(np.mean(edges > 0))
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink_ratio = float(np.mean(binary > 0))
    features = [mean, std, lap_var, edge_ratio, ink_ratio, fft_high_ratio(small), *patch_std_features(small), *lbp_hist_feature(small).tolist()]
    return np.array(features, dtype=np.float32)

def tao_tamper_dataset():
    records = []
    for path in lay_ds_anh(REAL_DIR):
        index = lay_source_index(path)
        if index is not None:
            records.append({'path': path, 'source_index': index, 'label': 0, 'type': 'real'})
    for path in lay_ds_anh(TEXT_TAMPER_DIR):
        index = lay_source_index(path)
        if index is not None:
            records.append({'path': path, 'source_index': index, 'label': 1, 'type': 'text_tamper'})
    X = []
    y = []
    groups = []
    valid_records = []
    for record in records:
        img = doc_anh(record['path'])
        if img is None:
            continue
        X.append(trich_feature_tamper(img))
        y.append(record['label'])
        groups.append(record['source_index'])
        valid_records.append(record)
    if not X:
        return None
    return (np.vstack(X), np.array(y, dtype=int), np.array(groups, dtype=int), valid_records)

def tao_group_folds(X, y, groups):
    n_splits = min(N_FOLDS, len(np.unique(groups)))
    if n_splits < 2:
        return []
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    else:
        splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(X, y, groups))

def ve_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Real', 'Tampered'])
    ax.set_yticklabels(['Real', 'Tampered'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix - Text Tampering')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'confusion_matrix.png', dpi=180)
    plt.close(fig)

def ve_roc_curve(y_true, y_score, auc):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Text Tampering')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'roc_curve.png', dpi=180)
    plt.close(fig)

def danh_gia_tamper_auto():
    data = tao_tamper_dataset()
    if data is None:
        print('[TAMPER] khong co du lieu')
        return None
    X, y, groups, records = data
    if len(np.unique(y)) < 2:
        print('[TAMPER] can ca Real va Text Tamper')
        return None
    folds = tao_group_folds(X, y, groups)
    if not folds:
        print('[TAMPER] khong tao duoc group split')
        return None
    scores = np.full(len(y), np.nan, dtype=float)
    preds = np.full(len(y), -1, dtype=int)
    fold_id = np.full(len(y), -1, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=RANDOM_SEED))
        model.fit(X[train_idx], y[train_idx])
        scores[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        preds[test_idx] = (scores[test_idx] >= 0.5).astype(int)
        fold_id[test_idx] = fold
    valid = np.isfinite(scores) & (preds >= 0)
    if valid.sum() == 0:
        print('[TAMPER] khong co OOF prediction hop le')
        return None
    y_true = y[valid]
    y_score = scores[valid]
    y_pred = preds[valid]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    metrics = {'n_samples': int(valid.sum()), 'n_real': int(np.sum(y_true == 0)), 'n_tampered': int(np.sum(y_true == 1)), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp), 'accuracy': float(accuracy), 'precision': float(precision), 'recall': float(recall), 'specificity': float(specificity), 'f1': float(f1), 'balanced_accuracy': float(balanced_acc), 'auc_roc': float(auc), 'split_note': 'Out-of-fold grouped by source_index'}
    details = []
    for i in np.where(valid)[0]:
        record = records[i]
        details.append({'sample_id': record['path'].name, 'source_index': record['source_index'], 'type': record['type'], 'y_true': int(y[i]), 'y_score': float(scores[i]), 'y_pred': int(preds[i]), 'fold': int(fold_id[i])})
    ghi_csv(OUT_DIR / 'tamper_predictions.csv', details)
    ghi_csv(OUT_DIR / 'tamper_metrics.csv', [metrics])
    ve_confusion_matrix(cm)
    ve_roc_curve(y_true, y_score, auc)
    print('\n--- TAMPERING DETECTION ---')
    print('OOF samples       :', int(valid.sum()))
    print('Real / Tampered   :', metrics['n_real'], '/', metrics['n_tampered'])
    print('TN / FP           :', tn, '/', fp)
    print('FN / TP           :', fn, '/', tp)
    print(f'Accuracy          : {accuracy:.4f}')
    print(f'Precision         : {precision:.4f}')
    print(f'Recall            : {recall:.4f}')
    print(f'Specificity       : {specificity:.4f}')
    print(f'F1-score          : {f1:.4f}')
    print(f'Balanced Accuracy : {balanced_acc:.4f}')
    print(f'AUC-ROC           : {auc:.4f}')
    return metrics

def tim_font():
    candidates = ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/tahoma.ttf', 'C:/Windows/Fonts/calibri.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']
    for path in candidates:
        if Path(path).exists():
            return path
    return None

def tao_ocr_sample(rng):
    so_thua = str(rng.integers(1, 1000))
    to_ban_do = str(rng.integers(1, 100))
    dien_tich = f'{rng.integers(20, 1000)},{rng.integers(0, 10)}'
    lines = [f'a) Thửa đất số: {so_thua}', f'Tờ bản đồ số: {to_ban_do}', f'c) Diện tích: {dien_tich} m²']
    width = 900
    height = 230
    pil_img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(pil_img)
    font_path = tim_font()
    font = ImageFont.truetype(font_path, 32) if font_path else ImageFont.load_default()
    y = 25
    for line in lines:
        draw.text((30, y), line, font=font, fill='black')
        y += 60
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    angle = rng.uniform(-1.2, 1.2)
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 0.8))
    noise = rng.normal(0, rng.uniform(2.0, 7.0), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return (img, {'so_thua': so_thua, 'to_ban_do': to_ban_do, 'dien_tich': dien_tich})

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

def chuan_hoa_gia_tri_so(value):
    return str(value).strip().replace(' ', '').replace('.', ',')

def trich_ocr_fields(text):
    norm = bo_dau(text)

    def first(patterns):
        for pattern in patterns:
            match = re.search(pattern, norm)
            if match:
                return chuan_hoa_gia_tri_so(match.group(1))
        return ''
    so_thua = first(['thua\\s*(?:dat|dt)?\\s*(?:so|s)?\\s*[:\\-]?\\s*([0-9]+)', 'so\\s*thua\\s*[:\\-]?\\s*([0-9]+)'])
    to_ban_do = first(['to\\s*(?:ban|bn)?\\s*(?:do|d)?\\s*(?:so|s)?\\s*[:\\-]?\\s*([0-9]+)'])
    dien_tich = first(['(?:dien|din)\\s*(?:tich|tch)?\\s*[:\\-]?\\s*([0-9]+(?:[.,][0-9]+)?)'])
    numbers = re.findall('(?<![0-9])([0-9]+(?:[.,][0-9]+)?)(?![0-9])', norm)
    numbers = [chuan_hoa_gia_tri_so(x) for x in numbers]
    if not so_thua and len(numbers) >= 1:
        so_thua = numbers[0]
    if not to_ban_do and len(numbers) >= 2:
        to_ban_do = numbers[1]
    if not dien_tich and len(numbers) >= 3:
        dien_tich = numbers[2]
    return {'so_thua': so_thua, 'to_ban_do': to_ban_do, 'dien_tich': dien_tich}

def edit_distance(seq1, seq2):
    prev = list(range(len(seq2) + 1))
    for i, a in enumerate(seq1, start=1):
        curr = [i]
        for j, b in enumerate(seq2, start=1):
            cost = 0 if a == b else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]

def cer_for_pairs(refs, hyps):
    total = sum((len(ref) for ref in refs))
    if total == 0:
        return 0.0
    errors = sum((edit_distance(list(ref), list(hyp)) for ref, hyp in zip(refs, hyps)))
    return errors / total

def wer_for_pairs(refs, hyps):
    ref_words = [ref.split() for ref in refs]
    hyp_words = [hyp.split() for hyp in hyps]
    total = sum((len(words) for words in ref_words))
    if total == 0:
        return 0.0
    errors = sum((edit_distance(ref, hyp) for ref, hyp in zip(ref_words, hyp_words)))
    return errors / total

def danh_gia_ocr_auto():
    rng = np.random.default_rng(RANDOM_SEED)
    ocr = khoi_tao_ocr()
    details = []
    refs_by_field = {'so_thua': [], 'to_ban_do': [], 'dien_tich': []}
    hyps_by_field = {'so_thua': [], 'to_ban_do': [], 'dien_tich': []}
    print(f'[OCR] controlled benchmark: {OCR_TEST_COUNT} mau')
    for i in range(1, OCR_TEST_COUNT + 1):
        img, gt = tao_ocr_sample(rng)
        result = ocr.predict(img)
        raw_text = ' '.join(parse_paddle_text(result))
        pred = trich_ocr_fields(raw_text)
        if i <= 5:
            luu_anh(OCR_BENCH_DIR / f'ocr_sample_{i:03d}.png', img)
        details.append({'sample_id': i, 'gt_so_thua': gt['so_thua'], 'pred_so_thua': pred['so_thua'], 'gt_to_ban_do': gt['to_ban_do'], 'pred_to_ban_do': pred['to_ban_do'], 'gt_dien_tich': gt['dien_tich'], 'pred_dien_tich': pred['dien_tich'], 'raw_text': raw_text})
        for key in refs_by_field:
            refs_by_field[key].append(chuan_hoa_gia_tri_so(gt[key]))
            hyps_by_field[key].append(chuan_hoa_gia_tri_so(pred[key]))
    metrics_rows = []
    for key, label in [('so_thua', 'Số thửa'), ('to_ban_do', 'Tờ bản đồ'), ('dien_tich', 'Diện tích')]:
        refs = refs_by_field[key]
        hyps = hyps_by_field[key]
        cer = cer_for_pairs(refs, hyps)
        wer = wer_for_pairs(refs, hyps)
        exact = float(np.mean([ref == hyp for ref, hyp in zip(refs, hyps)]))
        metrics_rows.append({'field': key, 'field_name': label, 'n_samples': len(refs), 'cer': float(cer), 'wer': float(wer), 'exact_match_accuracy': exact, 'benchmark': 'controlled synthetic OCR'})
    all_refs = []
    all_hyps = []
    for key in refs_by_field:
        all_refs.extend(refs_by_field[key])
        all_hyps.extend(hyps_by_field[key])
    overall_cer = cer_for_pairs(all_refs, all_hyps)
    overall_wer = wer_for_pairs(all_refs, all_hyps)
    overall_exact = float(np.mean([ref == hyp for ref, hyp in zip(all_refs, all_hyps)]))
    metrics_rows.append({'field': 'overall', 'field_name': 'Tổng hợp 3 trường', 'n_samples': len(all_refs), 'cer': float(overall_cer), 'wer': float(overall_wer), 'exact_match_accuracy': overall_exact, 'benchmark': 'controlled synthetic OCR'})
    ghi_csv(OUT_DIR / 'ocr_details.csv', details)
    ghi_csv(OUT_DIR / 'ocr_metrics.csv', metrics_rows)
    print('\n--- OCR PERFORMANCE ---')
    for row in metrics_rows:
        print(f"{row['field_name']}: CER={row['cer']:.2%} | WER={row['wer']:.2%} | Exact={row['exact_match_accuracy']:.2%}")
    return {'cer': float(overall_cer), 'wer': float(overall_wer), 'exact': overall_exact}

def tao_geometry_sample(rng):
    size = 600
    img = np.full((size, size), 255, dtype=np.uint8)
    length = float(rng.integers(90, 380))
    angle_deg = float(rng.uniform(-75, 75))
    angle = math.radians(angle_deg)
    cx = int(rng.integers(180, 420))
    cy = int(rng.integers(180, 420))
    dx = math.cos(angle) * length / 2
    dy = math.sin(angle) * length / 2
    x1 = int(round(cx - dx))
    y1 = int(round(cy - dy))
    x2 = int(round(cx + dx))
    y2 = int(round(cy + dy))
    cv2.line(img, (x1, y1), (x2, y2), 0, 2, cv2.LINE_AA)
    if rng.random() < 0.65:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.2, 0.8))
    noise = rng.normal(0, rng.uniform(1.0, 5.0), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    gt_length = float(np.hypot(x2 - x1, y2 - y1))
    return (img, {'gt_length_px': gt_length, 'gt_angle_deg': angle_deg})

def do_geometry_length(img):
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    skeleton = skeletonize(binary > 0)
    skeleton_img = skeleton.astype(np.uint8) * 255
    lines = cv2.HoughLinesP(skeleton_img, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=12)
    if lines is None:
        return None
    lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        lengths.append(length)
    return max(lengths)

def ve_geometry_scatter(gt, pred):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(gt, pred)
    lower = float(min(np.min(gt), np.min(pred)))
    upper = float(max(np.max(gt), np.max(pred)))
    ax.plot([lower, upper], [lower, upper], linestyle='--')
    ax.set_xlabel('Ground truth length (px)')
    ax.set_ylabel('Predicted length (px)')
    ax.set_title('Geometry Length Evaluation')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'geometry_scatter.png', dpi=180)
    plt.close(fig)

def danh_gia_geometry_auto():
    rng = np.random.default_rng(RANDOM_SEED + 1)
    rows = []
    print(f'[GEOMETRY] controlled benchmark: {GEOMETRY_TEST_COUNT} mau')
    for i in range(1, GEOMETRY_TEST_COUNT + 1):
        img, gt = tao_geometry_sample(rng)
        pred_length = do_geometry_length(img)
        if i <= 5:
            luu_anh(GEOMETRY_BENCH_DIR / f'geometry_sample_{i:03d}.png', img)
        detected = int(pred_length is not None)
        if pred_length is None:
            pred_length = 0.0
        error = pred_length - gt['gt_length_px']
        rows.append({'sample_id': i, 'gt_length_px': gt['gt_length_px'], 'pred_length_px': pred_length, 'error_px': error, 'abs_error_px': abs(error), 'detected': detected})
    gt_values = np.array([row['gt_length_px'] for row in rows], dtype=float)
    pred_values = np.array([row['pred_length_px'] for row in rows], dtype=float)
    mae = float(np.mean(np.abs(pred_values - gt_values)))
    rmse = float(np.sqrt(np.mean((pred_values - gt_values) ** 2)))
    mape = float(np.mean(np.abs((pred_values - gt_values) / gt_values)))
    detection_rate = float(np.mean([row['detected'] for row in rows]))
    metrics = {'n_samples': len(rows), 'mae_px': mae, 'rmse_px': rmse, 'mape': mape, 'line_detection_rate': detection_rate, 'benchmark': 'controlled synthetic geometry'}
    ghi_csv(OUT_DIR / 'geometry_details.csv', rows)
    ghi_csv(OUT_DIR / 'geometry_metrics.csv', [metrics])
    ve_geometry_scatter(gt_values, pred_values)
    print('\n--- GEOMETRY EXTRACTION ---')
    print(f'Detection rate : {detection_rate:.2%}')
    print(f'MAE            : {mae:.3f} px')
    print(f'RMSE           : {rmse:.3f} px')
    print(f'MAPE           : {mape:.2%}')
    return metrics

def luu_tong_hop(tamper_metrics, ocr_metrics, geometry_metrics):
    rows = []
    if tamper_metrics:
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
            rows.append({'group': 'Tampering Detection', 'metric': key, 'value': tamper_metrics[key], 'note': 'Grouped OOF on current Real/Text-Tamper dataset'})
    if ocr_metrics:
        rows.extend([{'group': 'OCR', 'metric': 'CER', 'value': ocr_metrics['cer'], 'note': 'Controlled synthetic OCR benchmark'}, {'group': 'OCR', 'metric': 'WER', 'value': ocr_metrics['wer'], 'note': 'Controlled synthetic OCR benchmark'}])
    if geometry_metrics:
        rows.extend([{'group': 'Geometry', 'metric': 'MAE_px', 'value': geometry_metrics['mae_px'], 'note': 'Controlled synthetic geometry benchmark'}, {'group': 'Geometry', 'metric': 'RMSE_px', 'value': geometry_metrics['rmse_px'], 'note': 'Controlled synthetic geometry benchmark'}])
    ghi_csv(OUT_DIR / 'experiment_summary.csv', rows)

def main():
    tao_output()
    print('===== DANH GIA TUAN 5 - AUTO =====')
    print('\n[1] Tampering Detection')
    tamper_metrics = danh_gia_tamper_auto()
    print('\n[2] OCR Performance')
    ocr_metrics = danh_gia_ocr_auto()
    print('\n[3] Geometry Extraction')
    geometry_metrics = danh_gia_geometry_auto()
    luu_tong_hop(tamper_metrics, ocr_metrics, geometry_metrics)
    print('\nHOAN THANH TUAN 5')
    print('Ket qua:', OUT_DIR)
    print('Luu y: OCR va Geometry dung controlled synthetic benchmark; khong goi la ground truth real-world.')
if __name__ == '__main__':
    main()
