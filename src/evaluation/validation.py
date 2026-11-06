"""Strict, shared evaluation inputs. No real-to-synthetic fallback."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from PIL import Image


def require_mode(mock: bool, names=(), cfg=None):
    if not mock and ("mock" in names or os.getenv("MOCK_MODELS", "").lower() in
                     ("true", "1", "yes") or (cfg or {}).get("mock_models")):
        raise ValueError("Real evaluation conflicts with mock models; use --mock explicitly.")


def required_text(value, field):
    if not isinstance(value, str) or not value.strip() or value.strip().lower() == "nan":
        raise ValueError(f"Missing or invalid {field}")
    return value.strip()


def load_image(value, base=Path('.'), mock=False):
    try:
        if isinstance(value, Image.Image):
            return value.convert('RGB')
        path = Path(required_text(value, 'image path'))
        if not path.is_absolute():
            path = base / path
        with Image.open(path) as image:
            return image.convert('RGB')
    except (ValueError, OSError, TypeError) as exc:
        if mock:
            return Image.new('RGB', (448, 448), color=128)
        raise ValueError(f"Unreadable evaluation image: {value!r}") from exc


def check_split_disjoint(rows):
    """Check the whole metadata file, including rows outside the selected subset."""
    for field in ('study_id', 'subject_id'):
        seen = {}
        for row in rows:
            ident = row.get(field, '').strip()
            split = row.get('split', '').strip()
            if ident:
                if ident in seen and seen[ident] != split:
                    raise ValueError(f"{field} overlaps dataset splits")
                seen[ident] = split


def load_report_samples(data_dir, n, split='test', mock=False):
    if n < 1:
        raise ValueError('n-samples must be positive')
    if mock:
        return [dict(study_id=f'synthetic_{i}', subject_id=f'synthetic_{i}',
                     split=split, image=Image.new('RGB', (448, 448), 128),
                     report='Synthetic fixture: no acute findings.') for i in range(n)]
    path = Path(data_dir) / 'metadata.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = [{k.strip().lower().replace(' ', '_'): (v or '') for k, v in r.items()}
                for r in reader]
    if not rows or not {'study_id', 'text', 'split'}.issubset(rows[0]):
        raise ValueError('metadata.csv requires study_id, text, split and image_path/image')
    check_split_disjoint(rows)
    selected = [r for r in rows if r['split'].strip() == split][:n]
    if not selected:
        raise ValueError(f'No samples in requested split {split!r}; no split fallback allowed')
    samples = []
    for row in selected:
        samples.append(dict(study_id=required_text(row['study_id'], 'study_id'),
                            subject_id=row.get('subject_id', '').strip(), split=split,
                            image=load_image(row.get('image_path') or row.get('image'), path.parent),
                            report=required_text(row['text'], 'reference report')))
    return samples


def load_qa_pairs(path, n, mock=False):
    if n < 1:
        raise ValueError('n-samples must be positive')
    if mock:
        return [dict(study_id=f'synthetic_{i}', subject_id=f'synthetic_{i}',
                     image=None, question='Is there an acute finding?',
                     answer='No acute findings.', question_type='presence') for i in range(n)]
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list) or not data:
        raise ValueError('QA dataset must be a nonempty list')
    pairs = []
    for row in data[:n]:
        row = dict(row)
        for key in ('study_id', 'question', 'answer'):
            row[key] = required_text(row.get(key), key)
        row['image'] = load_image(row.get('image_path') or row.get('image'), path.parent)
        pairs.append(row)
    return pairs


def prepare_qa_batch(pairs, mock=False):
    if not pairs:
        raise ValueError('Empty QA evaluation set')
    return [dict(image=load_image(q.get('image'), mock=mock),
                 question=required_text(q.get('question'), 'question'),
                 answer=required_text(q.get('answer'), 'reference answer'),
                 study_id=required_text(q.get('study_id'), 'study_id')) for q in pairs]


def validate_reports(samples):
    if not samples:
        raise ValueError('Empty report evaluation set')
    for sample in samples:
        required_text(sample.get('report'), 'reference report')
        required_text(sample.get('study_id'), 'study_id')
        if not isinstance(sample.get('image'), Image.Image):
            raise ValueError('Every evaluation sample must contain a loaded PIL image')


def validate_index(retriever, samples):
    """Fail closed on legacy indexes without split/patient provenance."""
    if retriever is None or not retriever.is_indexed:
        raise ValueError('A populated retrieval index is required for RAG evaluation')
    metadata = getattr(retriever, '_metadata', [])
    if not metadata:
        raise ValueError('Index metadata is missing')
    for row in metadata:
        if row.get('split') != 'train' or not row.get('subject_id') or not row.get('study_id'):
            raise ValueError('Rebuild index with train split, study_id and subject_id provenance')
        if row.get('data_mode') != 'real' or row.get('model_mode') != 'real':
            raise ValueError('Real RAG evaluation cannot use a synthetic/mock or unverified index')
    for field in ('study_id', 'subject_id'):
        test_ids = {required_text(s.get(field), field) for s in samples}
        if test_ids & {str(r[field]) for r in metadata}:
            raise ValueError(f'Retrieval index overlaps evaluation {field}')


def validate_outputs(results, expected, field):
    if len(results) != expected:
        raise ValueError('Inference output count differs from input count')
    failures = sum(bool(r.get('error')) or not str(r.get(field, '')).strip() for r in results)
    if failures:
        raise ValueError(f'Inference failed or returned empty outputs for {failures}/{expected} samples; no aggregate scored')
