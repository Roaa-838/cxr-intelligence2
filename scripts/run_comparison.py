#!/usr/bin/env python3
"""Compare pipelines without inventing results. Outputs are separated by mode."""
from __future__ import annotations
import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_report_gen import load_test_samples, run_model_evaluation
from scripts.evaluate_qa import load_qa_dataset, run_qa_evaluation
from src.evaluation.validation import require_mode


def compare_report_generation(samples, models, cfg, mock, output_dir):
    results = {}
    for name in models:
        try:
            results[name] = run_model_evaluation(name, samples, cfg, mock, output_dir)
        except Exception as exc:
            logging.exception('Report evaluation failed: %s', name)
            results[name] = {'status': 'failed', 'error': str(exc)}
    return results


def compare_qa(qa_pairs, retrievers, cfg, mock, output_dir):
    results = {}
    for name in retrievers:
        try:
            results[name] = run_qa_evaluation(name, qa_pairs, cfg, mock, output_dir)
        except Exception as exc:
            logging.exception('QA evaluation failed: %s', name)
            results[name] = {'status': 'failed', 'error': str(exc)}
    return results


def write_markdown_report(report_metrics, qa_metrics, args, output_path):
    mode = 'mock' if args.mock else 'real'
    lines = ['# CXR evaluation report', '', f'Evaluation mode: **{mode}**.',
             f'Requested maximum samples per task: {args.n_samples}.', '',
             'Mock runs test software only; they do not measure medical model performance.' if args.mock else
             'Real inference is not clinical validation. Review data provenance and metric availability.', '',
             '| Task | Method | Valid / input | Metric | Value |',
             '|---|---|---|---|---|']
    for task, group in [('Report generation', report_metrics), ('QA', qa_metrics)]:
        for name, m in group.items():
            if 'error' in m:
                lines.append(f'| {task} | {name} | — | Failed; see JSON error | — |')
                continue
            for key, label in [('radgraph_f1', 'RadGraph F1'),
                               ('clinical_entity_overlap_f1', 'Clinical entity-overlap heuristic F1'),
                               ('bertscore_f1', 'BERTScore F1'), ('rouge_l', 'ROUGE-L'),
                               ('bleu_4', 'BLEU-4'), ('token_f1', 'Token F1 (unique tokens)'),
                               ('exact_match', 'Exact match')]:
                if key not in m:
                    continue
                value = m[key]
                value = f'{value:.4f}' if isinstance(value, (int, float)) else 'Unavailable'
                lines.append(f'| {task} | {name} | {m.get("n_valid", 0)} / {m.get("n_input", 0)} | {label} | {value} |')
    lines += ['', '## Interpretation', '',
              'Unavailable metrics are null, not zero. The entity-overlap heuristic is not RadGraph and does not validate negation or laterality.',
              'No superiority, fine-tuning benefit, hallucination reduction, or clinical accuracy is inferred automatically.',
              'See the per-method JSON files for evaluator status, versions, failures, and effective model settings.']
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='data/processed')
    p.add_argument('--qa-dataset', default='data/qa_dataset.json')
    p.add_argument('--output-dir', default='results')
    p.add_argument('--n-samples', type=int, default=500)
    p.add_argument('--report-models', nargs='+', default=['medgemma', 'clip-rag'],
                   choices=['medgemma', 'medgemma-ft', 'clip-rag', 'colpali-rag', 'mock'])
    p.add_argument('--qa-retrievers', nargs='+', default=['colpali', 'clip'], choices=['colpali', 'clip', 'mock'])
    p.add_argument('--mock', action='store_true')
    p.add_argument('--skip-report', action='store_true')
    p.add_argument('--skip-qa', action='store_true')
    p.add_argument('--config', default='configs/default.yaml')
    return p.parse_args()


def main():
    args = parse_args()
    if args.skip_report and args.skip_qa:
        raise ValueError('At least one evaluation task must be enabled')
    from src.config import load_config
    cfg = load_config(args.config)
    require_mode(args.mock, args.report_models + args.qa_retrievers, cfg)
    mode = 'mock' if args.mock else 'real'
    output_dir = Path(args.output_dir) / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    # Validate all requested inputs before loading expensive models.
    samples = [] if args.skip_report else load_test_samples(Path(args.data_dir), args.n_samples, mock=args.mock)
    pairs = [] if args.skip_qa else load_qa_dataset(args.qa_dataset, args.n_samples, args.mock)
    reports = {} if args.skip_report else compare_report_generation(
        samples, ['mock'] if args.mock else args.report_models, cfg, args.mock, output_dir)
    qa = {} if args.skip_qa else compare_qa(
        pairs, ['mock'] if args.mock else args.qa_retrievers, cfg, args.mock, output_dir)
    payload = dict(report_generation=reports, qa=qa, evaluation_mode=mode,
                   args=vars(args), timestamp=datetime.now(timezone.utc).isoformat())
    (output_dir / 'all_metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    keys = ['task', 'model', 'evaluation_mode', 'status', 'n_input', 'n_valid', 'n_excluded',
            'radgraph_f1', 'clinical_entity_overlap_f1', 'bertscore_f1', 'rouge_l', 'bleu_4',
            'token_f1', 'exact_match', 'error']
    with (output_dir / 'comparison_table.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for task, group in [('report_gen', reports), ('qa', qa)]:
            for name, m in group.items():
                writer.writerow(dict(m, task=task, model=name, evaluation_mode=mode,
                                     status='failed' if 'error' in m else 'completed'))
    write_markdown_report(reports, qa, args, output_dir / 'comparison_report.md')
    if any('error' in m for group in (reports, qa) for m in group.values()):
        raise SystemExit(1)
    print(f'Completed {mode} software/evaluation run: {output_dir}')


if __name__ == '__main__':
    main()
