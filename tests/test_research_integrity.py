"""Regression tests for invalid research inputs and misleading outputs; no weights."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from PIL import Image
from src.evaluation.validation import (load_report_samples, load_qa_pairs, prepare_qa_batch,
                                       require_mode, validate_index, validate_outputs)
from src.evaluation.metrics import compute_bertscore, compute_radgraph_f1


def metadata(tmp_path, **changes):
    Image.new('RGB', (8, 8)).save(tmp_path / 'cxr.png')
    row = dict(study_id='s1', subject_id='p1', split='test', text='No acute findings.', image_path='cxr.png')
    row.update(changes)
    with (tmp_path / 'metadata.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row); w.writeheader(); w.writerow(row)
    return row


def test_missing_metadata_fails(tmp_path):
    with pytest.raises(FileNotFoundError): load_report_samples(tmp_path, 1)


def test_mock_is_explicit_and_has_requested_count(tmp_path):
    assert len(load_report_samples(tmp_path, 3, mock=True)) == 3


def test_loads_relative_real_image(tmp_path):
    metadata(tmp_path)
    assert isinstance(load_report_samples(tmp_path, 1)[0]['image'], Image.Image)


@pytest.mark.parametrize('change', [dict(image_path='missing.png'), dict(text=''), dict(split='validate'), dict(study_id='')])
def test_bad_metadata_rejected(tmp_path, change):
    metadata(tmp_path, **change)
    with pytest.raises(ValueError): load_report_samples(tmp_path, 1)


def test_corrupt_image_rejected(tmp_path):
    metadata(tmp_path); (tmp_path / 'cxr.png').write_text('not an image')
    with pytest.raises(ValueError): load_report_samples(tmp_path, 1)


def test_patient_overlap_rejected(tmp_path):
    row = metadata(tmp_path)
    row.update(study_id='s2', split='train')
    with (tmp_path / 'metadata.csv').open('a', newline='') as f:
        csv.DictWriter(f, fieldnames=row).writerow(row)
    with pytest.raises(ValueError, match='subject_id'): load_report_samples(tmp_path, 1)


def test_missing_qa_no_synthetic_fallback(tmp_path):
    with pytest.raises(FileNotFoundError): load_qa_pairs(tmp_path / 'missing.json', 1)


@pytest.mark.parametrize('change', [dict(answer=''), dict(image='missing.png'), dict(question=''), dict(study_id='')])
def test_bad_qa_rejected(tmp_path, change):
    metadata(tmp_path)
    row = dict(study_id='s1', image='cxr.png', question='Acute finding?', answer='No.')
    row.update(change); path=tmp_path / 'qa.json';path.write_text(json.dumps([row]))
    with pytest.raises(ValueError): load_qa_pairs(path, 1)


def test_mock_environment_cannot_be_real(monkeypatch):
    monkeypatch.setenv('MOCK_MODELS', 'true')
    with pytest.raises(ValueError): require_mode(False)
    require_mode(True)


def test_mock_method_requires_flag(monkeypatch):
    monkeypatch.delenv('MOCK_MODELS', raising=False)
    with pytest.raises(ValueError): require_mode(False, ['mock'])


def test_index_provenance_and_overlap():
    row=dict(study_id='train1',subject_id='p1',split='train',data_mode='real',model_mode='real')
    idx=SimpleNamespace(is_indexed=True,_metadata=[row])
    validate_index(idx,[dict(study_id='test1',subject_id='p2')])
    with pytest.raises(ValueError): validate_index(idx,[dict(study_id='test1',subject_id='p1')])
    row['data_mode']='synthetic'
    with pytest.raises(ValueError): validate_index(idx,[dict(study_id='test1',subject_id='p2')])
    idx._metadata=[dict(study_id='old')]
    with pytest.raises(ValueError): validate_index(idx,[dict(study_id='test1',subject_id='p2')])


def test_failed_prediction_not_scored():
    with pytest.raises(ValueError): validate_outputs([{'answer':'ERROR: broken','error':'broken'}],1,'answer')
    with pytest.raises(ValueError): validate_outputs([],1,'answer')


def test_missing_bertscore_is_null():
    with patch.dict(sys.modules, {'bert_score': None}):
        result=compute_bertscore(['a'], ['a'])
    assert result['bertscore_f1'] is None
    assert result['bertscore_status']=='unavailable'


def test_heuristic_is_not_radgraph():
    with patch.dict(sys.modules, {'radgraph': None}):
        result=compute_radgraph_f1(['No effusion'], ['No effusion'])
    assert result['radgraph_f1'] is None
    assert 'clinical_entity_overlap_f1' in result


def test_report_does_not_invent_findings(tmp_path):
    from scripts.run_comparison import write_markdown_report
    out=tmp_path / 'report.md'
    write_markdown_report({'mock':dict(n_valid=1,n_input=1,radgraph_f1=None,clinical_entity_overlap_f1=0.5)}, {}, SimpleNamespace(mock=True,n_samples=1), out)
    text=out.read_text()
    assert 'Unavailable' in text and 'Clinical entity-overlap heuristic F1' in text
    assert 'expected +3' not in text and '2,500 QA' not in text


def test_missing_finetuned_checkpoint_never_uses_base(tmp_path):
    from src.pipelines.report_generation import ReportGenerationPipeline
    p = ReportGenerationPipeline('medgemma-ft', cfg={'finetune': {'output_dir': str(tmp_path/'missing')}})
    with pytest.raises(FileNotFoundError): p._load_medgemma()


@pytest.mark.parametrize('kind', ['clip','colpali'])
def test_missing_rag_index_never_becomes_zero_shot(tmp_path,kind):
    from src.pipelines.report_generation import ReportGenerationPipeline
    p = ReportGenerationPipeline(kind+'-rag',cfg={'rag': {'index_path': str(tmp_path)}})
    with pytest.raises(FileNotFoundError): p._try_load_index(kind)


def test_new_index_retains_patient_provenance():
    from src.models.clip_retriever import CLIPRetriever
    r=CLIPRetriever.load(mock=True)
    r.build_index([dict(image=Image.new('RGB',(8,8)),report='Fixture',study_id='s1',
                       subject_id='p1',split='train',data_mode='synthetic')])
    assert r._metadata[0]['subject_id']=='p1'
    assert r._metadata[0]['model_mode']=='mock'
    with pytest.raises(ValueError):validate_index(r,[dict(study_id='s2',subject_id='p2')])


def test_blank_input_never_reaches_real_medgemma():
    from src.models.medgemma import MedGemmaModel
    model=MedGemmaModel(device='cpu')
    model._model = object()
    model._processor = object()
    # Private generation path rejects None before processor/model inference.
    with pytest.raises(ValueError, match='real input image'):
        model._run_inference(None, 'test', 1)
