import pytest

from myvla.eval_utils import prepare_eval_output


def test_output_uses_checkpoint_step_and_keeps_parent(tmp_path):
    out = prepare_eval_output(tmp_path / "eval" / "v1", 27000)
    assert out == tmp_path / "eval" / "v1_27000"
    assert out.is_dir()
    assert prepare_eval_output(out, 27000) == out


def test_execution_length_or_incorrect_step_in_prefix_does_not_hide_actual_step(tmp_path):
    assert prepare_eval_output(tmp_path / "v1_12", 27000).name == "v1_12_27000"
    assert prepare_eval_output(tmp_path / "v1_25000", 4000).name == "v1_25000_4000"


@pytest.mark.parametrize("filename", ["results.json", "episodes.jsonl"])
def test_existing_reports_are_not_overwritten(tmp_path, filename):
    out = prepare_eval_output(tmp_path / "v1", 4000)
    report = out / filename
    report.write_text("existing result")
    with pytest.raises(FileExistsError, match="v1_4000"):
        prepare_eval_output(tmp_path / "v1", 4000)
    assert report.read_text() == "existing result"
