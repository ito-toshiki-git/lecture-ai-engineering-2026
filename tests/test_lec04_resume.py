"""途中保存の再利用で重複課金を防ぎ，異なる評価条件を混ぜないことを検査する．"""

import json
import sys

import pytest

from app import judge
from tools import run_lec04


def test_completed_scoring_resumes_without_another_request(tmp_path, monkeypatch):
    out = tmp_path / "result.json"
    monkeypatch.setattr(run_lec04, "MODEL", "offline-test")
    monkeypatch.setattr(judge, "build_llm_model", lambda name: judge.build_test_model())
    monkeypatch.setattr(sys, "argv", ["run_lec04", "judge", "--cases", str(judge.DEFAULT_CASES),
                                     "--labels", str(judge.DEFAULT_LABELS), "--out", str(out), "--wait", "0"])
    run_lec04.main()
    first = json.loads(out.read_text(encoding="utf-8"))
    assert len(first["records"]) == 10
    assert sum(r["usage"]["requests"] for r in first["records"].values()) == 10

    class MustNotRun:
        def run_sync(self, *args, **kwargs):
            pytest.fail("保存済みの問題を再度採点しようとした")

    monkeypatch.setattr(judge, "build_judge", lambda *args: MustNotRun())
    run_lec04.main()
    resumed = json.loads(out.read_text(encoding="utf-8"))
    assert resumed["records"] == first["records"]
    assert resumed["summary"] == first["summary"]


def test_changed_rubric_cannot_reuse_old_results(tmp_path, monkeypatch):
    rubric = tmp_path / "rubric.md"
    rubric.write_text(judge.DEFAULT_RUBRIC.read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "result.json"
    monkeypatch.setattr(run_lec04, "MODEL", "offline-test")
    monkeypatch.setattr(judge, "build_llm_model", lambda name: judge.build_test_model())
    monkeypatch.setattr(sys, "argv", ["run_lec04", "judge", "--cases", str(judge.DEFAULT_CASES),
                                     "--labels", str(judge.DEFAULT_LABELS), "--rubric", str(rubric),
                                     "--out", str(out), "--wait", "0"])
    run_lec04.main()
    original = out.read_bytes()
    rubric.write_text(rubric.read_text(encoding="utf-8") + "\n別の方針\n", encoding="utf-8")
    with pytest.raises(ValueError, match="実行条件が違います"):
        run_lec04.main()
    assert out.read_bytes() == original
