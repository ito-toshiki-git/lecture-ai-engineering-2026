"""保存済みの第4回実測結果を，外部通信なしで再集計・検証する．"""

import hashlib
import json
from pathlib import Path

from app import judge


def digest(path):
    # WindowsとLinuxの改行の違いだけを吸収する．
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main():
    runs = [
        ("judge_before", "data/eval/lec04_cases.jsonl", "lectures/04/labels.jsonl", "data/eval/rubric_before.md"),
        ("judge_after", "data/eval/lec04_cases.jsonl", "lectures/04/labels.jsonl", "data/eval/rubric.md"),
        ("my_judge_result", "data/eval/my_cases.jsonl", "lectures/04/my_labels.jsonl", "data/eval/rubric.md"),
    ]
    for name, cp, lp, rp in runs:
        data = json.loads(Path(f"lectures/04/{name}.json").read_text(encoding="utf-8"))
        cases, labels = judge.read_jsonl(Path(cp)), judge.load_labels(Path(lp))
        ids = {c["id"] for c in cases}
        assert len(cases) == len(ids) == 10
        assert ids == set(labels) == set(data["records"]) == set(data["verdicts"])
        assert data["completed_at"]
        conditions = data["conditions"]
        assert conditions["model"] == "gemini-3.5-flash-lite"
        assert conditions["cases_sha256"] == digest(cp)
        assert conditions["labels_sha256"] == digest(lp)
        assert conditions["rubric_sha256"] == digest(rp)
        for path, expected in conditions["sources_sha256"].items():
            assert digest(path) == expected, path
        assert judge.summarize(cases, data["verdicts"], labels) == data["summary"]
        for c in cases:
            assert data["verdicts"][c["id"]] == data["records"][c["id"]]["verdicts"]
            assert data["verdicts"][c["id"]]["lecture_mark"]["verdict"] == judge.check_lecture_mark(c["answer"])
        print(f"{name}: 10 results, conditions and summary verified")
    evidence = json.loads(Path("lectures/04/generation_evidence.json").read_text(encoding="utf-8"))
    assert evidence["conditions"]["cases_sha256"] == digest("data/eval/my_questions.jsonl")
    assert evidence["answers_sha256"] == digest("data/eval/my_cases.jsonl")
    for c in judge.read_jsonl(Path("data/eval/my_cases.jsonl")):
        r = evidence["records"][c["id"]]
        assert r["answer"] == c["answer"]
        assert r["observed_model_names"] == ["gemini-3.5-flash-lite"]
        assert all(t["tool_name"] != "save_learning_log" for t in r["tool_trace"])
    print("generation: 10 actual answers verified; no learning-log write calls")


if __name__ == "__main__":
    main()
