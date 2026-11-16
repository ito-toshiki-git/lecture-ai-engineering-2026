"""第4回の実測．配布の採点関数を使い，1件ごとの結果と実行条件を保存する．"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import ModelHTTPError

from app import judge
from app.config import MODEL


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["generate", "judge"])
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--answers", type=Path)
    p.add_argument("--labels", type=Path)
    p.add_argument("--rubric", type=Path, default=judge.DEFAULT_RUBRIC)
    p.add_argument("--wait", type=float, default=2.0)
    args = p.parse_args()
    if args.mode == "generate" and args.answers is None:
        p.error("generate には --answers が必要です")
    if args.mode == "judge" and args.labels is None:
        p.error("judge には --labels が必要です")
    cases = judge.read_jsonl(args.cases)
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError("問題が空か，識別子が重複しています")
    sources = ["app/agent.py", "app/tools.py", "app/judge.py", "app/config.py", "data/glossary.json", "uv.lock"]
    sources += sorted(str(f) for f in Path("data/lectures").glob("*.md"))
    conditions = {"mode": args.mode, "model": MODEL, "cases_sha256": digest(args.cases),
                  "sources_sha256": {s: digest(s) for s in sources}}
    if args.mode == "judge":
        conditions.update(rubric_sha256=digest(args.rubric), labels_sha256=digest(args.labels), temperature=0.0)
        labels = judge.load_labels(args.labels)
        if set(labels) != {c['id'] for c in cases}:
            raise ValueError("問題と比較ラベルの識別子が一致しません")
        evaluator = judge.build_judge(judge.build_llm_model(MODEL), args.rubric.read_text(encoding="utf-8"))
    else:
        from app.agent import agent, build_model
        model = build_model()
    if args.out.exists():
        data = json.loads(args.out.read_text(encoding="utf-8"))
        if data["conditions"] != conditions:
            raise ValueError("既存結果と実行条件が違います．別の出力先を指定してください")
    else:
        data = {"conditions": conditions, "started_at": datetime.now(timezone.utc).isoformat(), "records": {}}
        save(args.out, data)
    for c in cases:
        if c["id"] in data["records"]:
            continue
        start = time.perf_counter()
        if args.mode == "generate":
            result = agent.run_sync(c["question"], model=model,
                                    usage_limits=UsageLimits(request_limit=8, tool_calls_limit=12))
            if not isinstance(result.output, str):
                raise RuntimeError("承認が必要な操作が要求されました．自動承認せず停止します")
            record = {"answer": result.output,
                      "messages": json.loads(result.all_messages_json())}
        else:
            result = evaluator.run_sync(judge.judge_prompt(c), usage_limits=UsageLimits(request_limit=3))
            record = {"verdicts": {k: getattr(result.output, k).model_dump() for k, _ in judge.LLM_CRITERIA}}
            record["verdicts"]["lecture_mark"] = {
                "reason": "最後の非空行の第N回表記を配布コードで判定",
                "verdict": judge.check_lecture_mark(c["answer"])}
        usage = result.usage
        record.update(finished_at=datetime.now(timezone.utc).isoformat(), seconds=round(time.perf_counter()-start, 3),
                      usage={k: getattr(usage, k) for k in ["requests", "input_tokens", "output_tokens", "tool_calls"]})
        data["records"][c["id"]] = record
        save(args.out, data)
        print(f"{args.mode} {c['id']}: saved ({record['seconds']}s)", flush=True)
        time.sleep(args.wait)
    if args.mode == "generate":
        judge.write_jsonl(args.answers, [{**c, "answer": data["records"][c["id"]]["answer"]} for c in cases])
        data["answers_sha256"] = digest(args.answers)
    else:
        data["verdicts"] = {cid: r["verdicts"] for cid, r in data["records"].items()}
        data["summary"] = judge.summarize(cases, data["verdicts"], labels)
        judge.print_summary(data["summary"], labeler="比較用ラベル")
    data["completed_at"] = datetime.now(timezone.utc).isoformat()
    save(args.out, data)


if __name__ == "__main__":
    # 回数制限・一時的な混雑だけを再試行する．保存済みの問題はmain内で飛ばす．
    for attempt in range(4):
        try:
            main()
            break
        except ModelHTTPError as exc:
            if exc.status_code not in (429, 503) or attempt == 3:
                raise
            delay = 40 + 10 * attempt
            event = {"at": datetime.now(timezone.utc).isoformat(), "status": exc.status_code,
                     "retry_in_seconds": delay, "attempt": attempt + 1}
            event_path = Path("lectures/04/runtime_events.jsonl")
            event_path.parent.mkdir(parents=True, exist_ok=True)
            with event_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            print(f"一時的な接続エラー {exc.status_code}．{delay}秒後に保存済み結果から再開します．", flush=True)
            time.sleep(delay)
