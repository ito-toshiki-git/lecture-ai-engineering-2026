"""明示的に実行するGemini実測．通常の自動テストからは呼ばない．

実行例: uv run python scripts/check_lec03_live.py
既存の学習記録がある場合は止まり，検証用データでの上書きを避ける．
承認・拒否の入力は検証用にこのスクリプトが送る（本人の手動操作ではない）．
"""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "lectures/03/live_results.json"
LOG_DIR = ROOT / "learning_log"
CONTENT = "エージェントではモデルが道具を選び，実行はプログラムが行う．"


def snapshot():
    return {path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(LOG_DIR.glob("*.json"))}


def main():
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY が未設定です．")
    if LOG_DIR.exists() and any(LOG_DIR.iterdir()):
        raise SystemExit("learning_log/ に既存記録があります．別の作業コピーで実行してください．")
    results = {
        "date_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("pydantic-ai-slim", "google-genai", "pytest")},
        "approval_input": "検証用のy/nはスクリプトが入力した．本人の手動承認とは区別する．",
        "runs": [],
    }
    course = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
    save = f"第3回の学習記録として，次の本文をそのまま保存してください．本文：{CONTENT}"
    cases = [
        ("plain_course", "app.ask", [course], ""),
        ("agent_course", "app.agent", ["--trace", course], ""),
        ("agent_lora_qlora", "app.agent", ["--trace", "LoRAとQLoRAの違いは？"], ""),
        ("agent_quiz", "app.agent", ["--trace", "第2回の内容から確認問題を1問出して"], ""),
        ("save_denied", "app.agent", ["--trace", save], "n\n"),
        ("save_approved", "app.agent", ["--trace", save], "y\n"),
        ("save_duplicate", "app.agent", ["--trace", save], "y\n"),
    ]
    env = {**os.environ, "PYDANTIC_AI_NO_BANNER": "1", "PYTHONIOENCODING": "utf-8"}
    for name, module, args, stdin in cases:
        before = snapshot()
        start = time.perf_counter()
        run = subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, env=env,
                             input=stdin, capture_output=True, text=True, encoding="utf-8", timeout=120)
        after = snapshot()
        record = {"name": name, "module": module, "args": args, "stdin": stdin,
                  "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
                  "elapsed_seconds": round(time.perf_counter() - start, 3),
                  "files_before": before, "files_after": after}
        results["runs"].append(record)
        # 念のため，認証キーがライブラリのエラーに混ざっても保存しない．
        serialized = json.dumps(results, ensure_ascii=False, indent=2)
        OUTPUT.write_text(serialized.replace(os.environ["GEMINI_API_KEY"], "[REDACTED]") + "\n", encoding="utf-8")
        print(f"{name}: exit={run.returncode}, files={len(before)}->{len(after)}", flush=True)
        if run.returncode:
            raise SystemExit("実測が失敗したため停止しました．live_results.json を確認してください．")
        if name == "save_denied":
            assert not after and "[拒否]" in run.stdout
        if name == "save_approved":
            assert len(after) == 1 and "[承認]" in run.stdout
            assert next(iter(after.values()))["content"] == CONTENT
        if name == "save_duplicate":
            assert before == after and "already_saved" in run.stdout
    print("実測と保存ファイルの確認が完了しました．", flush=True)


if __name__ == "__main__":
    main()
