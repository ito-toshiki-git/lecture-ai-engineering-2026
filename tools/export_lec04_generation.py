"""回答生成の記録から，回答・使用モデル・ツール操作・使用量を取り出す．"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("lectures/04/generation_result.json"))
    p.add_argument("--out", type=Path, default=Path("lectures/04/generation_evidence.json"))
    args = p.parse_args()
    raw = args.raw.read_bytes()
    data = json.loads(raw)
    if "completed_at" not in data:
        raise ValueError("回答生成が完了していません")
    data["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    for record in data["records"].values():
        messages = record.pop("messages")
        record["observed_model_names"] = sorted({m["model_name"] for m in messages if m.get("model_name")})
        record["tool_trace"] = [
            {key: part[key] for key in ["part_kind", "tool_name", "args", "content"] if key in part}
            for m in messages for part in m.get("parts", [])
            if part.get("part_kind") in ("tool-call", "tool-return")
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(data['records'])} completed answers to {args.out}")


if __name__ == "__main__":
    main()
