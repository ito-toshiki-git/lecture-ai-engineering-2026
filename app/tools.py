"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
import hashlib
import random
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_RESULTS = 3
LEARNING_LOG_DIR = Path(__file__).resolve().parent.parent / "learning_log"


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """指定した回の用語から1つを選び，確認問題と答えを返す．

    「第2回の確認問題を出して」のように，復習問題を求められたときに使う．
    lecture は講義の回数を表す整数．該当する用語がなければ理由を返す．
    """
    entries = [entry for entry in load_glossary() if entry["lecture"] == lecture]
    if not entries:
        return json.dumps({"error": f"第{lecture}回の用語はありません．"}, ensure_ascii=False)
    entry = random.choice(entries)
    return json.dumps({
        "lecture": lecture,
        "question": f"「{entry['term']}」の意味は？",
        "answer": entry["definition"],
    }, ensure_ascii=False)


def save_learning_log(lecture: int, content: str) -> str:
    """利用者が保存を依頼した学習記録を learning_log/ に保存する．承認が必要．

    lecture は講義の回数，content は保存する本文そのもの．質問だけでは使わない．
    保存先はプログラム側で固定し，同じ回と本文は再実行しても1件にする．
    成功を返してから保存済みと伝える．拒否された保存は再度要求しない．
    """
    def result(**values) -> str:
        return json.dumps(values, ensure_ascii=False)

    if type(lecture) is not int or lecture not in {e["lecture"] for e in load_glossary()}:
        return result(status="error", message="用語集にある講義の回数を指定してください．")
    if not isinstance(content, str) or not content.strip() or len(content) > 5000:
        return result(status="error", message="本文は空白のみを除く1〜5000文字にしてください．")

    # 本文をファイル名にしない．同じ回と本文には同じ名前を使う．
    payload = {"lecture": lecture, "content": content}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    name = hashlib.sha256(canonical).hexdigest() + ".json"
    path = LEARNING_LOG_DIR / name
    try:
        LEARNING_LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            # x は新規作成専用．既存の学習記録を上書きしない．
            with path.open("x", encoding="utf-8") as stream:
                json.dump({**payload, "created_at": datetime.now(timezone.utc).isoformat()},
                          stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except FileExistsError:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict) or any(existing.get(k) != v for k, v in payload.items()):
                return result(status="error", message="既存記録の内容が一致しません．上書きせず停止しました．")
            return result(status="already_saved", path=f"learning_log/{name}")
    except (OSError, ValueError):
        return result(status="error", message="記録の読み書きに失敗しました．保存先を確認してください．")
    return result(status="saved", path=f"learning_log/{name}")
