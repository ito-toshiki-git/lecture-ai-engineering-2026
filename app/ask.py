"""第1回: LLMへの単発質問と、会話履歴を保持した対話。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話を続ける（/reset で履歴消去、/exit で終了）
    uv run python -m app.ask --chat "LoRAとは何ですか？"

用語集を検索するツールは使わない。--chat のときだけ、同じセッションで履歴を送る。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response: types.GenerateContentResponse) -> None:
    """APIの返した値を表示する。欠損を0トークンと誤解しないよう区別する。"""
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage else None
    output_tokens = usage.candidates_token_count if usage else None
    print(
        f"[トークン数] 入力: {input_tokens if input_tokens is not None else '不明'} / "
        f"出力: {output_tokens if output_tokens is not None else '不明'}"
    )


def response_text(response: types.GenerateContentResponse) -> str:
    return response.text or "（テキストの回答がありませんでした）"


def ask(question: str) -> str:
    """従来どおり、履歴なしで1回だけ質問する。"""
    with genai.Client(api_key=get_api_key()) as client:
        response = client.models.generate_content(model=MODEL, contents=question)
    print_usage(response)
    return response_text(response)


def run_chat(first_question: str | None = None) -> None:
    """SDKのChatを使い回す。履歴はこの実行中だけ保持し、ファイルには保存しない。"""
    print("対話モード：/exit で終了，/reset で会話履歴を消去します．")
    with genai.Client(api_key=get_api_key()) as client:
        chat = client.chats.create(model=MODEL)
        question = first_question
        while True:
            if question is None:
                try:
                    question = input("あなた> ")
                except (EOFError, KeyboardInterrupt):
                    print("\n対話を終了します．")
                    return
            else:
                print(f"あなた> {question}")

            message = question.strip()
            question = None
            if not message:
                continue
            if message == "/exit":
                print("対話を終了します．")
                return
            if message == "/reset":
                chat = client.chats.create(model=MODEL)
                print("会話履歴を消去しました．")
                continue

            try:
                # 同じChatを使うと、前の質問とモデルの回答もSDKが送信する。
                # 回答を文字列だけで再構成せず、SDKに履歴管理を任せる。
                response = chat.send_message(message)
            except errors.APIError as e:
                print(api_error_message(e.code, e.message))
                print("質問は自動再送しません．再入力するか，/exit で終了してください．")
                continue
            print(f"AI> {response_text(response)}")
            print_usage(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", help="最初の質問（単発モードの省略時は講座の質問）")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保持して続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            run_chat(args.question)
        else:
            question = args.question if args.question is not None else DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\n対話を終了します．")


if __name__ == "__main__":
    main()
