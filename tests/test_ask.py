"""第1回の対話機能。実際のSDKを使い、HTTP通信だけを模擬する（APIキー不要）。"""

import json

import httpx
import pytest
from google import genai
from google.genai import types

from app import ask


def reply(text="模擬回答", *, usage=True):
    payload = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": text, "thoughtSignature": "dGVzdA=="}],
            },
            "finishReason": "STOP",
        }],
    }
    if usage:
        payload["usageMetadata"] = {"promptTokenCount": 12, "candidatesTokenCount": 7}
    return payload


@pytest.fixture
def api_stub(monkeypatch):
    """リクエスト本文を記録する。外部サーバーには接続しない。"""
    requests = []
    responses = []
    real_client = genai.Client
    clients = []
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-credential")
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)

    def handle(request):
        requests.append(json.loads(request.content))
        status, payload = responses.pop(0) if responses else (200, reply())
        return httpx.Response(status, json=payload)

    def make_client(**kwargs):
        transport_client = httpx.Client(transport=httpx.MockTransport(handle))
        clients.append(transport_client)
        return real_client(
            **kwargs,
            http_options=types.HttpOptions(
                httpx_client=transport_client,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    monkeypatch.setattr(ask.genai, "Client", make_client)
    yield requests, responses
    for client in clients:
        client.close()


def inputs(monkeypatch, lines):
    lines = iter(lines)

    def read(_prompt):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", read)


def test_second_request_contains_previous_question_and_full_answer(api_stub, monkeypatch):
    requests, responses = api_stub
    responses.append((200, reply("合言葉は青いペンギンです")))
    inputs(monkeypatch, ["さっきの合言葉は？", "/exit"])
    ask.run_chat("合言葉は青いペンギンです")

    assert len(requests) == 2
    history = requests[1]["contents"]
    assert [item["role"] for item in history] == ["user", "model", "user"]
    assert [item["parts"][0]["text"] for item in history] == [
        "合言葉は青いペンギンです", "合言葉は青いペンギンです", "さっきの合言葉は？",
    ]
    # 本文だけを保存すると失われるSDKのメタデータも保持する。
    assert history[1]["parts"][0]["thoughtSignature"] == "dGVzdA=="


def test_reset_starts_a_new_history(api_stub, monkeypatch):
    requests, _ = api_stub
    inputs(monkeypatch, ["/reset", "新しい質問", "/exit"])
    ask.run_chat("古い質問")
    assert len(requests) == 2
    assert requests[1]["contents"] == [{"role": "user", "parts": [{"text": "新しい質問"}]}]


def test_blank_input_and_exit_do_not_call_api(api_stub, monkeypatch):
    requests, _ = api_stub
    inputs(monkeypatch, ["", "   ", "/exit"])
    ask.run_chat()
    assert requests == []


@pytest.mark.parametrize("stop", [EOFError, KeyboardInterrupt])
def test_input_interrupt_exits_cleanly(api_stub, monkeypatch, capsys, stop):
    requests, _ = api_stub

    def interrupted(_prompt):
        raise stop

    monkeypatch.setattr("builtins.input", interrupted)
    ask.run_chat()
    assert requests == []
    assert "対話を終了" in capsys.readouterr().out


@pytest.mark.parametrize("status", [400, 429, 503])
def test_api_error_does_not_add_failed_turn_or_erase_success(api_stub, monkeypatch, capsys, status):
    requests, responses = api_stub
    responses.extend([
        (200, reply("最初の回答")),
        (status, {"error": {"code": status, "message": "simulated failure", "status": "UNKNOWN"}}),
        (200, reply("続きの回答")),
    ])
    inputs(monkeypatch, ["失敗する質問", "続きの質問", "/exit"])
    ask.run_chat("最初の質問")
    assert len(requests) == 3
    assert [c["parts"][0]["text"] for c in requests[2]["contents"]] == [
        "最初の質問", "最初の回答", "続きの質問",
    ]
    assert "自動再送しません" in capsys.readouterr().out


@pytest.mark.parametrize("question", [None, "LoRAとは？"])
def test_single_question_cli_remains_available(api_stub, monkeypatch, capsys, question):
    requests, _ = api_stub
    monkeypatch.setattr("sys.argv", ["app.ask"] + ([question] if question else []))
    ask.main()
    assert len(requests) == 1
    assert requests[0]["contents"][0]["parts"][0]["text"] == (question or ask.DEFAULT_QUESTION)
    output = capsys.readouterr().out
    assert "模擬回答" in output
    assert "入力: 12 / 出力: 7" in output


def test_chat_cli_uses_follow_up_questions(api_stub, monkeypatch):
    requests, _ = api_stub
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat", "最初の質問"])
    inputs(monkeypatch, ["続きの質問", "/exit"])
    ask.main()
    assert len(requests) == 2
    assert len(requests[1]["contents"]) == 3


@pytest.mark.parametrize("metadata, expected", [
    (None, "入力: 不明 / 出力: 不明"),
    ({"prompt_token_count": 0, "candidates_token_count": 0}, "入力: 0 / 出力: 0"),
    ({"prompt_token_count": 12}, "入力: 12 / 出力: 不明"),
])
def test_usage_distinguishes_missing_data_from_zero(capsys, metadata, expected):
    ask.print_usage(types.GenerateContentResponse(usage_metadata=metadata))
    assert expected in capsys.readouterr().out


def test_response_without_text_is_explained(api_stub, capsys):
    _, responses = api_stub
    responses.append((200, {"promptFeedback": {"blockReason": "SAFETY"}}))
    assert ask.ask("質問") == "（テキストの回答がありませんでした）"
    assert "不明" in capsys.readouterr().out


def test_missing_key_fails_with_setup_instructions(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        ask.ask("質問")
