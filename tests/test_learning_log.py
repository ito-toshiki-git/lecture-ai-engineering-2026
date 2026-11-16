"""保存の前後をファイルで検証する．実際のモデルや認証キーは使わない．"""

import json

import pytest
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent, run_with_approvals


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    directory = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", directory)
    return directory


def save_model(contents=("道具の実行はプログラムが担当する．",)):
    """最初だけ保存を要求し，続行後は受け取ったツール結果を返す模擬モデル．"""
    def respond(messages, info):
        returns = [part for message in messages for part in message.parts
                   if isinstance(part, ToolReturnPart)]
        if returns:
            return ModelResponse(parts=[TextPart(json.dumps(
                [part.content for part in returns], ensure_ascii=False))])
        return ModelResponse(parts=[
            ToolCallPart("save_learning_log", {"lecture": 3, "content": content},
                         tool_call_id=f"save-{index}")
            for index, content in enumerate(contents)
        ])
    return FunctionModel(respond)


def test_testmodel_stops_before_write(log_dir):
    """TestModel（テスト用モデル）も承認前にはツールを実行できない．"""
    model = TestModel(call_tools=["save_learning_log"], seed=3)
    result = agent.run_sync("保存してください", model=model)
    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.approvals[0].tool_name == "save_learning_log"
    assert not log_dir.exists()
    decisions = DeferredToolResults(approvals={
        call.tool_call_id: False for call in result.output.approvals
    })
    agent.run_sync(model=model, message_history=result.all_messages(),
                   deferred_tool_results=decisions)
    assert not log_dir.exists()


def test_approval_previews_then_saves_exact_content(log_dir, capsys):
    content = "ツールの実行はプログラムが担当する．\n保存前に確認する．"

    def approve(prompt):
        assert not log_dir.exists()
        displayed = capsys.readouterr().out
        assert json.dumps(content, ensure_ascii=False) in displayed
        assert str(log_dir) in displayed
        return "y"

    result = run_with_approvals(agent, "保存", save_model((content,)), input_func=approve)
    files = list(log_dir.glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert record["lecture"] == 3 and record["content"] == content
    assert record["created_at"]
    assert "saved" in result.output
    assert result.usage.requests == 2  # 中断・再開を通して累積する


@pytest.mark.parametrize("reply", ["n", "", "maybe"])
def test_rejection_and_ambiguous_reply_do_not_write(log_dir, reply):
    result = run_with_approvals(agent, "保存", save_model(), input_func=lambda _: reply)
    assert not log_dir.exists()
    assert "拒否" in result.output


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_interrupted_input_never_saves(log_dir, exception):
    def interrupt(prompt):
        raise exception
    with pytest.raises(SystemExit) as error:
        run_with_approvals(agent, "保存", save_model(), input_func=interrupt)
    assert error.value.code == 130
    assert not log_dir.exists()


def test_multiple_calls_get_individual_decisions(log_dir):
    replies = iter(["y", "n"])
    run_with_approvals(agent, "2件保存", save_model(("承認する本文", "拒否する本文")),
                       input_func=lambda _: next(replies))
    files = list(log_dir.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["content"] == "承認する本文"


def test_duplicate_is_not_written_twice(log_dir):
    first = json.loads(tools.save_learning_log(3, "同じ内容"))
    path = log_dir / first["path"].split("/")[-1]
    before = path.read_bytes()
    second = json.loads(tools.save_learning_log(3, "同じ内容"))
    assert first["status"] == "saved"
    assert second["status"] == "already_saved"
    assert path.read_bytes() == before
    assert len(list(log_dir.glob("*.json"))) == 1
    assert json.loads(tools.save_learning_log(3, "別の内容"))["status"] == "saved"
    assert len(list(log_dir.glob("*.json"))) == 2


@pytest.mark.parametrize("lecture, content", [(0, "本文"), (99, "本文"), (True, "本文"),
                                                 (3, ""), (3, " \n"), (3, "字" * 5001)])
def test_invalid_log_does_not_create_directory(log_dir, lecture, content):
    assert json.loads(tools.save_learning_log(lecture, content))["status"] == "error"
    assert not log_dir.exists()


def test_content_cannot_choose_a_file_path(log_dir):
    content = "../../outside.json\n学習記録"
    result = json.loads(tools.save_learning_log(3, content))
    assert result["status"] == "saved"
    assert not (log_dir.parent / "outside.json").exists()
    record = json.loads(next(log_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert record["content"] == content


def test_conflicting_existing_file_is_never_overwritten(log_dir):
    tools.save_learning_log(3, "本文")
    path = next(log_dir.glob("*.json"))
    path.write_text("[]", encoding="utf-8")
    assert json.loads(tools.save_learning_log(3, "本文"))["status"] == "error"
    assert path.read_text(encoding="utf-8") == "[]"


def test_io_failure_is_not_reported_as_saved(log_dir):
    log_dir.write_text("ここはフォルダではない", encoding="utf-8")
    assert json.loads(tools.save_learning_log(3, "本文"))["status"] == "error"


def test_model_cannot_loop_approval_forever(log_dir):
    def repeat(messages, info):
        return ModelResponse(parts=[ToolCallPart(
            "save_learning_log", {"lecture": 3, "content": "拒否された内容"},
            tool_call_id=f"save-{len(messages)}")])
    with pytest.raises(UsageLimitExceeded):
        run_with_approvals(agent, "保存", FunctionModel(repeat), input_func=lambda _: "n")
    assert not log_dir.exists()


def test_quiz_matches_one_of_the_requested_lecture_terms():
    result = json.loads(tools.make_quiz(2))
    assert result["lecture"] == 2
    assert any(result["answer"] == term["definition"]
               and result["question"] == f"「{term['term']}」の意味は？"
               for term in tools.load_glossary() if term["lecture"] == 2)


def test_quiz_handles_missing_lecture():
    assert "error" in json.loads(tools.make_quiz(0))
