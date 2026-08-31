from registry_grounded_rl.local_qwen import parse_agentgym_action, parse_model_action


def test_parse_single_tool_call() -> None:
    parsed = parse_model_action(
        '<tool_call>{"name":"add_integers","arguments":{"lhs":2,"rhs":3}}</tool_call>'
    )
    assert parsed.parse_error is None
    assert parsed.action == {
        "type": "tool_call",
        "name": "add_integers",
        "arguments": {"lhs": 2, "rhs": 3},
    }


def test_parse_final_and_unavailable() -> None:
    assert parse_model_action('<final>{"answer":5}</final>').action == {
        "type": "final",
        "answer": 5,
    }
    assert parse_model_action('<final>{"answer":"done"}</final>').action == {
        "type": "final",
        "answer": "done",
    }
    assert parse_model_action(
        '<unavailable>{"reason":"missing multiply"}</unavailable>'
    ).action == {"type": "unavailable", "reason": "missing multiply"}


def test_multiple_or_plain_blocks_fail_closed() -> None:
    assert parse_model_action("5").action is None
    assert parse_model_action(
        '<final>{"answer":5}</final><final>{"answer":6}</final>'
    ).action is None


def test_agentgym_parser_accepts_one_whole_response_json_action() -> None:
    assert parse_agentgym_action(
        '{"type":"tool_call","name":"list_projects","arguments":{}}'
    ).action == {"type": "tool_call", "name": "list_projects", "arguments": {}}
    assert parse_agentgym_action(
        '{"name":"list_projects","arguments":{}}'
    ).action == {"type": "tool_call", "name": "list_projects", "arguments": {}}
    assert parse_agentgym_action('{"type":"final","answer":"done"}').action == {
        "type": "final",
        "answer": "done",
    }


def test_agentgym_parser_rejects_prose_and_concatenated_json() -> None:
    assert parse_agentgym_action(
        'I will act. {"type":"tool_call","name":"list_projects","arguments":{}}'
    ).action is None
    assert parse_agentgym_action(
        '{"type":"tool_call","name":"a","arguments":{}}'
        '{"type":"tool_call","name":"b","arguments":{}}'
    ).action is None
