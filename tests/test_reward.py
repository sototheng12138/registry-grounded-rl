from registry_grounded_rl.environment import RegistryView
from registry_grounded_rl.reward import replay_and_score
from registry_grounded_rl.tasks import OperationStep, TaskSpec


def test_replay_reward_requires_execution_and_final_state() -> None:
    task = TaskSpec(
        task_id="rg-smoke-reward",
        split="smoke",
        request="Add 2 and 8.",
        steps=(OperationStep("add", 2, 8),),
        template_family=0,
    )
    good = replay_and_score(
        task,
        RegistryView.ORIGINAL,
        [
            {
                "type": "tool_call",
                "name": "add_integers",
                "arguments": {"lhs": 2, "rhs": 8},
            },
            {"type": "final", "answer": 10},
        ],
    )
    shortcut = replay_and_score(task, RegistryView.ORIGINAL, [{"type": "final", "answer": 10}])
    assert good["reward"] == 1.0
    assert shortcut["reward"] == 0.0

