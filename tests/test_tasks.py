from registry_grounded_rl.tasks import PREVIOUS, OperationStep, TaskSpec, generate_tasks


def test_task_expected_answer_and_roundtrip() -> None:
    task = TaskSpec(
        task_id="rg-smoke-manual",
        split="smoke",
        request="Add 3 and 4, then multiply by 5.",
        steps=(
            OperationStep("add", 3, 4),
            OperationStep("multiply", PREVIOUS, 5),
        ),
        template_family=0,
    )
    assert task.expected_answer == 35
    assert TaskSpec.from_dict(task.to_dict()) == task
    assert task.gold_calls()[1]["arguments"] == {"lhs": 7, "rhs": 5}


def test_generator_is_deterministic() -> None:
    left = generate_tasks(10, seed=9, split="smoke")
    right = generate_tasks(10, seed=9, split="smoke")
    assert left == right
    assert len({task.task_id for task in left}) == 10

