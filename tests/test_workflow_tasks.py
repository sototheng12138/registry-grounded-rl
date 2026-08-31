from registry_grounded_rl.workflow_tasks import WorkflowTaskSpec, generate_workflow_tasks


def test_workflow_generation_is_deterministic_and_roundtrips() -> None:
    first = generate_workflow_tasks(8, seed=1701, split="train")
    second = generate_workflow_tasks(8, seed=1701, split="train")
    assert first == second
    for task in first:
        assert WorkflowTaskSpec.from_dict(task.to_dict()) == task
        assert task.target_ticket.ticket_id not in task.request
        assert task.target_project.project_id not in task.request
        assert task.expected_target.status == task.desired_status
        assert task.expected_target.assignee == task.desired_assignee
        assert task.desired_label in task.expected_target.labels


def test_workflow_split_seeds_produce_disjoint_ids() -> None:
    train = generate_workflow_tasks(5, seed=1701, split="train")
    test = generate_workflow_tasks(5, seed=1703, split="test")
    assert {task.task_id for task in train}.isdisjoint(task.task_id for task in test)
    assert {ticket.ticket_id for task in train for ticket in task.tickets}.isdisjoint(
        ticket.ticket_id for task in test for ticket in task.tickets
    )
