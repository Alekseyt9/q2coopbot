"""Unequal family allocations using the existing native episode compiler."""
from process_combat_architecture_pool import read, save, run
from run_combat_target_refresh import compile_plan


def validate_counts(families, counts):
    assert len(families) == len(set(families)) == 20
    assert set(counts) == set(families), 'Every training family needs an explicit allocation'
    assert all(type(n) is int and 4 <= n <= 80 and n % 4 == 0 for n in counts.values())
    assert 80 <= sum(counts.values()) <= 320, 'Bounded all-family training budget'
    return max(counts.values())


def compile_weighted(compiler, registry, repo, model, folder, families, offset, counts):
    validate_counts(families, counts)
    folder.mkdir()
    tasks, template = {}, None
    for count in sorted(set(counts.values())):
        selected = [family for family in families if counts[family] == count]
        part = read(compile_plan(compiler, registry, repo, model,
                                 folder / ('allocation-' + str(count)), 'train', selected, offset, count))
        if template is None:
            template = part
        for task in part['tasks']:
            family = task['episode']['id']
            assert family not in tasks and task['split'] == 'train'
            assert len(task['seeds']) == counts[family]
            tasks[family] = task
    assert set(tasks) == set(families)
    template['tasks'] = [tasks[family] for family in families]
    template['output_root'] = str((folder / 'capture').resolve())
    path = folder / 'plan.json'
    save(path, template)
    run([compiler, '--verify-plan', path, '--root', repo], folder / 'verify-weighted-plan.log')
    return path
