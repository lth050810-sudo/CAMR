"""Bound current-data traversal while allowing separate historical replay steps."""


def training_batches(target, replay, rng, replay_fraction, probabilities,
                     step_limit, current_batches=None):
    if current_batches is not None and not 0 <= replay_fraction < 1:
        raise ValueError('Single-pass training requires replay_fraction < 1')
    step = consumed = 0
    while (consumed < current_batches if current_batches is not None else step < step_limit):
        replay_index = None
        loader = target
        if replay and rng.random() < replay_fraction:
            replay_index = rng.choices(range(len(replay)), weights=probabilities)[0]
            loader = replay[replay_index]
        else:
            consumed += 1
        yield step, replay_index, next(loader)
        step += 1
