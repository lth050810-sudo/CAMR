"""CPU-only scheduling mechanisms for explicitly labelled recent replay ports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random


@dataclass
class ReviewState:
    ease: float = 2.5
    successes: int = 0
    interval: int = 1
    due: int = 0
    reviews: int = 0

    def review(self, nll: float, step: int, thresholds):
        if not math.isfinite(nll) or nll < 0:
            raise ValueError('NLL must be finite and nonnegative')
        if len(thresholds) != 5 or any(x <= 0 for x in thresholds) or sorted(set(thresholds)) != list(thresholds):
            raise ValueError('Five strictly increasing positive thresholds required')
        quality = max(0, 5 - sum(nll >= math.log(x) for x in thresholds))
        self.ease = max(1.3, self.ease + .1 - (5-quality) * (.08 + .02*(5-quality)))
        if quality < 3:
            self.successes, self.interval = 0, 1
        else:
            self.successes += 1
            self.interval = 1 if self.successes == 1 else 6 if self.successes == 2 else math.ceil(self.interval*self.ease)
        self.due = step + self.interval
        self.reviews += 1
        return quality


def due_batch(old, new, states, step, batch_size, old_fraction, rng):
    """Paper old/new due allocation, releasing unused capacity to the other pool.

    If both pools are not due, advance the scheduling clock to the earliest due
    item without spending an optimizer step. This unspecified corner-case is
    reported explicitly in the port metadata.
    """
    if not old and not new:
        raise ValueError('No available stage-visible examples')
    available = old + new
    clock = max(step, min(states[i].due for i in available))
    old_due = [i for i in old if states[i].due <= clock]
    new_due = [i for i in new if states[i].due <= clock]
    selected_old = rng.sample(old_due, min(len(old_due), math.floor(batch_size*old_fraction)))
    selected_new = rng.sample(new_due, min(len(new_due), batch_size-len(selected_old)))
    remaining_old = [i for i in old_due if i not in selected_old]
    selected_old += rng.sample(remaining_old, min(len(remaining_old), batch_size-len(selected_old)-len(selected_new)))
    selected = selected_old + selected_new
    if not selected or len(selected) != len(set(selected)):
        raise AssertionError('Invalid due sampling')
    return selected, clock, len(selected_old)


def balanced_top_rollouts(rollouts, budget, rng):
    """Rank within each past stage; equal task quotas with deterministic remainder.

    The model-generated response is the replay target, including low-score
    rollouts if they lie in the top quota. Gold never silently replaces it.
    Empty responses are excluded because answer-only SFT needs a target.
    """
    groups = {}
    for record in rollouts:
        if record['response'].strip():
            groups.setdefault(int(record['row']['stage']), []).append(record)
    if not groups or budget <= 0:
        return []
    stages = sorted(groups)
    for stage in stages:
        rng.shuffle(groups[stage])
        groups[stage].sort(key=lambda r: r['score'], reverse=True)
    quotas = {s: min(len(groups[s]), budget//len(stages)+(j < budget%len(stages))) for j,s in enumerate(stages)}
    while sum(quotas.values()) < min(budget, sum(map(len, groups.values()))):
        for s in stages:
            if quotas[s] < len(groups[s]) and sum(quotas.values()) < budget:
                quotas[s] += 1
    return [dict(item['row'], ans=item['response'], replay_gold=item['row']['ans'],
                 replay_reward=item['score'], replay_origin='on_policy_generation')
            for s in stages for item in groups[s][:quotas[s]]]


def test_core():
    t = [50.,100.,500.,1000.,5000.]
    s = ReviewState()
    assert s.review(0, 0, t) == 5 and s.interval == 1
    assert s.review(0, 1, t) == 5 and s.interval == 6
    s.review(0, 7, t)
    assert s.interval == 17 and abs(s.ease-2.8) < 1e-12
    assert s.review(math.log(6000), 24, t) == 0 and s.interval == 1 and s.successes == 0
    for nll, q in [(0,5),(math.log(50),4),(math.log(100),3),(math.log(500),2),(math.log(1000),1),(math.log(5000),0)]:
        assert ReviewState().review(nll, 0, t) == q
    rng = random.Random(42)
    states = {i: ReviewState(due=10) for i in range(4)}
    batch, clock, count = due_batch([0,1],[2,3],states,0,2,.5,rng)
    assert clock == 10 and count == 1 and len(batch) == 2
    states[0].due = states[1].due = 100
    batch, _, count = due_batch([0,1],[2,3],states,10,2,.5,rng)
    assert count == 0 and set(batch) == {2,3}
    records = [{'row':{'stage':stage,'ans':'GOLD','sample_id':str(i)},'response':'SAMPLED'+str(i),'score':i}
               for i,stage in enumerate([1,1,2,2])]
    chosen = balanced_top_rollouts(records,2,rng)
    assert {r['sample_id'] for r in chosen} == {'1','3'}
    assert all(r['ans'].startswith('SAMPLED') and r['replay_gold']=='GOLD' for r in chosen)
    print('replay_recent_core: SM-2 quality/boundaries/reset, due allocation/empty-pool, task quotas and generated-target preservation passed')


if __name__ == '__main__':
    test_core()
