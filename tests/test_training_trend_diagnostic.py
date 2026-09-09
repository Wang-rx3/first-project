from types import SimpleNamespace
import pytest
import torch
from src.experiments import training_trend_diagnostic as diagnostic


def row(**changes):
    return dict(valid=True, actor_nonzero=True, critic_squared_fraction=.9,
                actor_retention_ratio=3., **changes)


def test_gate_requires_twenty_and_joint_criteria():
    assert not diagnostic.gate([row() for _ in range(19)])['eligible']
    assert diagnostic.gate([row() for _ in range(20)])['eligible']
    rows = [row() for _ in range(20)]
    for r in rows[:11]:
        r['actor_retention_ratio'] = 1.5
    assert not diagnostic.gate(rows)['eligible']
    rows[0]['actor_retention_ratio'] = 3.
    assert diagnostic.gate(rows)['eligible']
    rows[0]['valid'] = False
    assert not diagnostic.gate(rows)['eligible']


def test_gradient_probe_is_read_only_and_zero_cosine_na(monkeypatch):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.measurement_head = torch.nn.Linear(1, 1, bias=False)
            self.critic = torch.nn.Linear(1, 1, bias=False)
    policy = Policy()
    trainer = SimpleNamespace(policy=policy, config=SimpleNamespace(max_grad_norm=.5))
    def losses(t, records):
        return {'ppo': t.policy.measurement_head.weight.sum(),
                'teacher': t.policy.measurement_head.weight.sum()*0,
                'entropy': torch.tensor(0.), 'communication': torch.tensor(0.),
                'critic': 10*t.policy.critic.weight.sum()}
    monkeypatch.setattr(diagnostic, 'components', losses)
    before = diagnostic.policy_vector(trainer)
    result = diagnostic.gradient_probe(trainer, [1])
    assert result['ppo_teacher_cosine'] == 'NA'
    assert result['critic_squared_fraction'] == pytest.approx(100/101)
    assert result['actor_retention_ratio'] > 2
    assert torch.equal(before, diagnostic.policy_vector(trainer))
    assert all(p.grad is None for p in policy.parameters())


def test_new_directory_and_seed_boundaries(tmp_path):
    diagnostic.fresh(tmp_path/'out')
    (tmp_path/'out'/'evidence').write_text('keep')
    with pytest.raises(FileExistsError): diagnostic.fresh(tmp_path/'out')
    with pytest.raises(ValueError): diagnostic.run('frozen-audit', 2026)


def test_failed_gate_does_not_start_smoke(tmp_path,monkeypatch):
    import json
    root=tmp_path/'frozen-audit'
    root.mkdir()
    report={'audit_ok':True,'gate':diagnostic.gate([row() for _ in range(19)])}
    (root/'audit_report.json').write_text(json.dumps(report))
    monkeypatch.setattr(diagnostic,'LOGS',tmp_path)
    result=diagnostic.run('smoke',42)
    assert result['status']=='not_run_gate_failed'
    assert sorted(p.name for p in root.iterdir())==['audit_report.json']


def test_source_guard_blocks_test_and_source_writes(tmp_path):
    root=tmp_path/'source'; root.mkdir()
    (root/'test').mkdir()
    (root/'test'/'labels.csv').write_text('sealed')
    (root/'checkpoint').write_text('frozen')
    with diagnostic.SourceGuard([root],[tmp_path/'output']) as guard:
        assert (root/'checkpoint').read_text()=='frozen'
        with pytest.raises(PermissionError): (root/'test'/'labels.csv').read_text()
        with pytest.raises(PermissionError): (root/'checkpoint').write_text('changed')
        assert all(r['unchanged'] for r in guard.verify())
    assert guard.test_attempts==1


def test_probe_mask_and_return_checks():
    record={'return':1.,'advantage':.2,'sample':SimpleNamespace(measurement_index=3),
            'observation':{'action_mask':[0,1,0,1]}}
    assert diagnostic.batch_metadata([record])['legal_measurement_options_mean']==2
    record['sample'].measurement_index=0
    with pytest.raises(ValueError): diagnostic.batch_metadata([record])
    record['sample'].measurement_index=3
    record['return']=float('nan')
    with pytest.raises(ValueError): diagnostic.batch_metadata([record])
