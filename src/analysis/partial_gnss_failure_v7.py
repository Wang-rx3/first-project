"""Recover precise safety-stop evidence from a bounded identical prefix."""
from __future__ import annotations

from ..experiments import partial_gnss_baseline_v7 as v7
import numpy as np


def main():
    spec=v7.old.read_json(v7.CONFIG)
    case=spec["cases"][1]
    output=v7.BASE/"safety_prefix_s20267102"
    output.mkdir(exist_ok=False)
    env,obs=v7.prepare(case,spec)
    rows=[]
    with v7.old._forbid_training_runtime():
        for step in range(1,121):
            masked=v7.old.v4.mask_observations(obs,step)
            comm=v7.old.v3.contract().communication_actions(masked,{u:"none" for u in v7.UIDS})
            actions,nav=v7.compose_fixed_navigation_actions(masked,comm)
            previous_p={u:env.states[u].position_m.copy() for u in v7.UIDS}
            previous_v={u:env.states[u].velocity_mps.copy() for u in v7.UIDS}
            obs,_,terminated,truncated,info=env.step(actions)
            rows.append(v7.old.normalize({"step":step,"collision":info["collision"],
                "acceleration_norms":{u:np.linalg.norm((env.states[u].velocity_mps-previous_v[u])/.2) for u in v7.UIDS},
                "boundary_count":info["boundary_projection_count"],"safety_count":info["safety_projection_count"],
                "states":{u:env.states[u].position_m for u in v7.UIDS},
                "estimates":{u:env.ekfs[u].state for u in v7.UIDS},
                "info_action":info.get("action_info"),"actions":actions,"navigation":nav,
                "minimum_separation":min(r["distance_m"] for r in v7.old.joint_closest_approach(previous_p,{u:env.states[u].position_m for u in v7.UIDS}))}))
            try:
                v7.check_safety(env,info,previous_p,previous_v)
            except RuntimeError as exc:
                v7.old.write_json(output/"failure.json",{"case":case,"steps":step,"reason":str(exc),"rows":rows,"diagnostic_reproduction_only":True})
                print(v7.old.encode(rows[-1]))
                return
    raise RuntimeError("safety failure not reproduced")


if __name__=="__main__":main()
