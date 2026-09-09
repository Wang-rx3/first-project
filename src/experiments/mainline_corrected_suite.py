"""Sequential bounded suite; never launches parallel experiments or overwrites.

Usage: python -m src.experiments.mainline_corrected_suite
Already complete stages are accepted only at the exact current configuration
signature. Failed/nonempty stages require an explicit archived path decision.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from .mainline_baseline_comparison import MainlineBaselineComparisonConfig, run_mainline_baseline_comparison
from .mainline_corrected_stages import configuration_signature, finalize_development


def run_bounded_suite():
    for arm in ("reference","proxy-aligned","rollout4"):
        for stage in ("smoke","development"):
            config=MainlineBaselineComparisonConfig(protocol="corrected-v2",stage=stage,experiment_arm=arm,figure_profile="compact-mainline")
            path=config.resolved_logs_dir/"audit_report.json"
            if path.exists():
                report=json.loads(path.read_text(encoding="utf-8"))
                if report.get("audit_ok") and report.get("configuration_signature")==configuration_signature(config):
                    print(f"[corrected-v2 suite] reuse completed {arm}/{stage}",flush=True)
                    continue
                raise RuntimeError(f"nonempty, incomplete or different-version stage: {path}")
            print(f"[corrected-v2 suite] start {arm}/{stage}",flush=True)
            result=run_mainline_baseline_comparison(config)
            if not result.get("audit_ok"):
                raise RuntimeError(f"structure failed: {arm}/{stage}")
    result=finalize_development()
    print(json.dumps({"finished":True,"selected_arm":result["selected_arm"],"optimization_success":result["optimization_success"]}),flush=True)


def main():
    root=Path("logs/preliminary/mainline_baseline_comparison_v2_seed42")
    root.mkdir(parents=True,exist_ok=True)
    lock=root/"suite.lock"
    # Fail closed if another runner is active or an interrupted run left a lock.
    # Never terminate a process or remove a stale lock automatically.
    descriptor=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    os.write(descriptor,str(os.getpid()).encode()); os.close(descriptor)
    folder=root/"suite_runs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True,exist_ok=False)
    original_out,original_error=sys.stdout,sys.stderr
    class Tee:
        def __init__(self,original,stream):self.original,self.stream=original,stream
        def write(self,value):
            self.stream.write(value); self.stream.flush()
            return self.original.write(value)
        def flush(self):self.stream.flush();self.original.flush()
    try:
        with (folder/"console.log").open("x",encoding="utf-8") as stream:
            sys.stdout=Tee(original_out,stream); sys.stderr=Tee(original_error,stream)
            try:
                run_bounded_suite()
            except Exception:
                import traceback
                traceback.print_exc()
                raise
    finally:
        sys.stdout,sys.stderr=original_out,original_error
        lock.unlink()


if __name__=="__main__":
    main()
