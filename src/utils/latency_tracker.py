import time


class LatencyTracker:
    def __init__(self):
        self.durations = {}
        self._starts = {}

    def start(self, stage: str) -> None:
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        if stage not in self._starts:
            raise ValueError(f'Stage was not started: {stage}')

        elapsed_ms = (time.perf_counter() - self._starts.pop(stage)) * 1000.0
        self.durations[stage] = elapsed_ms
        return elapsed_ms

    def summary(self) -> dict:
        summary = {
            'preprocessing': self.durations.get('preprocessing', 0.0),
            'rf_inference': self.durations.get('rf_inference', 0.0),
            'xgb_inference': self.durations.get('xgb_inference', 0.0),
            'if_inference': self.durations.get('if_inference', 0.0),
        }
        summary['total'] = sum(summary.values())
        return summary

    def reset(self) -> None:
        self.durations.clear()
        self._starts.clear()