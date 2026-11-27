# Backend

Python code for the swarm modules, the fire model and the API.

| Folder | What it holds | Jira |
|---|---|---|
| `optimizer/` | Response optimisation: plan decode/repair, fitness, PSO, GWO, baselines, `run()`, wind-change re-planning | GC-16, GC-25, GC-26, GC-35 |
| `tests/` | pytest suite (uses a stand-in fire model until the real one lands) | GC-30 |

Coming next: `firemodel/` (GC-13), `risk/` (GC-21), `routing/` (GC-27), `api/` (GC-12, GC-33), `experiments/` (GC-38).

```bash
pip install numpy scipy pytest
python -m pytest -q
```

See the main [README](../README.md) for the architecture and the interfaces between modules.
