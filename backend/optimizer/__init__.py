"""Response optimisation for the Margalla Hills wildfire project.

Modules
-------
response   decode / repair a plan vector, build the suppression schedule (GC-16)
fitness    F(x) = mean burned cells over common seeds + lambda * travel + penalty (GC-25)
pso        Particle Swarm Optimization, written from scratch (GC-25)
gwo        Grey Wolf Optimizer, written from scratch (GC-26)
baselines  no response, random placement, greedy downwind arc (GC-26)
replan     re-plan units not yet arrived after a wind change, warm-start PSO (GC-35)
runner     run(method, scenario, ...) - one interface for the experiments (GC-26)
"""
