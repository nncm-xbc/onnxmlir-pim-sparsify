# Documentation

Post-training **manifold-distance sparsification** for MLPs: greedily remove the weight
whose deletion least perturbs the network's function (measured as a Monte-Carlo distance
in function space over a compact input domain), then a gradient **adjust** step recovers
behaviour under the fixed sparsity pattern. The project spans the method and its theory
(including Lean stability proofs), a JAX/C++ experiment framework, and an ARM/PIM compiler
backend for the resulting sparse networks — the subject of an MSc thesis at Politecnico di Milano.

## Contents

| Doc | Description |
| --- | --- |
| [algorithm.md](algorithm.md) | The sparsification method & mathematics — distance `d_W`, pruning + adjust steps, MC approximation, complexity |
| [roadmap.md](roadmap.md) | Thesis roadmap — chapters, proof strategy, reading schedule, timeline |
| [todo.md](todo.md) | Open work: experiment backlog, code-review findings, viz/theory/writing tasks |
| [handover.md](handover.md) | Handover notes for the `fable-profiling` performance branch |
| [design/](design/) | Dated design specs (datasets/architectures, parallel & neuron pruning, magnitude, Kwon, Lazarevich, OBD, OBS) |
| [plans/](plans/) | Dated implementation plans (benchmark framework, parallel pruning, neuron pruning, …) |
| [notebooks/](notebooks/) | Exploratory Jupyter notebooks (sparsification examples, compiler memalloc, transfer-cost optimization) |
| [proofs/](proofs/) | Lean stability proofs (Lipschitz bound, stupidity-point existence, weight-vs-neuron) + notes |
