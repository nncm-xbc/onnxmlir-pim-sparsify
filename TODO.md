# Master Thesis TODO List

## Foundational Proofs

### Sparsification Stability Proof

- [ ] Read essential background (Nocedal & Wright Ch. 2-3, Lee Ch. 1-3)
- [ ] Study NTK paper (Jacot et al.) for function space perspective
- [ ] Study mode connectivity papers (Garipov et al., Draxler et al.)
- [ ] Review lottery ticket hypothesis papers (Frankle & Carbin)
- [ ] Define precise function space metric $d_{\mathscr{F}}$ rigorously
- [ ] Characterize smoothness properties of network realization map $\mathcal{F}$
- [ ] Prove (or bound) Lipschitz continuity of adjustment step
- [ ] Analyze pruning step: continuity or stability under perturbations
- [ ] Define critical parameter sets formally
- [ ] Prove reparametrization invariance of critical sets
- [ ] Develop Approach 1: Lipschitz continuity proof
- [ ] Develop Approach 2: Convergence to critical sparsity pattern
- [ ] Choose final proof approach (or hybrid)
- [ ] Write complete proof with all lemmas and theorems
- [ ] Construct counterexamples for edge cases
- [ ] Validate proof with numerical experiments

### Proof of the Stupidity Point (Core Theoretical Contribution)

- [ ] Read essential background on critical points and loss landscape geometry
- [ ] Study mode connectivity and flat minima literature (Garipov et al., Draxler et al.)
- [ ] Review information-theoretic compression bounds (Shannon, Kolmogorov complexity)
- [ ] Define "stupidity point" formally: the critical sparsity level $s^*$ beyond which the network cannot maintain its function
  - [ ] Characterize in terms of parameter space geometry
  - [ ] Relate to rank deficiency or expressivity collapse
  - [ ] Connect to critical parameter sets
- [ ] Develop existence proof: show $s^*$ is well-defined (non-trivial and finite)
- [ ] Develop uniqueness or multiplicity analysis: is $s^*$ unique or architecture-dependent?
- [ ] Derive analytical bounds or necessary conditions for $s^*$
  - [ ] Lower bound: minimum expressivity requirements
  - [ ] Upper bound: connectivity / lottery ticket argument
- [ ] Prove phase transition behavior near $s^*$ (catastrophic loss of function)
- [ ] Analyze relationship between $s^*$ and training data, architecture depth/width
- [ ] Develop semi-analytical characterization (closed-form approximation where possible)
- [ ] Validate analytically derived bounds with numerical experiments
- [ ] Construct counterexamples for edge cases (degenerate architectures, etc.)
- [ ] Write complete proof with all lemmas and theorems

### Weight Pruning vs Neuron Pruning

- [ ] Formalize neuron pruning in the manifold framework
- [ ] Compare parameter space neighborhoods for weight vs neuron pruning
- [ ] Prove relationship between weight and neuron sparsity patterns
- [ ] Analyze which approach preserves function space distance better
- [ ] Derive theoretical complexity comparison

## Introduction (2 weeks allocated)

### Literature Review

- [ ] Research current state of inference pruning methods (novelty factor)
- [ ] Identify and analyze similarity research in post-training sparsification
- [ ] Review magnitude pruning (Han et al. 2015)
- [ ] Review SNIP and sensitivity-based methods (Lee et al. 2019)
- [ ] Review lottery ticket and related work (Frankle et al.)
- [ ] Review structured vs unstructured pruning literature
- [ ] Compare manifold-based approach to existing methods
- [ ] Compile full bibliography for introduction section

### Writing

- [ ] Write motivation section (why post-training sparsification matters)
- [ ] Write related work section
- [ ] Write contribution summary
- [ ] Write thesis outline
- [ ] Complete introduction chapter

## Algorithm Chapter

### Method Documentation

- [ ] Write introduction to the manifold-based sparsification method
- [ ] Formalize parameter space $\mathscr{W}$ and function space $\mathscr{F}$
- [ ] Define local semantic distance $d_{\mathscr{W}}$ with full justification
- [ ] Define neighborhood structure and sparsity graph
- [ ] Document the sparsification algorithm with detailed pseudocode
- [ ] Explain greedy selection strategy
- [ ] Explain gradient-based adjustment phase
- [ ] Analyze computational complexity

### Theoretical Proofs

- [ ] Prove smoothness of the realization function with respect to network settings
  - [ ] Formalize smoothness assumptions
  - [ ] Prove differentiability of forward pass
  - [ ] Bound Lipschitz constants in terms of architecture
  - [ ] Derive explicit formulas for gradients
- [ ] Develop semi-analytical proof of the stupidity point (core theoretical contribution)
  - [ ] Define "stupidity point" formally
  - [ ] Characterize when further sparsification causes catastrophic loss
  - [ ] Derive analytical bounds or necessary conditions
  - [ ] Prove existence/uniqueness properties
  - [ ] Connect to critical parameter sets
- [ ] Prove monotonicity of sparsity (NZ decreases)
- [ ] Prove greedy optimality at each step
- [ ] Derive convergence rate (if possible)

### Writing

- [ ] Write mathematical preliminaries section
- [ ] Write algorithm description section
- [ ] Write theoretical analysis section
- [ ] Include all proofs with clear lemma/theorem structure
- [ ] Add illustrative diagrams (parameter manifold, distance visualization)
- [ ] Write complete algorithm chapter

## Experiments Chapter

### Gaussian Data (Structureless)

- [ ] Design Gaussian data generation (dimensionality, distribution)
- [ ] Generate synthetic classification/regression tasks
- [ ] Train dense networks on Gaussian data
- [ ] Run sparsification algorithm on Gaussian-trained networks
- [ ] Measure stability: vary initialization, measure function space distance
- [ ] Analyze sparsity patterns: random or structured?
- [ ] Compare to magnitude pruning baseline
- [ ] Document findings

### MNIST Dataset

- [ ] Run sparsification experiments on existing MNIST networks
- [ ] Vary network architectures (width, depth)
- [ ] Measure accuracy vs sparsity tradeoff
- [ ] Test stability: multiple random initializations
- [ ] Visualize critical parameters (which weights survive?)
- [ ] Compare to lottery ticket hypothesis results
- [ ] Analyze convergence speed of adjustment phase
- [ ] Document findings

### Additional Datasets

- [ ] Define and select additional experimental datasets beyond MNIST
  - [ ] Consider: CIFAR-10, Fashion-MNIST, SVHN
  - [ ] Consider: Synthetic data with known structure
  - [ ] Consider: Small-scale ImageNet subset
- [ ] Adapt network architectures for new datasets
- [ ] Run full sparsification pipeline
- [ ] Analyze dataset-specific patterns
- [ ] Compare results across all datasets

### Comparative Analysis

- [ ] Compare manifold-based vs magnitude pruning
- [ ] Compare manifold-based vs SNIP/sensitivity methods
- [ ] Measure computational cost (time, memory)
- [ ] Analyze when manifold approach works best
- [ ] Identify failure modes or limitations

### Writing

- [ ] Write experimental setup section
- [ ] Write results section with tables and figures
- [ ] Write analysis and discussion section
- [ ] Include convergence plots, sparsity curves, accuracy graphs
- [ ] Write complete experiments chapter

## Hardware Considerations

### PIM Analysis

- [ ] Research PIM (Processing-In-Memory) architectures
- [ ] Formalize weight pruning cost model for PIM
- [ ] Formalize neuron pruning cost model for PIM
- [ ] Analyze memory access patterns for each approach
- [ ] Measure energy consumption (simulated or analytical)
- [ ] Determine optimal pruning strategy for PIM architectures
- [ ] Compare to traditional von Neumann architectures
- [ ] Benchmark and document PIM-specific performance characteristics

### Compiler Extensions

- [ ] Review current ARM compiler implementation
- [ ] Identify bottlenecks in generated code
- [ ] Propose optimizations for PIM targets
- [ ] Implement or sketch PIM-specific code generation

### Software Engineering

- [ ] Decide software engineering approach: standalone module vs full onnx-mlir integration
  - [ ] Evaluate pros/cons of standalone module
  - [ ] Evaluate pros/cons of onnx-mlir integration
  - [ ] Consult with advisor on strategic direction
  - [ ] Document decision rationale
- [ ] Design software architecture
- [ ] Implement chosen software engineering architecture for hardware integration
- [ ] Write tests and validation suite
- [ ] Document API and usage

### Writing

- [ ] Write PIM architecture background
- [ ] Write pruning strategy analysis for PIM
- [ ] Write compiler considerations section
- [ ] Write software engineering section
- [ ] Write complete hardware considerations chapter

## Finalization

- [ ] Write conclusion section
  - [ ] Summarize main contributions
  - [ ] Discuss limitations
  - [ ] Contextualize impact
- [ ] Write future work section
  - [ ] Theoretical extensions
  - [ ] Experimental directions
  - [ ] Hardware/systems work
- [ ] Complete final thesis review and revisions
  - [ ] Check all citations and bibliography
  - [ ] Verify all equations and proofs
  - [ ] Proofread for clarity and grammar
  - [ ] Ensure consistent notation throughout
  - [ ] Check figure quality and captions
  - [ ] Format according to university guidelines
- [ ] Prepare thesis defense presentation
  - [ ] Create slides (30-45 minutes)
  - [ ] Prepare demo/visualizations
  - [ ] Practice presentation
  - [ ] Prepare for Q&A

---

## Reading Schedule (8-12 weeks)

### Phase 1: Foundations (Weeks 1-3)
- [ ] Nocedal & Wright - Numerical Optimization (Ch. 2-3)
- [ ] Tao - Analysis I (Ch. 1-4: Metric spaces, continuity)
- [ ] Velleman or Houston - Proof techniques refresher

### Phase 2: Geometric Perspective (Weeks 4-6)
- [ ] Lee - Introduction to Smooth Manifolds (Ch. 1-3)
- [ ] Do Carmo - Riemannian Geometry (Ch. 1-2)
- [ ] Jacot et al. - Neural Tangent Kernel paper

### Phase 3: Specialized Topics (Weeks 7-10)
- [ ] Bertsekas - Nonlinear Programming (Ch. 1-3)
- [ ] Krantz & Parks - Implicit Function Theorem
- [ ] Frankle & Carbin - Lottery Ticket Hypothesis
- [ ] Garipov et al. - Mode Connectivity
- [ ] Liu et al. - Rethinking Network Pruning

### Phase 4: Advanced Topics (Weeks 11-12)
- [ ] Bach - Submodular Functions (if applicable)
- [ ] Additional papers based on proof approach chosen

---

## Notes

### Core Theoretical Contribution
**The semi-analytical proof of the stupidity point** is the main theoretical contribution of this thesis. This characterizes the critical sparsity level beyond which the network cannot maintain its function.

### Proof Strategy
**Hybrid approach recommended**: Combine Lipschitz continuity (Approach 1) with convergence to critical sparsity pattern (Approach 2).

### Timeline
- Introduction: 2 weeks
- Reading & Proof Development: 8-12 weeks (parallel)
- Algorithm Chapter: 3-4 weeks
- Experiments: 4-6 weeks (can overlap with proof development)
- Hardware: 2-3 weeks
- Writing & Revision: 3-4 weeks
- Defense Preparation: 1-2 weeks

**Total estimated**: 6-8 months

### Open Questions
- Which additional datasets to use beyond MNIST? → Consider CIFAR-10, Fashion-MNIST
- Standalone module or full onnx-mlir integration? → Depends on industrial partner requirements
- Can we prove submodularity of the sparsification objective?
- What is the precise relationship between critical parameters and lottery tickets?

### Resources Created
- [Sparsification Stability Approaches](Docs/Proofs/Sparsification_Stability_Approaches.md) - Four detailed proof approaches
- [Reading List](Docs/Proofs/Reading_List.md) - Comprehensive bibliography with reading order

### Key Concepts to Master
1. Riemannian metrics on parameter manifolds
2. Lipschitz continuity in function spaces
3. Implicit function theorem for optimization
4. Greedy algorithm approximation theory
5. Neural tangent kernel theory
6. Mode connectivity and loss landscape geometry
