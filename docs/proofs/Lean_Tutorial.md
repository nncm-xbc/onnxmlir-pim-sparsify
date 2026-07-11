# Lean 4 Tutorial for Stability Proof

## Getting Started with Lean

### Installation

```bash
# Install Lean 4 using elan (Lean version manager)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Restart your shell, then create a new Lean project
lake new MyProject
cd MyProject

# Or work directly with .lean files (less recommended)
```

### VS Code Setup

1. Install VS Code
2. Install the "Lean 4" extension
3. Open your `.lean` file
4. The extension will download dependencies automatically

### Basic Lean Concepts

#### 1. Everything is a Type

```lean
#check 5          -- 5 : ℕ (natural number)
#check 3.14       -- 3.14 : Float
#check "hello"    -- "hello" : String
#check ℕ          -- ℕ : Type (ℕ itself is a type)
#check Type       -- Type : Type 1 (types have types!)
```

#### 2. Functions

```lean
-- Function definition
def square (n : ℕ) : ℕ := n * n

-- Apply function
#eval square 5   -- 25

-- Anonymous function (lambda)
#check fun x => x + 1    -- ℕ → ℕ

-- Function with multiple arguments
def add (a b : ℕ) : ℕ := a + b
```

#### 3. Propositions and Proofs

```lean
-- A proposition is a type in Prop
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- A proof is a term of a proposition type
theorem zero_is_even : even 0 := by
  use 0           -- Provide k = 0
  rfl             -- Reflexivity: 0 = 2 * 0

-- Propositions are types, proofs are terms!
```

## Understanding the Stability Proof Structure

### Step-by-Step Walkthrough

#### Step 1: Parameter Space

```lean
abbrev ParameterSpace (N : ℕ) := Fin N → ℝ
```

**What this means:**
- `ParameterSpace N` is a vector of N real numbers
- `Fin N` is the type {0, 1, ..., N-1}
- `Fin N → ℝ` means "function from {0,...,N-1} to reals"
- This is how we represent vectors in Lean

**Example:**
```lean
-- A network with 3 parameters
def w : ParameterSpace 3 := fun i =>
  match i with
  | 0 => 1.5    -- w[0] = 1.5
  | 1 => -0.3   -- w[1] = -0.3
  | 2 => 2.1    -- w[2] = 2.1
```

#### Step 2: Function Space

```lean
abbrev FunctionSpace (InputDim OutputDim : ℕ) :=
  InputSpace InputDim → OutputSpace OutputDim
```

**What this means:**
- A function in function space takes inputs and produces outputs
- `InputSpace InputDim` is a vector of input dimension
- `OutputSpace OutputDim` is a vector of output dimension
- This represents "what the network computes"

#### Step 3: Distance Metrics

```lean
noncomputable def parameterDistance (N : ℕ) (w₁ w₂ : ParameterSpace N) : ℝ :=
  Real.sqrt (Finset.sum Finset.univ (fun i => (w₁ i - w₂ i)^2))
```

**Breaking it down:**
- `noncomputable` = can't directly compute, but logically valid
- `(w₁ w₂ : ParameterSpace N)` = two parameters as input
- `Finset.sum Finset.univ` = sum over all indices
- `fun i => (w₁ i - w₂ i)^2` = for each index i, compute squared difference
- This is Euclidean distance: √(Σᵢ (w₁ᵢ - w₂ᵢ)²)

#### Step 4: Axioms vs Definitions

```lean
axiom functionDistance
  {InputDim OutputDim : ℕ}
  (f₁ f₂ : FunctionSpace InputDim OutputDim) : ℝ
```

**Axioms:**
- `axiom` declares something we assume without proof
- Used for:
  - Abstract interfaces (define later)
  - Mathematical assumptions
  - Placeholders during development
- Eventually, you replace axioms with actual `def` or `theorem`

**Why use axioms here?**
- `functionDistance` needs probability measures (complex!)
- We focus on the proof structure first
- Later, we can instantiate with concrete definitions

#### Step 5: The Main Theorem

```lean
theorem sparsification_stability
  {N InputDim OutputDim : ℕ}
  (K : ℕ)
  (ε : ℝ)
  (hε : ε > 0)
  (w₁⁰ w₂⁰ : ParameterSpace N)
  (h_close : functionDistance (ℱ(w₁⁰)) (ℱ(w₂⁰)) < ε)
  :
  ∃ C : ℝ, C > 0 ∧
    functionDistance (ℱ(S[K, w₁⁰](w₁⁰))) (ℱ(S[K, w₂⁰](w₂⁰))) ≤ C * ε
```

**Reading the signature:**
- Everything before `:` is input (assumptions)
- `{N InputDim OutputDim : ℕ}` = implicit arguments (Lean infers them)
- `(K : ℕ)` = explicit argument (we provide it)
- `(hε : ε > 0)` = hypothesis (assumption that ε > 0)
- `(h_close : ...)` = another hypothesis (closeness assumption)
- After `:` is what we're proving
- `∃ C : ℝ, ...` = "there exists C such that..."

## Essential Lean Tactics

### Basic Tactics

```lean
-- 1. intro: Move ∀ into context
theorem example1 : ∀ n : ℕ, n + 0 = n := by
  intro n        -- Now n is in context
  rfl            -- Reflexivity

-- 2. apply: Apply a theorem/function
theorem example2 (h : P → Q) (hp : P) : Q := by
  apply h        -- Need to prove P
  exact hp       -- Provide P

-- 3. exact: Provide an exact proof term
theorem example3 : 5 = 5 := by
  exact rfl      -- rfl is proof of reflexivity

-- 4. sorry: Placeholder (admits anything)
theorem example4 : False := by
  sorry          -- "Trust me, I'll prove this later"
```

### Working with ∃ (Exists)

```lean
-- Proving ∃: use `use` tactic
theorem exists_even_number : ∃ n : ℕ, n % 2 = 0 := by
  use 4          -- Provide witness
  rfl            -- Prove 4 % 2 = 0

-- Using ∃: extract with `obtain`
theorem use_exists (h : ∃ n : ℕ, n > 5) : True := by
  obtain ⟨n, hn⟩ := h    -- Extract n and proof hn : n > 5
  trivial                 -- Prove True
```

### Working with ∧ (And)

```lean
-- Proving ∧: use `constructor`
theorem prove_and : 5 > 0 ∧ 5 < 10 := by
  constructor
  · norm_num     -- Prove 5 > 0
  · norm_num     -- Prove 5 < 10

-- Using ∧: extract with `.1` and `.2`
theorem use_and (h : P ∧ Q) : P := by
  exact h.1      -- or: h.left
```

### Induction

```lean
-- Induction on natural numbers
theorem sum_formula : ∀ n : ℕ, 2 * (sum k=0 to n) = n * (n + 1) := by
  intro n
  induction n with
  | zero =>
    -- Base case: n = 0
    simp
  | succ n ih =>
    -- Inductive case: n = k + 1
    -- ih is inductive hypothesis
    sorry
```

## How to Develop Your Proof

### Workflow

1. **Start with structure:**
   ```lean
   theorem my_theorem (assumptions) : conclusion := by
     sorry
   ```

2. **Check types:**
   ```lean
   theorem my_theorem : ... := by
     -- What do I need to prove?
     -- Lean shows in the "Info View" panel
   ```

3. **Break down the proof:**
   ```lean
   theorem my_theorem : ... := by
     intro x         -- Introduce variables
     obtain ⟨y, hy⟩ := some_lemma  -- Use lemmas
     apply another_lemma            -- Apply results
     · sorry         -- Goal 1
     · sorry         -- Goal 2
   ```

4. **Fill in the sorries one by one**

### Interactive Development

```lean
theorem example : P → Q := by
  intro hp
  -- Hover over `intro` in VS Code: see what's in context
  -- The goal view shows what remains to prove
  ?_           -- Hole: Lean shows expected type
```

## Exercises for Practice

### Exercise 1: Simple Function

```lean
-- Define a function that doubles a number
def double (n : ℕ) : ℕ := 2n -- Is this correct ? 

-- Prove a property
theorem double_zero : double 0 = 0 := by
  sorry
```

### Exercise 2: Lipschitz Property

```lean
-- Prove that doubling is Lipschitz with L = 2
theorem double_lipschitz (a b : ℕ) :
  |double a - double b| ≤ 2 * |a - b| := by
  sorry
```

### Exercise 3: Exists Proof

```lean
-- Prove that there exists a stable constant
theorem exists_stability_constant
  (f : ℝ → ℝ)
  (h_lip : ∀ x y, |f x - f y| ≤ 2 * |x - y|)
  :
  ∃ C : ℝ, C > 0 ∧ ∀ x y, |f x - f y| ≤ C * |x - y| := by
  -- Hint: use C = 2
  sorry
```

## Resources

### Official Documentation
- [Lean 4 Manual](https://leanprover.github.io/lean4/doc/)
- [Theorem Proving in Lean 4](https://leanprover.github.io/theorem_proving_in_lean4/)
- [Mathematics in Lean](https://leanprover-community.github.io/mathematics_in_lean/)

### Learning Materials
- [Natural Number Game](https://adam.math.hhu.de/#/g/leanprover-community/NNG4) - Interactive tutorial
- [Lean Zulip Chat](https://leanprover.zulipchat.com/) - Active community
- [Mathlib Docs](https://leanprover-community.github.io/mathlib4_docs/) - Standard library

### Tips
1. **Start small**: Prove simple lemmas first
2. **Use #check liberally**: Understand types
3. **Read the goal view**: It tells you what to prove
4. **Ask for help**: Lean community is very friendly
5. **Iterate**: Start with sorry, refine gradually

## Next Steps for Your Stability Proof

1. **Familiarize with Mathlib:**
   - Look at `Mathlib.Analysis.Normed.Group.Basic`
   - Study existing Lipschitz proofs
   - See how metric spaces are formalized

2. **Replace one axiom:**
   - Pick `parameterDistance` properties
   - Prove they form a metric (triangle inequality, etc.)

3. **Prove a simple lemma:**
   - Start with `adjustment_lipschitz`
   - Use gradient descent theory from Mathlib

4. **Build incrementally:**
   - One lemma at a time
   - Test each piece
   - Gradually complete the proof

5. **Connect to your Python code:**
   - The formal proof validates your implementation
   - Use experiments to guide proof assumptions
   - If proof gets stuck, check code for insights

## Common Errors and Solutions

### Error: "unknown identifier"
```lean
theorem my_theorem : n + 0 = n := by  -- ERROR: n not declared
  rfl

-- Fix: introduce n first
theorem my_theorem : ∀ n : ℕ, n + 0 = n := by
  intro n
  rfl
```

### Error: "type mismatch"
```lean
def f (n : ℕ) : ℕ := n + 1.5  -- ERROR: 1.5 is not ℕ

-- Fix: use correct type
def f (n : ℝ) : ℝ := n + 1.5
```

### Error: "failed to synthesize instance"
```lean
theorem test (x y : MyType) : x ≤ y := sorry  -- ERROR: no ≤ on MyType

-- Fix: add appropriate typeclass instances
```

Good luck with your formalization! Start by reading through the stability.lean file,
then try modifying small parts. The Lean compiler will guide you with helpful error messages.
