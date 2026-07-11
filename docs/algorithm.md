# Mathematics of the sparsification

## Manifold assumption on the space of parameters (local distance between sets of parameters)

Let $\mathscr W$ the space of possible parameters. <br>
Let $\mathcal F_f: \mathscr W \rightarrow \mathscr F$ the <b>network realizing function</b>, which is the map that
converts parameters in $\mathscr W$ into actual functions, belonging to the functional space $\mathscr F$;
$f$ here is defined as the <b>activation function</b>, namely the <b>ReLU</b> function<br>
It is easy to show that 

$$
\exists N \in \mathbb N_+: \mathscr W \sim \mathbb R^N
$$

given the simple argument, that our representation of parameters is fundamentally based on floating point numbers.
Given that the maps $\varphi$ represent the isomorphism between $\mathscr W$ and $\mathbb R^N$ we can hence consider the 
metric space $(\mathscr W, (w,w') \mapsto || \varphi(w) - \varphi(w') ||_2 )$, where $||\cdot||_2$ is the usual euclidean norm
on the euclidean space $\mathbb R^N$ which is used in this case to induce a distance.
We would like to define, also, a suitable distance that considers also the information regarding the realized neural network, through 
the map $\mathcal F$.In fact, if we were able to produce such distance, we would be able to characterize --- in a quantitative manner ---
the difference between two networks in terms of expression.
The problem lies in the fact, in principle, two networks could have very different realization and perform realtively similarly, due to
universal approximation theorem. This could imply , in principle, that we have zero (or almost zero,in practice) distance (according to this new notion)
between different networks, violating the formal definition of distance.
Our approach therefore, adds a further constraint; if two parameters $w$ and $w'$ are close to each other (according to the distance 
inherited by the euclidean space for which exists an isomorphism) then it is possible to define a new distance, which is given by the difference 
in expression (doing so we mitigate the problem of zero-loss classifiers having zero distance).

We define hence 

$$
d_{\mathscr W}(w,w') =  \mathbb E_{{\bf x} \sim \mathcal U(\Omega)} [ || \mathcal F(w)({\bf x}) - \mathcal F(w')({\bf x}) ||^2 ]
$$

Where $\Omega$ is the compact where data distribution appears (for example, in the case of the MNIST dataset could be $[0,255]^{28*28}$).
Morally speaking, we are taking the distance between the represented function as a (local)  distance. Mathematically speaking, this is possible
under the assumption that the map $\mathcal F : w \mapsto \mathcal F(w)$ is some how differentiable with respect to $w$ 
(small distances in the euclidean space correspond to small distances in the parameter space, and the variety is differntiable almost everywhere)
.

## Local distance minimization
Let $w,w' \in \mathscr W$ and $\theta,\theta'$ their image through the map $\varphi$ such that 

$$
	|| \varphi(w) - \varphi(w') ||_2 \textrm{ is "small"}  
$$

Then we can apply the assumption and we have that

$$
d_{\mathscr W}(w,w') =  \mathbb E_{{\bf x} \sim \mathcal U(\Omega)} [ || \mathcal F(w)({\bf x}) - \mathcal F(w')({\bf x}) ||^2 ]
$$

Which can be formulated also from the perspective of the eucliden representation, in this way

$$
d_{\mathscr W}(\theta,\theta') =  \mathbb E_{{\bf x} \sim \mathcal U(\Omega)} [ || \mathcal F(\varphi^{-1}(\theta))({\bf x}) - \mathcal F(\varphi^{-1}(\theta))({\bf x}) ||^2 ]
$$

(note that $\mathcal F \circ \varphi^{-1}$ is basically the actual python implementation of the network).
We could construct, if the distance is differentiable the gradient of this distance

$$
\nabla_{\theta'} d_{\mathscr W}(\theta,\theta')
$$

and use it to minimize the actual distance between the two objects.

## Sparsity graph

Let $w \in \mathscr W$. We say that $w' \in \mathscr W$ is a <b> neighbour </b> of $w$ if and only if 

$$
\exists n \le N \land \varphi(w)_n \neq 0 : {\bf e}_n \odot \varphi(w) = \varphi(w')
$$ 

with ${\bf e}_n$ the $n-th$ vector in the canonical basis.
This basically means that for each possible parameter we can construct a set of neighbours with its variations, where only a parameter is set to zero.
Given the function $NZ: \mathbb R^N \rightarrow [0,N]$ that counts the non zero element in a vector, 
We can assume that if $|NZ(w) - NZ(w')| = 1$ then also $|| \varphi(w) - \varphi(w') ||_2$ is sufficiently small, especially for "large" $NZ$.
Therefore the Manifold Hypothesis should hold for all the neighbours, allowing to define a distance between the network and its neighbours.


## The Algorithm
The algorithm follows a simple greedy approach, in particular
1. Find the neighbour which is closer (according to our special distance) to the original network
2. Solve a constrained optimization problem to minimize the distance keeping the sparsity gained by taking a neighbour (which has a more rich sparsity pattern)
3. Repeat until convergence (decay of the accuracy estimate)

The following sections formalise each step and discuss the termination criterion and computational cost.

### Notation

Let $w^{(0)} \in \mathscr W$ be the original dense network. At each iteration $k$ we maintain a pair $(w^{(k)}, M^{(k)})$ where $w^{(k)}$ are the current parameters and $M^{(k)} \in \{0,1\}^N$ is a binary mask encoding the current sparsity pattern, with $M^{(k)}_n = 0$ iff parameter $n$ has been permanently removed. The set of still-active parameters is

$$
A^{(k)} = \{ n \le N : M^{(k)}_n = 1 \}
$$

and $N^{(k)} = |A^{(k)}|$ is the number of non-zero weights at step $k$. The forward pass of the network always evaluates $\mathcal F(\varphi^{-1}(M^{(k)} \odot \varphi(w^{(k)})))$, i.e.\ the mask is applied element-wise before any computation, so parameters outside $A^{(k)}$ are always invisible to the network.

### Step 1 — Pruning

The pruning step searches for the neighbour of $w^{(k)}$ that is closest to the original network $w^{(0)}$. Concretely, it selects the index

$$
n^* = \arg\min_{n \in A^{(k)}} \; d_{\mathscr W}\!\left(w^{(0)},\; \varphi^{-1}\!\left( (M^{(k)} - {\bf e}_n) \odot \varphi(w^{(k)}) \right)\right)
$$

That is, for each currently active parameter we tentatively zero it out and measure the resulting distance to the original network. The parameter whose removal causes the smallest increase in distance is selected. The updated parameters after this step are

$$
w^{(k + \tfrac{1}{2})} = \varphi^{-1}\!\left((M^{(k)} - {\bf e}_{n^*}) \odot \varphi(w^{(k)})\right), \qquad M^{(k+\tfrac{1}{2})} = M^{(k)} - {\bf e}_{n^*}
$$

If $d_{\mathscr W}(w^{(0)}, w^{(k+\tfrac{1}{2})}) = 0$ — meaning the removed weight was entirely redundant and no adjustment is needed — the iteration terminates early and $w^{(k+1)} = w^{(k+\tfrac{1}{2})}$.

### Step 2 — Adjustment

After pruning, the remaining active parameters are optimised to recover as much of the original network's behaviour as possible, subject to the fixed sparsity pattern $M^{(k+\tfrac{1}{2})}$. The problem is

$$
w^{(k+1)} = \arg\min_{\theta \,:\, M^{(k+\tfrac{1}{2})} \odot \varphi(\theta) = \varphi(\theta)} \; d_{\mathscr W}(w^{(0)}, \theta)
$$

Since $\mathcal F$ is differentiable almost everywhere (ReLU networks are piecewise linear), the objective is differentiable with respect to the active parameters. The gradient, computed via automatic differentiation, is

$$
\nabla_\theta \, d_{\mathscr W}(w^{(0)}, \theta) = 2 \,\mathbb E_{{\bf x} \sim \mathcal U(\Omega)} \!\left[ \left(\mathcal F(\theta)({\bf x}) - \mathcal F(w^{(0)})({\bf x})\right) \cdot \nabla_\theta \mathcal F(\theta)({\bf x}) \right]
$$

The sparsity constraint is maintained automatically: because the forward pass evaluates $M \odot \varphi(\theta)$, the partial derivative of $d_{\mathscr W}$ with respect to any masked-out parameter is exactly zero, so gradient updates leave those parameters unchanged.

Minimisation proceeds by gradient descent with an adaptive step size $\alpha$. Starting from $\alpha_0 = 10^{-11}$, at each inner iteration the step is accepted if it decreases the objective, in which case $\alpha$ is gently increased ($\alpha \leftarrow 1.001 \cdot \alpha$), and rejected with a halving ($\alpha \leftarrow \alpha/2$) otherwise. The inner loop terminates when $\alpha < 10^{-14}$.

### Monte Carlo approximation of the distance

The expectation $\mathbb E_{{\bf x} \sim \mathcal U(\Omega)}$ cannot be computed exactly in practice. It is approximated by a fixed Monte Carlo sample of $B$ points drawn uniformly from $\Omega$ before the main loop begins:

$$
d_{\mathscr W}(w, w') \;\approx\; \frac{1}{B} \sum_{b=1}^{B} \left\| \mathcal F(w)({\bf x}_b) - \mathcal F(w')({\bf x}_b) \right\|^2
$$

The same sample is reused across all evaluations throughout the entire run. This introduces a fixed estimation bias but guarantees that the objective is consistent across iterations — a property that would be lost if the sample were redrawn at each step. In the current implementation $B = 10{,}000$ and $\Omega = [0, 255]^{14 \times 14}$.

### Termination

The outer loop runs for a fixed number of iterations $K$ (currently $K = 500$). A more principled stopping criterion is to halt when no single-weight removal can be compensated by adjustment to within a tolerance $\delta > 0$:

$$
\min_{n \in A^{(k)}} \; d_{\mathscr W}(w^{(0)},\; w^{(k+\tfrac{1}{2})}) \;>\; \delta
$$

At this point every remaining active parameter is $\delta$-<b>critical</b>: its removal causes an irreducible perturbation to the network's behaviour. This threshold marks the natural limit of the sparsification procedure and is directly connected to the notion of the <b>stupidity point</b> — the sparsity level beyond which further compression necessarily degrades the network's function.

### Computational complexity

Each pruning step evaluates $d_{\mathscr W}$ for all $N^{(k)}$ candidate neighbours, at a cost of $O(B \cdot C)$ per evaluation, where $C$ is the cost of a single forward pass. Each adjustment step performs $T$ gradient iterations, each also costing $O(B \cdot C)$. Over $K$ outer iterations the total cost is

$$
O\!\left(K \cdot B \cdot C \cdot \left(\bar{N} + T\right)\right)
$$

where $\bar{N} = \frac{1}{K}\sum_{k=0}^{K-1} N^{(k)}$ is the average number of active parameters. Since $N^{(k)}$ decreases by exactly one at each step, $\bar{N} \approx N^{(0)} - K/2$, and the pruning phase accelerates progressively as the network becomes sparser. The adjustment cost $T$ depends on the curvature of the objective near the optimum and is harder to bound analytically; in practice it is the dominant term at early iterations when the perturbation caused by pruning is largest.