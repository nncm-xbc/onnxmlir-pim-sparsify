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
2. Solve a constrained optimization problem to minimize the distance keeping the sparsity gained by taking a neighbour (which has a more rich saprsity pattern)
3. Repeat until convergence (decay of the accuracy estimate)