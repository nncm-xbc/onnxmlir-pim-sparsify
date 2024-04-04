# Mathematics of the sparsification

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
d_{\mathscr W}(w,w') =  \mathbb E_{{\bf x} \sim \mathcal U(\Omega)} [ || \mathcal F(w)({\bf x}) - \mathcal F(w')({\bf x}) || ]
$$

Where $\Omega$ is the compact where data distribution appears. For example, in the case of the MNIST dataset could be $[0,255]^{28*28}$