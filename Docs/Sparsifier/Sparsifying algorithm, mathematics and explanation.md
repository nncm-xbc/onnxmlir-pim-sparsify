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
metric space $(\mathscr W, w,w' \mapsto || \varphi(w) - \varphi(w') ||_2 )$, where $||\cdot||_2$ is the usual euclidean norm
on the euclidean space $\mathbb R^N$ which is used in this case to induce a distance.
We would like to define, also, a suitable distance that considers also the information regarding the realized neural network, through 
the map $\mathcal F$.In fact, if we were able to produce such distance, we would be able to characterize --- in a quantitative manner ---
the difference between two networks in terms of expression.