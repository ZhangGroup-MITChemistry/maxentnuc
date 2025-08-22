# Maximum Entropy Inversion

The Maximum Entropy Inversion (MEI) method solves for an ensemble
of 3D conformations that are together consistent with experimental data. For our application, the experimental data
are pairwise contact frequency maps derived from 3C experiments. The MEI method is composed of 5 components:

1. A protocol for going from raw 3C data to contact frequencies.
2. Model: a prior model of the system.
3. A simulation protocol for drawing 3D conformations from the model.
4. Optimizer: a method for updating the potential energy function to better reproduce the contact frequencies.
5. A method for calculating contact frequencies from 3D conformations.

The overall workflow is orchestrated by `mei.py`. To run the script, you need to produce 
a yaml formatted config file specifying which region of the genome to simulate, a path to a mcool file with
normalized contact frequencies, and the values for various parameter.

## 1. 3C data normalization and mapping to nucleosomes

See na_genes/microc/README.md for a discussion of the normalization of 3C data and mapping to nucleosomes.

## 2. Model

Polymer models are implemented in `model.py`.

For the simple polymer model, key decisions are: setting the bead_diameter, bond potential, and angle potential.

## 3. Simulation protocol

Report interval: ideally the report interval should be set such that each saved frame is independent of the previous
frames. Practically, it makes sense to save a frame somewhat more frequently than this because quantities like the
end-to-end distance decorrelate more slowly than individual contacts.

Number of steps: the number of steps should be set such that we are able to accurately estimate the contact frequencies.
This is a function of the neighbor factor and the lowest probability contacts we wish to estimate. For instance, when
using rc=15 and sigma=8, the neighbor factor is 5.0, so if we wish to estimate contacts with a probability of 1e-3, then
we need to generate much more than 5e3 samples. A reasonable rule of thumb is to generate 10 times this minumum number
of samples.

## 4. Optimization

Optimizers are implemented in `optimizer.py`. The optimizer is responsible for updating the coefficients of the
potential energy function to better reproduce the contact frequencies. There are several optimizers implemented,
but the recommended optimizer is the `GradientDescentOptimizer` optimizer. This optimizer updates the external
potential coefficients according to:

$$ \alpha_{ij}^{t+1} = \alpha_{ij}^t + \eta (t_{ij} - s_{ij}^t)$$

where $\eta$ is the learning rate, $t_{ij}$ is the target contact frequency, and $s_{ij}$ is the simulated contact 
frequency.

To help dampen oscillations and smooth out low frequency contacts, which are hard to sample, we perform an exponential
moving average of the simulated contact frequencies. This is equivalent to momentum in the context of optimization.
Specifically, we update the simulated contact frequencies according to:

$$ s_{ij}^t = \beta s_{ij}^{(t-1)} + (1 - \beta) s_{ij} $$

where $\beta$ is the momentum parameter.

The two hyperparameters, the learning rate and momentum, must be tuned for different choices of the contact
indicator function and baseline model. We recommend the following strategy:

1. Begin with a momentum parameter of 0.75.
2. Tune the learning rate on a small test system containing two microdomains or some other hard to fit
   local structure. You should select the smallest learning rate that sufficiently reproduces the target
   contact map in approximately 10 iterations. Using the smallest feasible learning rate will help prevent
   the optimization from diverging on the full system.
3. Run the full system. If the contact frequencies are oscillating, repeat the previous step with a higher
   target number of steps.

To monitor convergence, you can use the script `plot_convergence.py`. The two most important metrics to consider are
the log total contact ratio between the simulated and reference contact maps and the log smoothed individual contact
ratio. The total contact ratio should be close to zero, ideally less than 10% (log2(1.1) ~ 0.15).

## 5. Calculating contact frequencies from 3D conformations

We compute contact frequencies from 3D conformations by defining a "contact indicator function" that specifies the
likelihood of a ligation between two loci as a function of distance and then averaging the values of the indicator
function over all conformations. Additionally, we include a global scaling
parameter to make the simulated and experimental contact maps comparable. The contact indicator function and code for
computing contacts is defined in `simulated_contacts.py`.

We use the contact indicator function

$$ f(r) = \frac{1}{2} (1 + \tanh(\sigma(r_c - r))) $$

where $r$ is the distance between two loci, $r_c$ is the contact radius, and $\sigma$ is a steepness parameter.

**The contact indicator function must match the form of the external potential applied in the simulations**

### Scaling the contact map

After computing the simulated contact map, we apply a global scaling constant such that the average values of the first
diagonals of the experimental and simulated contact maps
are equal. If $r_c$ is greater than the bond length, then this constant is close to one. Note that this constant
perhaps more naturally belongs as a scaling factor of the experimental map, but it is more convenient to include it
in the simulated map.

### Choosing $r_c$ and $\sigma$

The values of $r_c$ and $\sigma$ have a major impact on the resulting conformational ensemble. Unfortunately, there
is not a clear way to determine these values from first-principles, so they must be determined empirically. Most
prominently, these parameters should be set such that the overall density of the system is consistent with
experimentally determined values. Moreover, the parameters should be set to recapitulate local features of chromatin
determined through cryo-ET.

There are two regions of parameter space that result in the correct overall density of the system. The first corresponds
to an $r_c$ of approximately 2 bond lengths and a $\sigma$ of 2 to 4. The second corresponds to an $r_c$ of ~15 and a
$\sigma$ of 8. Intermediate values of $r_c$ result in far too high of a system density and further increasing or decreasing
$r_c$ results in a far too low system density. Note that in the second regime, the global scaling constant is 
large because $r_c$ is less than the bond length, which is what causes the system density to decrease.

Of these two solutions, the first option is more reasonable because small values of $r_c$ are not consistent with the
i–i+2 stacking contacts observed in cryo-ET. If even a small fraction of nucleosomes formed stacking interactions, it
would cause the i–i+2 contacts to far outweigh the i–i+1 contacts in the second regime. However, in the experimental
data, the i–i+1 contacts are more frequent than the i–i+2 contacts. Therefore, only the first regime can recapitulate
the trends observed in the experimental data.

The value of $\sigma$ controls the steepness of the contact indicator function. In the first regime, the value of
$\sigma$ should be set to 2 as this gives little decay in contact frequency between direct contacts (11 nm) and one
bond length (21 nm) while not giving a particularly steep decay (which would give a seemingly arbitrary cutoff).
As mentioned above, a steep decay between direct contacts and a bond length is not consistent with the experimental data,
both due to the decay between i–i+1 and i–i+2 contacts and the lack of significant variation in i–i+2 contacts. It seems
reasonable that some i–i+2 pairs stack much more frequently than others, so we would expect a substantial variation in
i–i+2 contact frequencies if this strongly influenced ligation efficiency.

Other thoughts on the contact indicator function:
* An $r_c$ much greater than 2 bond lengths would be unable to reproduce the decay in contact frequency between i and i+1
* Given that each nucleosome is ~11 nm in diameter and has histone tails of ~10 nm, it seems unlikely that there would
  be a significant decrease in ligation efficiency over 21 nm (1 bond length).
* Ligations likely largely occur between small groups of cross-linked nucleosomes. This suggests an exponential
  decay as distance increases.

### Limitations of this formulation

This formulation assumes that ligation efficiency is solely a function of distance, which is not necessarily true.
First, the ligation efficiency could be influenced by the angle between the nucleosomes. If the free DNA ends
are lined up it might be more likely to ligate. This could be  particularly relevant for stacked i+2 nucleosomes,
which could form a rigid structure. Second, the presence of nucleosomes or other proteins between two nucleosomes
could either sterically
hinder ligation or could serve as a cross-linked bridge between them increasing ligation efficiency.

Finally, there could be competition between different ligation products. In a single cell, a nucleosome can only form a
productive ligation to one other nucleosome. This could result in nucleosomes in more dilute regions being more likely
to ligate. The code includes an option to account for this by dividing each contact by all possible contacts including
the two loci of interest to get an augmented contact frequency:

$$ f'_{ij} = \frac{f_{ij}}{\sum_{k != i} f_{ik} + \sum_{k != j} f_{kj} + f_{ij}} $$

Interestingly, this effect is largely accounted for by neighbor balancing. For the augmented contact frequency for a 
pair of neighboring loci $i$ and $i+1$ to match an experimental value $t_{ij, i+1}$,
it must be that the denominator of the above expression is equal to 1 / $t_{i, i+1}$. This is quite similar to the
effect of neighbor balancing, where at the beginning we (roughly speaking) divide each row by $t_{i, i+1}$. Empirically,
if we run MEI on a neighbor balanced contact map and then compute a contact map accounting for competition, the map
is more similar to the original ICE balanced map.
