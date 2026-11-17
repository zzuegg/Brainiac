package io.github.zzuegg.brainiac.experimental.train;

/**
 * A generalised Hebbian plasticity rule, the "ABCD" rule used in evolved plasticity:
 * {@code dw_ij = eta * (A pre_i post_j + B pre_i + C post_j + D)}. Plain Hebb is
 * {@code A = 1}; Oja-like decay and homeostasis come from the other terms. Instead of
 * designing the coefficients, evolve them (with {@code CmaEs} or {@code EvolutionStrategy}
 * over {@link #toArray()}) so that a randomly started network learns <em>during its
 * lifetime</em>, without gradients.
 *
 * <p>Soltoggio, Stanley and Risi 2018, "Born to learn: the inspiration, progress, and
 * future of evolved plastic artificial neural networks",
 * <a href="https://arxiv.org/abs/1703.10371">arXiv:1703.10371</a>; Najarro and Risi 2020,
 * "Meta-Learning through Hebbian Plasticity in Random Networks",
 * <a href="https://arxiv.org/abs/2007.02686">arXiv:2007.02686</a>.
 *
 * @param eta the learning rate
 * @param a the correlation term
 * @param b the presynaptic term
 * @param c the postsynaptic term
 * @param d the constant drift
 * @param decay a weight decay applied every update, {@code w -= eta * decay * w}
 */
public record HebbianRule(double eta, double a, double b, double c, double d, double decay) {

    /** Plain Hebbian learning: {@code dw = 0.1 * pre * post}. */
    public static final HebbianRule HEBB = new HebbianRule(0.1, 1, 0, 0, 0, 0);

    /**
     * Reads a rule from a genome of six numbers in the order of the components.
     *
     * @param genome the genome
     * @return the rule
     */
    public static HebbianRule of(double[] genome) {
        if (genome.length != 6) {
            throw new IllegalArgumentException("a Hebbian rule genome has 6 values");
        }
        return new HebbianRule(Math.abs(genome[0]), genome[1], genome[2], genome[3], genome[4], genome[5]);
    }

    /** @return the rule as a genome of six numbers */
    public double[] toArray() {
        return new double[] {eta, a, b, c, d, decay};
    }

    /**
     * Updates a weight matrix in place.
     *
     * @param w the weights {@code [pre][post]}
     * @param pre presynaptic activity
     * @param post postsynaptic activity
     */
    public void update(double[][] w, double[] pre, double[] post) {
        for (int i = 0; i < pre.length; i++) {
            double[] row = w[i];
            for (int j = 0; j < post.length; j++) {
                double dw = a * pre[i] * post[j] + b * pre[i] + c * post[j] + d - decay * row[j];
                row[j] += eta * dw;
                if (!Double.isFinite(row[j])) {
                    row[j] = 0;
                } else {
                    row[j] = Math.max(-1e3, Math.min(1e3, row[j]));
                }
            }
        }
    }
}
