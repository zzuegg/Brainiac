package io.github.zzuegg.brainiac.experimental.circuits;

/**
 * Inception probabilistic circuits (Wang and Van den Broeck, AAAI 2025,
 * <a href="https://arxiv.org/abs/2408.00876">arXiv:2408.00876</a>):
 * {@code p(x) proportional to sum_u |sum_w f(x, u, w)|^2}, with some latent variables ({@code u})
 * summed outside the square, monotonically, and the others ({@code w}, the circuit's units)
 * inside it, with real or complex weights. With one outer value it is a
 * {@link SquaredCircuit}; with one unit per region inside the square it is a monotone mixture of
 * squared product densities, so it contains both classes, which the paper proves are
 * incomparable at compact size. The {@code U} components share their leaf layer and differ in
 * their sum layers, and their normalising constants add, so {@code Z} stays exact.
 */
public final class InceptionCircuit {

    private final SquaredCircuit circuit;

    /**
     * Creates the circuit.
     *
     * @param categories categories per variable
     * @param units units per region inside the square
     * @param complex complex weights
     * @param outer {@code U}, the outer (monotone) latent's values
     * @param seed the seed
     */
    public InceptionCircuit(int[] categories, int units, boolean complex, int outer, long seed) {
        this.circuit = new SquaredCircuit(categories, units, complex, outer, seed);
    }

    /**
     * Trains by Adam.
     *
     * @param x samples
     * @param steps steps
     * @param batch batch size
     * @param lr learning rate
     * @param seed the seed
     * @return this
     */
    public InceptionCircuit fit(int[][] x, int steps, int batch, float lr, long seed) {
        circuit.fit(x, steps, batch, lr, seed);
        return this;
    }

    /**
     * @param x samples
     * @return the mean log-likelihood
     */
    public double meanLogLikelihood(int[][] x) {
        return circuit.meanLogLikelihood(x);
    }

    /** @return the number of real parameters */
    public int parameterCount() {
        return circuit.parameterCount();
    }
}
