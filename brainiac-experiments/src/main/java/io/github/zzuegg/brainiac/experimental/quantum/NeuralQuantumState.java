package io.github.zzuegg.brainiac.experimental.quantum;

import java.util.SplittableRandom;

/**
 * A neural-network quantum state: a network that gives the log-amplitude
 * {@code log psi(sigma)} of every spin configuration {@code sigma} (Carleo and Troyer,
 * "Solving the quantum many-body problem with artificial neural networks", Science 2017,
 * <a href="https://arxiv.org/abs/1606.02318">arXiv:1606.02318</a>). Configurations are basis
 * indices; spin {@code s_i = +1} when bit {@code i} is 0 (the Z eigenvalue).
 *
 * <ul>
 *   <li>{@link Ansatz#RBM}: a restricted Boltzmann machine with {@code alpha n} hidden units,
 *       {@code log psi = sum a_i s_i + sum_j log cosh(b_j + sum_i W_ji s_i)}.</li>
 *   <li>{@link Ansatz#JASTROW}: {@code log psi = sum a_i s_i + sum_(i<j) J_ij s_i s_j}.</li>
 *   <li>{@link Ansatz#MLP}: one tanh hidden layer summed to a scalar,
 *       {@code log psi = sum a_i s_i + sum_j v_j tanh(c_j + sum_i W_ji s_i)}.</li>
 * </ul>
 * The amplitudes are positive (right for stoquastic Hamiltonians such as the TFIM); with
 * {@link #marshallSign(boolean)} the sign {@code (-1)^(up spins on odd sites)} of Marshall's
 * rule is applied, which makes the antiferromagnetic Heisenberg ground state representable.
 *
 * <p>It provides log-derivatives {@code O_k = d log psi / d theta_k} for variational Monte Carlo,
 * local energies against a {@link PauliHamiltonian}, and a Metropolis sampler of
 * {@code |psi|^2} with single flips or magnetisation-conserving exchanges.
 */
public final class NeuralQuantumState implements WaveFunction {

    /** The network family. */
    public enum Ansatz {
        /** Restricted Boltzmann machine. */
        RBM,
        /** Two-body Jastrow factor. */
        JASTROW,
        /** One-hidden-layer tanh network. */
        MLP
    }

    private final Ansatz ansatz;
    private final int n;
    private final int hidden;
    private final double[] theta;
    private boolean marshall;

    private NeuralQuantumState(Ansatz ansatz, int n, int hidden, long seed, double scale) {
        this.ansatz = ansatz;
        this.n = n;
        this.hidden = hidden;
        int count = switch (ansatz) {
            case RBM -> n + hidden + hidden * n;
            case JASTROW -> n + n * (n - 1) / 2;
            case MLP -> n + hidden + hidden + hidden * n;
        };
        this.theta = new double[count];
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < count; i++) {
            theta[i] = scale * r.nextGaussian();
        }
    }

    /**
     * An RBM state with small random weights.
     *
     * @param n spins
     * @param alpha hidden units per spin
     * @param seed the seed
     * @return the state
     */
    public static NeuralQuantumState rbm(int n, int alpha, long seed) {
        return new NeuralQuantumState(Ansatz.RBM, n, alpha * n, seed, 0.01);
    }

    /**
     * A Jastrow state.
     *
     * @param n spins
     * @param seed the seed
     * @return the state
     */
    public static NeuralQuantumState jastrow(int n, long seed) {
        return new NeuralQuantumState(Ansatz.JASTROW, n, 0, seed, 0.01);
    }

    /**
     * An MLP state.
     *
     * @param n spins
     * @param hidden hidden units
     * @param seed the seed
     * @return the state
     */
    public static NeuralQuantumState mlp(int n, int hidden, long seed) {
        return new NeuralQuantumState(Ansatz.MLP, n, hidden, seed, 0.05);
    }

    /**
     * Turns Marshall's sign rule on or off.
     *
     * @param on whether to apply it
     * @return this state
     */
    public NeuralQuantumState marshallSign(boolean on) {
        this.marshall = on;
        return this;
    }

    /** @return the ansatz */
    public Ansatz ansatz() {
        return ansatz;
    }

    /** @return the number of spins */
    @Override
    public int spins() {
        return n;
    }

    /** @return the number of parameters */
    @Override
    public int parameterCount() {
        return theta.length;
    }

    /** @return the parameters (a live view) */
    @Override
    public double[] parameters() {
        return theta;
    }

    private static double spin(int basis, int i) {
        return ((basis >> i) & 1) == 0 ? 1 : -1;
    }

    /**
     * The sign of the amplitude.
     *
     * @param basis the configuration
     * @return +1 or -1
     */
    @Override
    public double sign(int basis) {
        if (!marshall) {
            return 1;
        }
        int odd = 0;
        for (int i = 1; i < n; i += 2) {
            odd += (basis >> i) & 1;
        }
        return (odd & 1) == 0 ? 1 : -1;
    }

    /**
     * {@code log |psi(sigma)|}.
     *
     * @param basis the configuration
     * @return the log-amplitude
     */
    @Override
    public double logPsi(int basis) {
        double v = 0;
        for (int i = 0; i < n; i++) {
            v += theta[i] * spin(basis, i);
        }
        switch (ansatz) {
            case RBM -> {
                int w = n + hidden;
                for (int j = 0; j < hidden; j++) {
                    double t = theta[n + j];
                    for (int i = 0; i < n; i++) {
                        t += theta[w + j * n + i] * spin(basis, i);
                    }
                    v += logCosh(t);
                }
            }
            case JASTROW -> {
                int k = n;
                for (int i = 0; i < n; i++) {
                    for (int j = i + 1; j < n; j++) {
                        v += theta[k++] * spin(basis, i) * spin(basis, j);
                    }
                }
            }
            case MLP -> {
                int c = n;
                int vv = n + hidden;
                int w = n + 2 * hidden;
                for (int j = 0; j < hidden; j++) {
                    double t = theta[c + j];
                    for (int i = 0; i < n; i++) {
                        t += theta[w + j * n + i] * spin(basis, i);
                    }
                    v += theta[vv + j] * Math.tanh(t);
                }
            }
            default -> throw new IllegalStateException();
        }
        return v;
    }

    private static double logCosh(double x) {
        double a = Math.abs(x);
        return a + Math.log1p(Math.exp(-2 * a)) - Math.log(2);
    }

    /**
     * The log-derivatives {@code O_k(sigma) = d log psi / d theta_k}.
     *
     * @param basis the configuration
     * @param out receives one value per parameter
     */
    @Override
    public void logDerivatives(int basis, double[] out) {
        for (int i = 0; i < n; i++) {
            out[i] = spin(basis, i);
        }
        switch (ansatz) {
            case RBM -> {
                int w = n + hidden;
                for (int j = 0; j < hidden; j++) {
                    double t = theta[n + j];
                    for (int i = 0; i < n; i++) {
                        t += theta[w + j * n + i] * spin(basis, i);
                    }
                    double th = Math.tanh(t);
                    out[n + j] = th;
                    for (int i = 0; i < n; i++) {
                        out[w + j * n + i] = th * spin(basis, i);
                    }
                }
            }
            case JASTROW -> {
                int k = n;
                for (int i = 0; i < n; i++) {
                    for (int j = i + 1; j < n; j++) {
                        out[k++] = spin(basis, i) * spin(basis, j);
                    }
                }
            }
            case MLP -> {
                int c = n;
                int vv = n + hidden;
                int w = n + 2 * hidden;
                for (int j = 0; j < hidden; j++) {
                    double t = theta[c + j];
                    for (int i = 0; i < n; i++) {
                        t += theta[w + j * n + i] * spin(basis, i);
                    }
                    double th = Math.tanh(t);
                    double d = theta[vv + j] * (1 - th * th);
                    out[vv + j] = th;
                    out[c + j] = d;
                    for (int i = 0; i < n; i++) {
                        out[w + j * n + i] = d * spin(basis, i);
                    }
                }
            }
            default -> throw new IllegalStateException();
        }
    }

    /**
     * Metropolis samples of {@code |psi|^2}.
     *
     * @param count samples to return
     * @param sweeps sweeps of {@code n} proposals between samples
     * @param burnIn sweeps discarded first
     * @param exchange whether to propose swaps of two opposite spins (conserving
     *     magnetisation, starting from zero magnetisation) instead of single flips
     * @param random the random source
     * @return the sampled configurations
     */
    public int[] sample(int count, int sweeps, int burnIn, boolean exchange, SplittableRandom random) {
        int b = 0;
        if (exchange) {
            for (int i = 0; i < n; i += 2) {
                b |= 1 << i;
            }
        } else {
            b = random.nextInt(1 << n);
        }
        double lp = logPsi(b);
        int[] out = new int[count];
        int taken = 0;
        int sweep = 0;
        while (taken < count) {
            for (int m = 0; m < n; m++) {
                int c;
                if (exchange) {
                    int i = random.nextInt(n);
                    int j = random.nextInt(n);
                    if (((b >> i) & 1) == ((b >> j) & 1)) {
                        continue;
                    }
                    c = b ^ (1 << i) ^ (1 << j);
                } else {
                    c = b ^ (1 << random.nextInt(n));
                }
                double lc = logPsi(c);
                if (Math.log(random.nextDouble()) < 2 * (lc - lp)) {
                    b = c;
                    lp = lc;
                }
            }
            sweep++;
            if (sweep > burnIn && (sweep - burnIn) % sweeps == 0) {
                out[taken++] = b;
            }
        }
        return out;
    }
}
