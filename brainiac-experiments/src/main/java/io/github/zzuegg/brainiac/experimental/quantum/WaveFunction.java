package io.github.zzuegg.brainiac.experimental.quantum;

/**
 * A real variational wave function over spin configurations (basis indices), with
 * log-derivatives with respect to its parameters: what variational Monte Carlo needs.
 */
public interface WaveFunction {

    /** @return the number of spins */
    int spins();

    /** @return the number of parameters */
    int parameterCount();

    /** @return the parameters (a live view; writing them changes the state) */
    double[] parameters();

    /**
     * @param basis a configuration
     * @return {@code log |psi|}
     */
    double logPsi(int basis);

    /**
     * @param basis a configuration
     * @return the sign of {@code psi}, +1 or -1
     */
    double sign(int basis);

    /**
     * @param basis a configuration
     * @param out receives {@code d log |psi| / d theta_k}
     */
    void logDerivatives(int basis, double[] out);

    /**
     * The local energy {@code E_loc(sigma) = sum_sigma' H_(sigma sigma') psi(sigma') / psi(sigma)}
     * of a real Hamiltonian.
     *
     * @param hamiltonian the Hamiltonian
     * @param basis the configuration
     * @return the local energy
     */
    default double localEnergy(PauliHamiltonian hamiltonian, int basis) {
        int[] rows = new int[hamiltonian.flipCount()];
        double[] vals = new double[rows.length];
        int k = hamiltonian.connections(basis, rows, vals);
        double lp = logPsi(basis);
        double sg = sign(basis);
        double e = 0;
        for (int j = 0; j < k; j++) {
            if (rows[j] == basis) {
                e += vals[j];
            } else {
                e += vals[j] * sign(rows[j]) * sg * Math.exp(logPsi(rows[j]) - lp);
            }
        }
        return e;
    }

    /**
     * The exact normalised state vector (for up to about 16 spins).
     *
     * @return the state
     */
    default StateVector toStateVector() {
        int dim = 1 << spins();
        double[] lp = new double[dim];
        double max = Double.NEGATIVE_INFINITY;
        for (int b = 0; b < dim; b++) {
            lp[b] = logPsi(b);
            max = Math.max(max, lp[b]);
        }
        double[] re = new double[dim];
        for (int b = 0; b < dim; b++) {
            re[b] = sign(b) * Math.exp(lp[b] - max);
        }
        return StateVector.of(re, null);
    }
}
