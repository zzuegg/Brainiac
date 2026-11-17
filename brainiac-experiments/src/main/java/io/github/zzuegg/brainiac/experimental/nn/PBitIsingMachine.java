package io.github.zzuegg.brainiac.experimental.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A network of probabilistic bits (p-bits; Camsari et al., "Stochastic p-bits for invertible
 * logic", <a href="https://arxiv.org/abs/1610.00377">1610.00377</a>), the building block of
 * probabilistic and thermodynamic computers. Each p-bit is a binary spin {@code m_i = +-1}
 * that flips according to its input
 *
 * <pre>
 * I_i = sum_j J_ij m_j + h_i,    m_i = sgn(tanh(beta I_i) - r),   r ~ U(-1, 1)
 * </pre>
 *
 * Updating the bits one at a time is Gibbs sampling, so the network settles into the
 * Boltzmann distribution {@code p(m) ~ exp(-beta E(m))} of the Ising energy
 * {@code E(m) = -sum_(i<j) J_ij m_i m_j - sum_i h_i m_i}. Clamping some bits and sampling
 * the rest computes conditionals, which runs logic "backwards" (invertible logic). Couplings
 * are stored as sparse neighbour lists, so large lattices are cheap.
 */
public final class PBitIsingMachine {
    private final int size;
    private final int[][] neighbours;
    private final double[][] couplings;
    private final double[] bias;
    private final int[] spins;
    private final boolean[] clamped;

    private PBitIsingMachine(double[][] j, double[] h) {
        this.size = h.length;
        this.bias = h.clone();
        this.neighbours = new int[size][];
        this.couplings = new double[size][];
        for (int i = 0; i < size; i++) {
            List<Integer> idx = new ArrayList<>();
            for (int k = 0; k < size; k++) {
                if (k != i && j[i][k] != 0) {
                    if (j[i][k] != j[k][i]) {
                        throw new IllegalArgumentException("couplings must be symmetric");
                    }
                    idx.add(k);
                }
            }
            neighbours[i] = idx.stream().mapToInt(Integer::intValue).toArray();
            couplings[i] = new double[neighbours[i].length];
            for (int n = 0; n < neighbours[i].length; n++) {
                couplings[i][n] = j[i][neighbours[i][n]];
            }
        }
        this.spins = new int[size];
        java.util.Arrays.fill(spins, 1);
        this.clamped = new boolean[size];
    }

    /**
     * Creates a machine from dense couplings.
     *
     * @param j symmetric couplings with a zero diagonal
     * @param h biases
     * @return the machine, all spins +1
     */
    public static PBitIsingMachine of(double[][] j, double[] h) {
        return new PBitIsingMachine(j, h);
    }

    /**
     * A ferromagnetic square lattice with periodic boundaries and unit couplings.
     *
     * @param side the side length
     * @return the machine
     */
    public static PBitIsingMachine lattice(int side) {
        int n = side * side;
        double[][] j = new double[n][n];
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                int i = y * side + x;
                int right = y * side + (x + 1) % side;
                int down = ((y + 1) % side) * side + x;
                j[i][right] = j[right][i] = 1;
                j[i][down] = j[down][i] = 1;
            }
        }
        return new PBitIsingMachine(j, new double[n]);
    }

    /** @return the number of p-bits */
    public int size() {
        return size;
    }

    /**
     * Sets all free spins at random.
     *
     * @param random the random source
     */
    public void randomize(RandomGenerator random) {
        for (int i = 0; i < size; i++) {
            if (!clamped[i]) {
                spins[i] = random.nextBoolean() ? 1 : -1;
            }
        }
    }

    /**
     * Fixes a spin.
     *
     * @param i the bit
     * @param value +1 or -1
     */
    public void clamp(int i, int value) {
        if (value != 1 && value != -1) {
            throw new IllegalArgumentException("spins are +1 or -1");
        }
        spins[i] = value;
        clamped[i] = true;
    }

    /** Frees all clamped spins. */
    public void release() {
        java.util.Arrays.fill(clamped, false);
    }

    /**
     * The input of one p-bit.
     *
     * @param i the bit
     * @return {@code I_i}
     */
    public double input(int i) {
        double s = bias[i];
        int[] nb = neighbours[i];
        double[] c = couplings[i];
        for (int n = 0; n < nb.length; n++) {
            s += c[n] * spins[nb[n]];
        }
        return s;
    }

    /**
     * One sequential sweep over all free p-bits.
     *
     * @param beta the inverse temperature
     * @param random the random source
     */
    public void sweep(double beta, RandomGenerator random) {
        for (int i = 0; i < size; i++) {
            if (!clamped[i]) {
                spins[i] = Math.tanh(beta * input(i)) > random.nextDouble(-1, 1) ? 1 : -1;
            }
        }
    }

    /** @return a copy of the spins */
    public int[] spins() {
        return spins.clone();
    }

    /**
     * The state as an integer, bit i set when spin i is +1 (for at most 31 spins).
     *
     * @return the code
     */
    public int code() {
        int c = 0;
        for (int i = 0; i < size; i++) {
            if (spins[i] > 0) {
                c |= 1 << i;
            }
        }
        return c;
    }

    /** @return the mean spin */
    public double magnetization() {
        double s = 0;
        for (int v : spins) {
            s += v;
        }
        return s / size;
    }

    /**
     * The Ising energy of a state.
     *
     * @param state spins
     * @return {@code E}
     */
    public double energy(int[] state) {
        double e = 0;
        for (int i = 0; i < size; i++) {
            e -= bias[i] * state[i];
            for (int n = 0; n < neighbours[i].length; n++) {
                if (neighbours[i][n] > i) {
                    e -= couplings[i][n] * state[i] * state[neighbours[i][n]];
                }
            }
        }
        return e;
    }

    /**
     * The exact Boltzmann distribution by enumeration, for small machines.
     *
     * @param beta the inverse temperature
     * @return the probability of each state code
     */
    public double[] boltzmann(double beta) {
        if (size > 20) {
            throw new IllegalStateException("too many states to enumerate");
        }
        double[] p = new double[1 << size];
        double z = 0;
        int[] s = new int[size];
        for (int c = 0; c < p.length; c++) {
            for (int i = 0; i < size; i++) {
                s[i] = (c >> i & 1) == 1 ? 1 : -1;
            }
            p[c] = Math.exp(-beta * energy(s));
            z += p[c];
        }
        for (int c = 0; c < p.length; c++) {
            p[c] /= z;
        }
        return p;
    }
}
