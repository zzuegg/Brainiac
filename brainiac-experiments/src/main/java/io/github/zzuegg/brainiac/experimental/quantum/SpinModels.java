package io.github.zzuegg.brainiac.experimental.quantum;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Test problems for the quantum blocks: spin-chain and lattice Hamiltonians as sparse Pauli
 * sums (transverse-field Ising, disordered Ising, Heisenberg XXZ), MaxCut instances (random
 * regular and Erdős–Rényi graphs) with their cost operators and brute-force optima, and
 * reference states for tomography (GHZ, W, random product states).
 */
public final class SpinModels {

    private SpinModels() {
    }

    /**
     * The transverse-field Ising chain {@code H = -J sum Z_i Z_(i+1) - h sum X_i}, critical at
     * {@code h = J} in the thermodynamic limit.
     *
     * @param n sites
     * @param j the coupling
     * @param h the transverse field
     * @param periodic whether the chain closes into a ring
     * @return the Hamiltonian
     */
    public static PauliHamiltonian tfim(int n, double j, double h, boolean periodic) {
        double[] js = new double[n];
        java.util.Arrays.fill(js, j);
        return tfim(js, h, periodic);
    }

    /**
     * A transverse-field Ising chain with bond-dependent couplings (the disordered variant).
     *
     * @param j the coupling of bond {@code (i, i+1)}; the last is used only when periodic
     * @param h the field
     * @param periodic whether the chain is a ring
     * @return the Hamiltonian
     */
    public static PauliHamiltonian tfim(double[] j, double h, boolean periodic) {
        int n = j.length;
        List<PauliHamiltonian.Term> t = new ArrayList<>();
        for (int i = 0; i < (periodic ? n : n - 1); i++) {
            t.add(PauliHamiltonian.term(-j[i], "ZZ", i, (i + 1) % n));
        }
        for (int i = 0; i < n; i++) {
            t.add(PauliHamiltonian.term(-h, "X", i));
        }
        return new PauliHamiltonian(n, t);
    }

    /**
     * Random couplings {@code J_i} uniform in {@code [1 - w, 1 + w]}.
     *
     * @param n sites
     * @param width the half-width {@code w}
     * @param seed the seed
     * @return the couplings
     */
    public static double[] disorderedCouplings(int n, double width, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] j = new double[n];
        for (int i = 0; i < n; i++) {
            j[i] = 1 + width * (2 * r.nextDouble() - 1);
        }
        return j;
    }

    /**
     * The transverse-field Ising model on an {@code lx x ly} open square lattice.
     *
     * @param lx width
     * @param ly height
     * @param j coupling
     * @param h field
     * @return the Hamiltonian
     */
    public static PauliHamiltonian tfim2d(int lx, int ly, double j, double h) {
        List<PauliHamiltonian.Term> t = new ArrayList<>();
        for (int y = 0; y < ly; y++) {
            for (int x = 0; x < lx; x++) {
                int s = y * lx + x;
                if (x + 1 < lx) {
                    t.add(PauliHamiltonian.term(-j, "ZZ", s, s + 1));
                }
                if (y + 1 < ly) {
                    t.add(PauliHamiltonian.term(-j, "ZZ", s, s + lx));
                }
                t.add(PauliHamiltonian.term(-h, "X", s));
            }
        }
        return new PauliHamiltonian(lx * ly, t);
    }

    /**
     * The Heisenberg XXZ chain {@code H = J sum (X X + Y Y + delta Z Z)} (antiferromagnetic for
     * {@code J > 0}; {@code delta = 1} is the isotropic Heisenberg chain).
     *
     * @param n sites
     * @param j the coupling
     * @param delta the anisotropy
     * @param periodic whether the chain is a ring
     * @return the Hamiltonian
     */
    public static PauliHamiltonian xxz(int n, double j, double delta, boolean periodic) {
        List<PauliHamiltonian.Term> t = new ArrayList<>();
        for (int i = 0; i < (periodic ? n : n - 1); i++) {
            int k = (i + 1) % n;
            t.add(PauliHamiltonian.term(j, "XX", i, k));
            t.add(PauliHamiltonian.term(j, "YY", i, k));
            t.add(PauliHamiltonian.term(j * delta, "ZZ", i, k));
        }
        return new PauliHamiltonian(n, t);
    }

    // ------------------------------------------------------------------ MaxCut

    /**
     * A random {@code d}-regular graph by the configuration model with restarts.
     *
     * @param n nodes ({@code n d} even)
     * @param d the degree
     * @param seed the seed
     * @return the 0/1 adjacency matrix
     */
    public static double[][] regularGraph(int n, int d, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        for (int attempt = 0; attempt < 10_000; attempt++) {
            int[] stubs = new int[n * d];
            for (int i = 0; i < stubs.length; i++) {
                stubs[i] = i / d;
            }
            for (int i = stubs.length - 1; i > 0; i--) {
                int k = r.nextInt(i + 1);
                int tmp = stubs[i];
                stubs[i] = stubs[k];
                stubs[k] = tmp;
            }
            double[][] w = new double[n][n];
            boolean ok = true;
            for (int i = 0; i < stubs.length && ok; i += 2) {
                int a = stubs[i];
                int b = stubs[i + 1];
                if (a == b || w[a][b] != 0) {
                    ok = false;
                } else {
                    w[a][b] = 1;
                    w[b][a] = 1;
                }
            }
            if (ok) {
                return w;
            }
        }
        throw new IllegalStateException("no simple regular graph found");
    }

    /**
     * An Erdős–Rényi graph {@code G(n, p)}.
     *
     * @param n nodes
     * @param p the edge probability
     * @param seed the seed
     * @return the 0/1 adjacency matrix
     */
    public static double[][] erdosRenyi(int n, double p, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] w = new double[n][n];
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                if (r.nextDouble() < p) {
                    w[a][b] = 1;
                    w[b][a] = 1;
                }
            }
        }
        return w;
    }

    /**
     * The cut value of every basis state, {@code C(b) = sum_(i<j) w_ij [b_i != b_j]}.
     *
     * @param w symmetric weights
     * @return the cost diagonal
     */
    public static double[] maxCutDiagonal(double[][] w) {
        int n = w.length;
        double[] c = new double[1 << n];
        for (int b = 0; b < c.length; b++) {
            double s = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    if (w[i][j] != 0 && (((b >> i) ^ (b >> j)) & 1) == 1) {
                        s += w[i][j];
                    }
                }
            }
            c[b] = s;
        }
        return c;
    }

    /**
     * The MaxCut cost as a Pauli sum, {@code C = sum w_ij (1 - Z_i Z_j) / 2} (the constant is
     * kept as a term on the identity).
     *
     * @param w symmetric weights
     * @return the cost operator
     */
    public static PauliHamiltonian maxCutCost(double[][] w) {
        int n = w.length;
        List<PauliHamiltonian.Term> t = new ArrayList<>();
        double constant = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (w[i][j] != 0) {
                    constant += w[i][j] / 2;
                    t.add(PauliHamiltonian.term(-w[i][j] / 2, "ZZ", i, j));
                }
            }
        }
        t.add(new PauliHamiltonian.Term(constant, 0, 0));
        return new PauliHamiltonian(n, t);
    }

    // ------------------------------------------------------------------ states

    /**
     * {@code (|0...0> + |1...1>) / sqrt 2}.
     *
     * @param n qubits
     * @return the GHZ state
     */
    public static StateVector ghz(int n) {
        double[] re = new double[1 << n];
        re[0] = 1;
        re[(1 << n) - 1] = 1;
        return StateVector.of(re, null);
    }

    /**
     * The W state, the equal superposition of all single-excitation states.
     *
     * @param n qubits
     * @return the state
     */
    public static StateVector w(int n) {
        double[] re = new double[1 << n];
        for (int q = 0; q < n; q++) {
            re[1 << q] = 1;
        }
        return StateVector.of(re, null);
    }

    /**
     * A Haar-random product of single-qubit states.
     *
     * @param n qubits
     * @param seed the seed
     * @return the state
     */
    public static StateVector randomProduct(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        StateVector s = new StateVector(n);
        for (int q = 0; q < n; q++) {
            s.ry(q, Math.acos(2 * r.nextDouble() - 1));
            s.rz(q, 2 * Math.PI * r.nextDouble());
        }
        return s;
    }
}
