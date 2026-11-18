package io.github.zzuegg.brainiac.experimental.quantum;

import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Quantum state tomography from single-qubit Pauli measurements: every shot measures each
 * qubit in a uniformly random basis X, Y or Z.
 *
 * <ul>
 *   <li>{@link #linearInversion}: the unbiased estimator {@code rho = 2^-n sum_P <P> P}, with each
 *       Pauli expectation estimated from the shots whose bases match its support; it is often
 *       unphysical (negative eigenvalues) at low shot counts.</li>
 *   <li>{@link #maximumLikelihood}: Hradil's {@code R rho R} fixed-point iteration (PRA 1997),
 *       {@code R = sum_k f_k / p_k(rho) Pi_k}, which keeps {@code rho} positive.</li>
 *   <li>{@link #neuralRbm}: neural-network tomography (Torlai et al., Nature Physics 2018): a
 *       pure state with an RBM amplitude and an RBM phase, {@code psi = exp(A(sigma) + i Phi(sigma))},
 *       trained by Adam on the negative log-likelihood of the measured outcomes in their rotated
 *       bases; it scales to more qubits than a density matrix.</li>
 *   <li>{@link #shadowEstimate}: classical shadows (Huang, Kueng and Preskill, Nature Physics
 *       2020, <a href="https://arxiv.org/abs/2002.08953">arXiv:2002.08953</a>): predict any local
 *       Pauli observable directly from the same random-basis records as the median of means of
 *       {@code prod_q 3 [basis_q = P_q] (-1)^(outcome_q)}, with a sample complexity that grows
 *       only logarithmically in the number of observables.</li>
 *   <li>{@link #directEstimate}: the naive protocol shadows replace: measure every observable in
 *       its own basis with an equal share of the shots.</li>
 * </ul>
 * Pauli strings use {@link PauliHamiltonian.Term} masks; bases are 0 = Z, 1 = X, 2 = Y per qubit.
 */
public final class Tomography {

    /** The reconstruction method. */
    public enum Method {
        /** Linear inversion. */
        LINEAR_INVERSION,
        /** Maximum likelihood by the R rho R iteration. */
        MLE_RRHO,
        /** Neural-network (RBM) tomography of a pure state. */
        NEURAL_RBM,
        /** Classical shadows for observables (no reconstruction). */
        CLASSICAL_SHADOWS
    }

    /**
     * One shot.
     *
     * @param basis per qubit 0 = Z, 1 = X, 2 = Y
     * @param outcome the outcome bits (1 = eigenvalue -1)
     */
    public record Shot(int[] basis, int outcome) {
    }

    /**
     * A Hermitian matrix as real and imaginary parts.
     *
     * @param re real part
     * @param im imaginary part
     */
    public record Density(double[][] re, double[][] im) {

        /**
         * {@code <psi|rho|psi>}, the fidelity with a pure state.
         *
         * @param psi the state
         * @return the fidelity (may exceed [0, 1] for unphysical estimates)
         */
        public double fidelity(StateVector psi) {
            double[] pr = psi.real();
            double[] pi = psi.imag();
            int d = pr.length;
            double f = 0;
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    // conj(psi_a) rho_ab psi_b
                    double rr = re[a][b] * pr[b] - im[a][b] * pi[b];
                    double ri = re[a][b] * pi[b] + im[a][b] * pr[b];
                    f += pr[a] * rr + pi[a] * ri;
                }
            }
            return f;
        }

        /** @return the smallest eigenvalue (negative means unphysical) */
        public double minEigenvalue() {
            int d = re.length;
            double[][] big = new double[2 * d][2 * d];
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    big[a][b] = re[a][b];
                    big[a + d][b + d] = re[a][b];
                    big[a][b + d] = -im[a][b];
                    big[a + d][b] = im[a][b];
                }
            }
            double[] v = Matrices.symmetricEigen(big).values();
            double min = Double.POSITIVE_INFINITY;
            for (double x : v) {
                min = Math.min(min, x);
            }
            return min;
        }
    }

    private Tomography() {
    }

    // ------------------------------------------------------------------ measurement

    /**
     * Simulates random-Pauli-basis shots.
     *
     * @param psi the state
     * @param shots the number of shots
     * @param random the random source
     * @return the records
     */
    public static Shot[] measure(StateVector psi, int shots, SplittableRandom random) {
        int n = psi.qubits();
        Map<Integer, double[]> cache = new HashMap<>();
        Shot[] out = new Shot[shots];
        for (int s = 0; s < shots; s++) {
            int[] basis = new int[n];
            int key = 0;
            for (int q = 0; q < n; q++) {
                basis[q] = random.nextInt(3);
                key = key * 3 + basis[q];
            }
            double[] p = cache.computeIfAbsent(key, k -> psi.basisProbabilities(basis));
            out[s] = new Shot(basis, StateVector.sample(p, 1, random)[0]);
        }
        return out;
    }

    private static int letter(PauliHamiltonian.Term t, int q) {
        boolean x = ((t.x() >> q) & 1) == 1;
        boolean z = ((t.z() >> q) & 1) == 1;
        return x && z ? 2 : x ? 1 : z ? 0 : -1;
    }

    private static double snapshotValue(PauliHamiltonian.Term t, Shot shot) {
        double v = 1;
        for (int q = 0; q < shot.basis().length; q++) {
            int l = letter(t, q);
            if (l < 0) {
                continue;
            }
            if (shot.basis()[q] != l) {
                return 0;
            }
            v *= ((shot.outcome() >> q) & 1) == 1 ? -3 : 3;
        }
        return v;
    }

    /**
     * The classical-shadow estimate of a Pauli observable by the median of means.
     *
     * @param observable the Pauli string (its coefficient is ignored)
     * @param shots the records
     * @param groups the number of groups for the median of means (1 = plain mean)
     * @return the estimate of {@code <P>}
     */
    public static double shadowEstimate(PauliHamiltonian.Term observable, Shot[] shots, int groups) {
        double[] means = new double[groups];
        int per = shots.length / groups;
        for (int g = 0; g < groups; g++) {
            double s = 0;
            for (int i = g * per; i < (g + 1) * per; i++) {
                s += snapshotValue(observable, shots[i]);
            }
            means[g] = s / per;
        }
        Arrays.sort(means);
        return groups % 2 == 1 ? means[groups / 2] : (means[groups / 2 - 1] + means[groups / 2]) / 2;
    }

    /**
     * The naive protocol: each observable measured in its own basis (Z on idle qubits) with
     * {@code shots / observables.size()} shots.
     *
     * @param psi the state
     * @param observables the Pauli strings
     * @param shots the total shot budget
     * @param random the random source
     * @return one estimate per observable
     */
    public static double[] directEstimate(StateVector psi, List<PauliHamiltonian.Term> observables, int shots,
            SplittableRandom random) {
        int n = psi.qubits();
        int per = Math.max(1, shots / observables.size());
        double[] out = new double[observables.size()];
        for (int k = 0; k < out.length; k++) {
            PauliHamiltonian.Term t = observables.get(k);
            int[] basis = new int[n];
            for (int q = 0; q < n; q++) {
                basis[q] = Math.max(0, letter(t, q));
            }
            int support = t.x() | t.z();
            int[] s = StateVector.sample(psi.basisProbabilities(basis), per, random);
            double sum = 0;
            for (int b : s) {
                sum += (Integer.bitCount(b & support) & 1) == 1 ? -1 : 1;
            }
            out[k] = sum / per;
        }
        return out;
    }

    /**
     * All Pauli strings of weight 1 and 2.
     *
     * @param n qubits
     * @return the strings
     */
    public static List<PauliHamiltonian.Term> localPaulis(int n) {
        List<PauliHamiltonian.Term> out = new java.util.ArrayList<>();
        String letters = "XYZ";
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < 3; a++) {
                out.add(PauliHamiltonian.term(1, String.valueOf(letters.charAt(a)), i));
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        out.add(PauliHamiltonian.term(1, "" + letters.charAt(a) + letters.charAt(b), i, j));
                    }
                }
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ reconstructions

    /**
     * Linear inversion over all {@code 4^n} Pauli strings.
     *
     * @param n qubits (at most about 6)
     * @param shots the records
     * @return the (possibly unphysical) density matrix
     */
    public static Density linearInversion(int n, Shot[] shots) {
        int d = 1 << n;
        double[][] re = new double[d][d];
        double[][] im = new double[d][d];
        for (int x = 0; x < d; x++) {
            for (int z = 0; z < d; z++) {
                PauliHamiltonian.Term t = new PauliHamiltonian.Term(1, x, z);
                double e = 1;
                if (x != 0 || z != 0) {
                    double s = 0;
                    for (Shot shot : shots) {
                        s += snapshotValue(t, shot);
                    }
                    e = s / shots.length;
                }
                if (e == 0) {
                    continue;
                }
                // add e P / d: P|b> = i^ny (-1)^(b.z) |b ^ x>, so P_(b^x, b) = that phase
                int ny = Integer.bitCount(x & z) & 3;
                for (int b = 0; b < d; b++) {
                    int k = (ny + 2 * (Integer.bitCount(b & z) & 1)) & 3;
                    double pr = k == 0 ? 1 : k == 2 ? -1 : 0;
                    double pi = k == 1 ? 1 : k == 3 ? -1 : 0;
                    re[b ^ x][b] += e * pr / d;
                    im[b ^ x][b] += e * pi / d;
                }
            }
        }
        return new Density(re, im);
    }

    private static double[][] localVector(int basis, int bit) {
        // eigenvector of the basis with eigenvalue (-1)^bit, as {re, im} of length 2
        double f = Math.sqrt(0.5);
        return switch (basis) {
            case 0 -> bit == 0 ? new double[][] {{1, 0}, {0, 0}} : new double[][] {{0, 1}, {0, 0}};
            case 1 -> bit == 0 ? new double[][] {{f, f}, {0, 0}} : new double[][] {{f, -f}, {0, 0}};
            default -> bit == 0 ? new double[][] {{f, 0}, {0, f}} : new double[][] {{f, 0}, {0, -f}};
        };
    }

    private static double[][] productVector(int[] basis, int outcome) {
        int n = basis.length;
        int d = 1 << n;
        double[] re = new double[d];
        double[] im = new double[d];
        for (int b = 0; b < d; b++) {
            double r = 1;
            double i = 0;
            for (int q = 0; q < n; q++) {
                double[][] v = localVector(basis[q], (outcome >> q) & 1);
                int bit = (b >> q) & 1;
                double vr = v[0][bit];
                double vi = v[1][bit];
                double nr = r * vr - i * vi;
                i = r * vi + i * vr;
                r = nr;
            }
            re[b] = r;
            im[b] = i;
        }
        return new double[][] {re, im};
    }

    /**
     * Maximum likelihood by the {@code R rho R} iteration, from the maximally mixed state.
     *
     * @param n qubits
     * @param shots the records
     * @param iterations the iterations
     * @return a positive density matrix
     */
    public static Density maximumLikelihood(int n, Shot[] shots, int iterations) {
        int d = 1 << n;
        Map<String, double[]> counts = new HashMap<>();
        Map<String, double[][]> vectors = new HashMap<>();
        for (Shot s : shots) {
            String key = Arrays.toString(s.basis()) + ":" + s.outcome();
            counts.computeIfAbsent(key, k -> new double[1])[0]++;
            vectors.computeIfAbsent(key, k -> productVector(s.basis(), s.outcome()));
        }
        List<String> keys = List.copyOf(counts.keySet());
        double[][] re = new double[d][d];
        double[][] im = new double[d][d];
        for (int a = 0; a < d; a++) {
            re[a][a] = 1.0 / d;
        }
        for (int it = 0; it < iterations; it++) {
            double[][] rr = new double[d][d];
            double[][] ri = new double[d][d];
            for (String key : keys) {
                double[][] v = vectors.get(key);
                // p = <v|rho|v>
                double p = 0;
                for (int a = 0; a < d; a++) {
                    double sr = 0;
                    double si = 0;
                    for (int b = 0; b < d; b++) {
                        sr += re[a][b] * v[0][b] - im[a][b] * v[1][b];
                        si += re[a][b] * v[1][b] + im[a][b] * v[0][b];
                    }
                    p += v[0][a] * sr + v[1][a] * si;
                }
                double w = counts.get(key)[0] / shots.length / Math.max(p, 1e-12);
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        // |v><v|_ab = v_a conj(v_b)
                        rr[a][b] += w * (v[0][a] * v[0][b] + v[1][a] * v[1][b]);
                        ri[a][b] += w * (v[1][a] * v[0][b] - v[0][a] * v[1][b]);
                    }
                }
            }
            double[][][] t = multiply(rr, ri, re, im);
            double[][][] u = multiply(t[0], t[1], rr, ri);
            double tr = 0;
            for (int a = 0; a < d; a++) {
                tr += u[0][a][a];
            }
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    re[a][b] = u[0][a][b] / tr;
                    im[a][b] = u[1][a][b] / tr;
                }
            }
        }
        return new Density(re, im);
    }

    private static double[][][] multiply(double[][] ar, double[][] ai, double[][] br, double[][] bi) {
        int d = ar.length;
        double[][] cr = new double[d][d];
        double[][] ci = new double[d][d];
        for (int a = 0; a < d; a++) {
            for (int k = 0; k < d; k++) {
                double xr = ar[a][k];
                double xi = ai[a][k];
                if (xr == 0 && xi == 0) {
                    continue;
                }
                for (int b = 0; b < d; b++) {
                    cr[a][b] += xr * br[k][b] - xi * bi[k][b];
                    ci[a][b] += xr * bi[k][b] + xi * br[k][b];
                }
            }
        }
        return new double[][][] {cr, ci};
    }

    /**
     * Neural-network tomography of a pure state with RBM amplitude and phase networks.
     *
     * @param n qubits
     * @param shots the records
     * @param alpha hidden units per qubit of each RBM
     * @param steps Adam steps
     * @param batch shots per step
     * @param learningRate the Adam step
     * @param seed the seed
     * @return the reconstructed state
     */
    public static StateVector neuralRbm(int n, Shot[] shots, int alpha, int steps, int batch, double learningRate,
            long seed) {
        NeuralQuantumState amp = NeuralQuantumState.rbm(n, alpha, seed);
        NeuralQuantumState phase = NeuralQuantumState.rbm(n, alpha, seed + 1);
        int pa = amp.parameterCount();
        int pp = phase.parameterCount();
        int d = 1 << n;
        // O-derivatives are fixed per configuration only through the parameters, so recompute each step
        double[][] oa = new double[d][pa];
        double[][] op = new double[d][pp];
        SplittableRandom r = new SplittableRandom(seed);
        ParameterObjective nll = (x, g) -> {
            System.arraycopy(x, 0, amp.parameters(), 0, pa);
            System.arraycopy(x, pa, phase.parameters(), 0, pp);
            double[] lr = new double[d];
            double max = Double.NEGATIVE_INFINITY;
            for (int b = 0; b < d; b++) {
                lr[b] = amp.logPsi(b);
                max = Math.max(max, lr[b]);
                amp.logDerivatives(b, oa[b]);
                phase.logDerivatives(b, op[b]);
            }
            double[] pr = new double[d];
            double[] pi = new double[d];
            double z = 0;
            for (int b = 0; b < d; b++) {
                double mod = Math.exp(lr[b] - max);
                double ph = phase.logPsi(b);
                pr[b] = mod * Math.cos(ph);
                pi[b] = mod * Math.sin(ph);
                z += mod * mod;
            }
            double[] dA = new double[d];
            double[] dP = new double[d];
            double loss = 0;
            for (int k = 0; k < batch; k++) {
                Shot s = shots[r.nextInt(shots.length)];
                double[][] v = productVector(s.basis(), s.outcome());
                // a = <v|psi>
                double ar = 0;
                double ai = 0;
                for (int b = 0; b < d; b++) {
                    ar += v[0][b] * pr[b] + v[1][b] * pi[b];
                    ai += v[0][b] * pi[b] - v[1][b] * pr[b];
                }
                double a2 = Math.max(ar * ar + ai * ai, 1e-300);
                loss -= Math.log(a2 / z);
                for (int b = 0; b < d; b++) {
                    // c = conj(a) conj(v_b) psi_b
                    double ur = v[0][b] * pr[b] + v[1][b] * pi[b];
                    double ui = v[0][b] * pi[b] - v[1][b] * pr[b];
                    double cr = ar * ur + ai * ui;
                    double ci = ar * ui - ai * ur;
                    dA[b] += -2 * cr / a2;
                    dP[b] += 2 * ci / a2;
                }
            }
            for (int b = 0; b < d; b++) {
                double p = (pr[b] * pr[b] + pi[b] * pi[b]) / z;
                dA[b] = dA[b] / batch + 2 * p;
                dP[b] /= batch;
            }
            Arrays.fill(g, 0);
            for (int b = 0; b < d; b++) {
                for (int k = 0; k < pa; k++) {
                    g[k] += dA[b] * oa[b][k];
                }
                for (int k = 0; k < pp; k++) {
                    g[pa + k] += dP[b] * op[b][k];
                }
            }
            return loss / batch;
        };
        double[] x0 = new double[pa + pp];
        System.arraycopy(amp.parameters(), 0, x0, 0, pa);
        System.arraycopy(phase.parameters(), 0, x0, pa, pp);
        double[] x = nll.minimize(Optimizer.adam((float) learningRate), x0, steps, null);
        System.arraycopy(x, 0, amp.parameters(), 0, pa);
        System.arraycopy(x, pa, phase.parameters(), 0, pp);
        double[] re = new double[d];
        double[] im = new double[d];
        double max = Double.NEGATIVE_INFINITY;
        for (int b = 0; b < d; b++) {
            max = Math.max(max, amp.logPsi(b));
        }
        for (int b = 0; b < d; b++) {
            double mod = Math.exp(amp.logPsi(b) - max);
            re[b] = mod * Math.cos(phase.logPsi(b));
            im[b] = mod * Math.sin(phase.logPsi(b));
        }
        return StateVector.of(re, im);
    }
}
