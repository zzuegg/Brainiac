package io.github.zzuegg.brainiac.experimental.quantum;

import io.github.zzuegg.brainiac.tensor.Matrices;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * A qubit Hamiltonian as a sparse sum of weighted Pauli strings,
 * {@code H = sum_t c_t P_t}, the form a quantum computer measures term by term and the form
 * spin models (transverse-field Ising, Heisenberg, MaxCut costs) are written in. It applies
 * itself to a {@link StateVector} without building a matrix, gives exact expectations or
 * shot-noise estimates, enumerates the non-zero matrix elements of a row (the "connected
 * configurations" a neural quantum state's local energy needs), and finds ground states of
 * real Hamiltonians with core {@link Matrices#lanczos}.
 */
public final class PauliHamiltonian {

    /**
     * One term {@code c P}.
     *
     * @param coefficient the real weight
     * @param x the X mask (bits with X or Y)
     * @param z the Z mask (bits with Z or Y)
     */
    public record Term(double coefficient, int x, int z) {

        /** @return the number of Y factors */
        public int yCount() {
            return Integer.bitCount(x & z);
        }

        /**
         * The matrix element {@code <b ^ x| P |b>}, real for an even number of Y factors.
         *
         * @param basis the column basis state {@code b}
         * @return {@code i^(#Y) (-1)^(popcount(b & z))}, as a real number when that is real
         */
        public double realElement(int basis) {
            int k = (yCount() + 2 * (Integer.bitCount(basis & z) & 1)) & 3;
            return k == 0 ? 1 : k == 2 ? -1 : Double.NaN;
        }
    }

    private final int qubits;
    private final List<Term> terms;
    private final Map<Integer, List<Term>> byFlip;

    /**
     * A Hamiltonian from its terms.
     *
     * @param qubits the number of qubits
     * @param terms the terms
     */
    public PauliHamiltonian(int qubits, List<Term> terms) {
        this.qubits = qubits;
        this.terms = List.copyOf(terms);
        Map<Integer, List<Term>> m = new LinkedHashMap<>();
        for (Term t : terms) {
            m.computeIfAbsent(t.x(), k -> new ArrayList<>()).add(t);
        }
        this.byFlip = m;
    }

    /**
     * Parses a term such as {@code ("XX", 0, 1)}: one Pauli letter (I, X, Y, Z) per site.
     *
     * @param coefficient the weight
     * @param letters the Pauli letters
     * @param sites the qubit of each letter
     * @return the term
     */
    public static Term term(double coefficient, String letters, int... sites) {
        int x = 0;
        int z = 0;
        for (int i = 0; i < sites.length; i++) {
            int m = 1 << sites[i];
            switch (letters.charAt(i)) {
                case 'X' -> x |= m;
                case 'Z' -> z |= m;
                case 'Y' -> {
                    x |= m;
                    z |= m;
                }
                case 'I' -> {
                }
                default -> throw new IllegalArgumentException("not a Pauli letter: " + letters.charAt(i));
            }
        }
        return new Term(coefficient, x, z);
    }

    /** @return the number of qubits */
    public int qubits() {
        return qubits;
    }

    /** @return the terms */
    public List<Term> terms() {
        return terms;
    }

    /** @return whether every term has an even number of Y factors (a real symmetric matrix) */
    public boolean isReal() {
        for (Term t : terms) {
            if ((t.yCount() & 1) == 1) {
                return false;
            }
        }
        return true;
    }

    /**
     * {@code H psi}.
     *
     * @param psi the state
     * @return a new (unnormalised) state
     */
    public StateVector apply(StateVector psi) {
        double[] r = new double[psi.dimension()];
        double[] i = new double[psi.dimension()];
        for (Term t : terms) {
            StateVector p = psi.copy();
            p.applyPauli(t.x(), t.z());
            double[] pr = p.real();
            double[] pi = p.imag();
            for (int b = 0; b < r.length; b++) {
                r[b] += t.coefficient() * pr[b];
                i[b] += t.coefficient() * pi[b];
            }
        }
        return rawState(r, i);
    }

    private static StateVector rawState(double[] r, double[] i) {
        StateVector s = new StateVector(Integer.numberOfTrailingZeros(r.length));
        System.arraycopy(r, 0, s.real(), 0, r.length);
        System.arraycopy(i, 0, s.imag(), 0, i.length);
        return s;
    }

    /**
     * The exact energy {@code <psi|H|psi>}.
     *
     * @param psi a normalised state
     * @return the energy
     */
    public double expectation(StateVector psi) {
        return psi.inner(apply(psi))[0];
    }

    /**
     * A finite-shot estimate of the energy: each term's expectation {@code e} is replaced by a
     * draw from the Gaussian approximation of its binomial estimate, variance
     * {@code (1 - e^2) / shots}, as when every term is measured {@code shots} times.
     *
     * @param psi a normalised state
     * @param shots the shots per term
     * @param random the noise source
     * @return the estimate
     */
    public double estimate(StateVector psi, int shots, SplittableRandom random) {
        double e = 0;
        for (Term t : terms) {
            double v = psi.expectation(t.x(), t.z());
            double sd = Math.sqrt(Math.max(0, 1 - v * v) / shots);
            e += t.coefficient() * (v + sd * random.nextGaussian());
        }
        return e;
    }

    /**
     * The non-zero entries of column {@code basis} of a real Hamiltonian: for every distinct
     * flip mask {@code x}, the row {@code basis ^ x} and the summed element.
     *
     * @param basis the column
     * @param rows receives the rows (length at least {@link #flipCount()})
     * @param values receives the elements
     * @return how many entries were written
     */
    public int connections(int basis, int[] rows, double[] values) {
        int k = 0;
        for (Map.Entry<Integer, List<Term>> e : byFlip.entrySet()) {
            double v = 0;
            for (Term t : e.getValue()) {
                v += t.coefficient() * t.realElement(basis);
            }
            if (v != 0) {
                rows[k] = basis ^ e.getKey();
                values[k] = v;
                k++;
            }
        }
        return k;
    }

    /** @return the number of distinct flip masks, the most entries a column has */
    public int flipCount() {
        return byFlip.size();
    }

    /**
     * The diagonal of a Hamiltonian made only of Z strings (such as a MaxCut cost).
     *
     * @return {@code H_bb} for every basis state
     */
    public double[] diagonal() {
        double[] d = new double[1 << qubits];
        for (Term t : terms) {
            if (t.x() != 0) {
                throw new IllegalStateException("not diagonal: a term flips qubits");
            }
            for (int b = 0; b < d.length; b++) {
                d[b] += t.coefficient() * ((Integer.bitCount(b & t.z()) & 1) == 1 ? -1 : 1);
            }
        }
        return d;
    }

    /** @return the action on real vectors, for a real Hamiltonian */
    public Matrices.Operator operator() {
        if (!isReal()) {
            throw new IllegalStateException("the Hamiltonian is not real");
        }
        int[] rows = new int[flipCount()];
        double[] vals = new double[flipCount()];
        return (in, out) -> {
            java.util.Arrays.fill(out, 0);
            for (int b = 0; b < in.length; b++) {
                if (in[b] == 0) {
                    continue;
                }
                int k = connections(b, rows, vals);
                for (int j = 0; j < k; j++) {
                    out[rows[j]] += vals[j] * in[b];
                }
            }
        };
    }

    /**
     * The ground state of a real Hamiltonian by core Lanczos (full reorthogonalisation).
     *
     * @param steps the Krylov dimension, such as 80
     * @param seed the starting-vector seed
     * @return the ground energy and its (real) state
     */
    public GroundState groundState(int steps, long seed) {
        int dim = 1 << qubits;
        Matrices.Eigenpairs e = Matrices.lanczos(operator(), dim, 1, false, Math.min(dim, steps), seed);
        return new GroundState(e.values()[0], StateVector.of(e.vectors()[0], null));
    }

    /**
     * A ground state.
     *
     * @param energy the ground energy
     * @param state the state
     */
    public record GroundState(double energy, StateVector state) {
    }
}
