package io.github.zzuegg.brainiac.experimental.quantum;

import java.util.SplittableRandom;

/**
 * An exact state vector of {@code n} qubits, {@code 2^n} complex amplitudes in double
 * precision, with the gates a variational circuit needs. Qubit {@code q} is bit {@code q} of
 * the basis index (little-endian), and bit 0 is the {@code +1} eigenstate of Z, the same
 * conventions as {@link io.github.zzuegg.brainiac.experimental.nn.QuantumCircuitLayer}.
 *
 * <p>Every parametrised gate is a <em>Pauli rotation</em> {@code exp(-i a P)} of a Pauli string
 * {@code P}, given by an X mask and a Z mask (a Y is a bit set in both): {@code RY(t)} on qubit
 * {@code q} is the rotation of {@code Y_q} by {@code a = t / 2}. Since {@code P^2 = I}, the
 * rotation is {@code cos(a) - i sin(a) P}, and a Pauli string acts on a basis state as
 * {@code P |b> = i^(#Y) (-1)^(popcount(b & z)) |b ^ x>}. This is what makes the adjoint
 * gradient in {@link QuantumCircuit} one line per gate.
 */
public final class StateVector {

    private final int qubits;
    private final double[] re;
    private final double[] im;

    /**
     * The all-zero state {@code |0...0>}.
     *
     * @param qubits the number of qubits, at most 24
     */
    public StateVector(int qubits) {
        if (qubits < 1 || qubits > 24) {
            throw new IllegalArgumentException("qubits must be in 1..24, not " + qubits);
        }
        this.qubits = qubits;
        this.re = new double[1 << qubits];
        this.im = new double[1 << qubits];
        re[0] = 1;
    }

    private StateVector(int qubits, double[] re, double[] im) {
        this.qubits = qubits;
        this.re = re;
        this.im = im;
    }

    /**
     * A state from its amplitudes (copied and normalised).
     *
     * @param re real parts, of length a power of two
     * @param im imaginary parts, or null for a real state
     * @return the state
     */
    public static StateVector of(double[] re, double[] im) {
        int n = Integer.numberOfTrailingZeros(re.length);
        if (1 << n != re.length) {
            throw new IllegalArgumentException("length must be a power of two");
        }
        StateVector s = new StateVector(n, re.clone(), im == null ? new double[re.length] : im.clone());
        s.normalize();
        return s;
    }

    /** @return an independent copy */
    public StateVector copy() {
        return new StateVector(qubits, re.clone(), im.clone());
    }

    /** @return the number of qubits */
    public int qubits() {
        return qubits;
    }

    /** @return {@code 2^qubits} */
    public int dimension() {
        return re.length;
    }

    /** @return the real parts (a live view) */
    public double[] real() {
        return re;
    }

    /** @return the imaginary parts (a live view) */
    public double[] imag() {
        return im;
    }

    /** Rescales to unit norm. */
    public void normalize() {
        double s = 0;
        for (int i = 0; i < re.length; i++) {
            s += re[i] * re[i] + im[i] * im[i];
        }
        double f = 1 / Math.sqrt(s);
        for (int i = 0; i < re.length; i++) {
            re[i] *= f;
            im[i] *= f;
        }
    }

    // ------------------------------------------------------------------ Pauli strings

    /**
     * Applies a Pauli string in place: {@code psi <- P psi}.
     *
     * @param x the X mask (bits with X or Y)
     * @param z the Z mask (bits with Z or Y)
     */
    public void applyPauli(int x, int z) {
        int ny = Integer.bitCount(x & z) & 3;
        if (x == 0) {
            for (int b = 0; b < re.length; b++) {
                if ((Integer.bitCount(b & z) & 1) == 1) {
                    re[b] = -re[b];
                    im[b] = -im[b];
                }
            }
            return;
        }
        for (int b = 0; b < re.length; b++) {
            int c = b ^ x;
            if (c < b) {
                continue;
            }
            // new[c] = f(b) old[b], new[b] = f(c) old[c]
            int kb = (ny + 2 * (Integer.bitCount(b & z) & 1)) & 3;
            int kc = (ny + 2 * (Integer.bitCount(c & z) & 1)) & 3;
            double br = re[b];
            double bi = im[b];
            double cr = re[c];
            double ci = im[c];
            re[c] = phaseRe(kb, br, bi);
            im[c] = phaseIm(kb, br, bi);
            re[b] = phaseRe(kc, cr, ci);
            im[b] = phaseIm(kc, cr, ci);
        }
    }

    private static double phaseRe(int k, double r, double i) {
        return switch (k) {
            case 0 -> r;
            case 1 -> -i;
            case 2 -> -r;
            default -> i;
        };
    }

    private static double phaseIm(int k, double r, double i) {
        return switch (k) {
            case 0 -> i;
            case 1 -> r;
            case 2 -> -i;
            default -> -r;
        };
    }

    /**
     * Applies {@code exp(-i angle P)} in place.
     *
     * @param x the X mask
     * @param z the Z mask
     * @param angle the rotation angle {@code a}
     */
    public void pauliRotation(int x, int z, double angle) {
        double c = Math.cos(angle);
        double s = Math.sin(angle);
        if (x == 0) {
            // diagonal: phase e^(-i a (+-1))
            for (int b = 0; b < re.length; b++) {
                double sign = (Integer.bitCount(b & z) & 1) == 1 ? -1 : 1;
                double r = re[b];
                double i = im[b];
                re[b] = c * r + sign * s * i;
                im[b] = c * i - sign * s * r;
            }
            return;
        }
        StateVector p = copy();
        p.applyPauli(x, z);
        for (int b = 0; b < re.length; b++) {
            // (c - i s P) psi
            re[b] = c * re[b] + s * p.im[b];
            im[b] = c * im[b] - s * p.re[b];
        }
    }

    /**
     * Multiplies every amplitude by {@code exp(-i scale * phase[b])}: a diagonal unitary, such
     * as QAOA's cost layer {@code exp(-i gamma C)}.
     *
     * @param phase the diagonal, one value per basis state
     * @param scale the factor
     */
    public void diagonalPhase(double[] phase, double scale) {
        for (int b = 0; b < re.length; b++) {
            double a = scale * phase[b];
            double c = Math.cos(a);
            double s = Math.sin(a);
            double r = re[b];
            double i = im[b];
            re[b] = c * r + s * i;
            im[b] = c * i - s * r;
        }
    }

    /**
     * The expectation of a Pauli string, {@code <psi|P|psi>} (real since P is Hermitian).
     *
     * @param x the X mask
     * @param z the Z mask
     * @return the expectation
     */
    public double expectation(int x, int z) {
        StateVector p = copy();
        p.applyPauli(x, z);
        return inner(p)[0];
    }

    // ------------------------------------------------------------------ fixed gates

    /**
     * Hadamard.
     *
     * @param q the qubit
     */
    public void h(int q) {
        int m = 1 << q;
        double f = Math.sqrt(0.5);
        for (int b = 0; b < re.length; b++) {
            if ((b & m) == 0) {
                int c = b | m;
                double ar = re[b];
                double ai = im[b];
                double br = re[c];
                double bi = im[c];
                re[b] = f * (ar + br);
                im[b] = f * (ai + bi);
                re[c] = f * (ar - br);
                im[c] = f * (ai - bi);
            }
        }
    }

    /**
     * Controlled NOT.
     *
     * @param control the control qubit
     * @param target the target qubit
     */
    public void cnot(int control, int target) {
        int cm = 1 << control;
        int tm = 1 << target;
        for (int b = 0; b < re.length; b++) {
            if ((b & cm) != 0 && (b & tm) == 0) {
                int c = b | tm;
                double r = re[b];
                double i = im[b];
                re[b] = re[c];
                im[b] = im[c];
                re[c] = r;
                im[c] = i;
            }
        }
    }

    /**
     * {@code RX(theta) = exp(-i theta X / 2)}.
     *
     * @param q the qubit
     * @param theta the angle
     */
    public void rx(int q, double theta) {
        pauliRotation(1 << q, 0, theta / 2);
    }

    /**
     * {@code RY(theta) = exp(-i theta Y / 2)}.
     *
     * @param q the qubit
     * @param theta the angle
     */
    public void ry(int q, double theta) {
        pauliRotation(1 << q, 1 << q, theta / 2);
    }

    /**
     * {@code RZ(theta) = exp(-i theta Z / 2)}.
     *
     * @param q the qubit
     * @param theta the angle
     */
    public void rz(int q, double theta) {
        pauliRotation(0, 1 << q, theta / 2);
    }

    // ------------------------------------------------------------------ overlaps and measurement

    /**
     * The inner product {@code <this|other>}.
     *
     * @param other the other state
     * @return {real, imaginary}
     */
    public double[] inner(StateVector other) {
        double r = 0;
        double i = 0;
        for (int b = 0; b < re.length; b++) {
            r += re[b] * other.re[b] + im[b] * other.im[b];
            i += re[b] * other.im[b] - im[b] * other.re[b];
        }
        return new double[] {r, i};
    }

    /**
     * The fidelity {@code |<this|other>|^2} of two pure states.
     *
     * @param other the other state
     * @return the fidelity in [0, 1]
     */
    public double fidelity(StateVector other) {
        double[] z = inner(other);
        return z[0] * z[0] + z[1] * z[1];
    }

    /** @return the probability of every computational basis state */
    public double[] probabilities() {
        double[] p = new double[re.length];
        for (int b = 0; b < p.length; b++) {
            p[b] = re[b] * re[b] + im[b] * im[b];
        }
        return p;
    }

    /**
     * The outcome probabilities when every qubit is measured in its own Pauli basis.
     *
     * @param basis per qubit 0 = Z, 1 = X, 2 = Y
     * @return the probabilities of the outcome bits (bit {@code q} = 1 means eigenvalue -1)
     */
    public double[] basisProbabilities(int[] basis) {
        StateVector s = copy();
        for (int q = 0; q < qubits; q++) {
            if (basis[q] == 1) {
                s.h(q);
            } else if (basis[q] == 2) {
                // S-dagger then H maps the Y eigenbasis to the Z eigenbasis
                s.pauliRotation(0, 1 << q, -Math.PI / 4);
                s.h(q);
            }
        }
        return s.probabilities();
    }

    /**
     * Draws measurement outcomes from a probability vector.
     *
     * @param probabilities the outcome probabilities
     * @param shots how many outcomes
     * @param random the random source
     * @return the outcomes
     */
    public static int[] sample(double[] probabilities, int shots, SplittableRandom random) {
        double[] cdf = new double[probabilities.length];
        double acc = 0;
        for (int i = 0; i < cdf.length; i++) {
            acc += probabilities[i];
            cdf[i] = acc;
        }
        int[] out = new int[shots];
        for (int s = 0; s < shots; s++) {
            double u = random.nextDouble() * acc;
            int lo = 0;
            int hi = cdf.length - 1;
            while (lo < hi) {
                int mid = (lo + hi) >>> 1;
                if (cdf[mid] < u) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            out[s] = lo;
        }
        return out;
    }
}
