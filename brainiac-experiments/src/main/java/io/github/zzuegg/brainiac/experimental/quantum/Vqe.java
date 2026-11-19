package io.github.zzuegg.brainiac.experimental.quantum;

import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * The variational quantum eigensolver: minimise {@code E(theta) = <psi(theta)|H|psi(theta)>}
 * over the angles of a parametrised circuit (Peruzzo et al., "A variational eigenvalue solver
 * on a photonic quantum processor", <a href="https://arxiv.org/abs/1304.3061">arXiv:1304.3061</a>).
 *
 * <p>Ansätze:
 * <ul>
 *   <li>{@link Ansatz#HARDWARE_EFFICIENT} (Kandala et al., Nature 2017): layers of RY and RZ on
 *       every qubit followed by a CNOT ring; expressive but problem-agnostic, and its gradients
 *       concentrate (barren plateaus) as depth and width grow.</li>
 *   <li>{@link Ansatz#HAMILTONIAN_VARIATIONAL} (Wiersema et al., PRX Quantum 2020): from the
 *       ground state of an easy part of H, alternate exponentials of the Hamiltonian's
 *       non-commuting term groups, one angle per group per layer; it respects H's symmetries and
 *       needs few parameters. Built here for the TFIM (ZZ and X groups from {@code |+...+>}) and
 *       for the XXZ chain (even/odd bond ZZ and XX+YY groups from singlets on even bonds).</li>
 *   <li>{@link Ansatz#PRODUCT}: one RY/RZ layer and no entanglers, the mean-field baseline.</li>
 * </ul>
 *
 * <p>Optimizers: core Adam through {@link ParameterObjective}, core L-BFGS, and SPSA on
 * shot-noise energy estimates, the hardware-realistic setting.
 */
public final class Vqe {

    /** The circuit family. */
    public enum Ansatz {
        /** RY/RZ layers with CNOT rings. */
        HARDWARE_EFFICIENT,
        /** Exponentials of the Hamiltonian's term groups. */
        HAMILTONIAN_VARIATIONAL,
        /** A product state (mean field). */
        PRODUCT
    }

    /** The optimizer. */
    public enum Method {
        /** Core Adam on exact adjoint gradients. */
        ADAM,
        /** Core L-BFGS on exact adjoint gradients. */
        LBFGS,
        /** Simultaneous-perturbation stochastic approximation on shot-noise energies. */
        SPSA
    }

    /**
     * An optimisation result.
     *
     * @param energy the exact energy at the final parameters
     * @param parameters the final parameters
     * @param evaluations energy evaluations (each a circuit run; an adjoint gradient counts 3)
     */
    public record Result(double energy, double[] parameters, long evaluations) {
    }

    private final QuantumCircuit circuit;
    private final PauliHamiltonian hamiltonian;
    private long evaluations;

    /**
     * @param circuit the ansatz
     * @param hamiltonian the Hamiltonian
     */
    public Vqe(QuantumCircuit circuit, PauliHamiltonian hamiltonian) {
        this.circuit = circuit;
        this.hamiltonian = hamiltonian;
    }

    /** @return the ansatz */
    public QuantumCircuit circuit() {
        return circuit;
    }

    // ------------------------------------------------------------------ ansätze

    /**
     * The hardware-efficient ansatz.
     *
     * @param n qubits
     * @param layers entangling layers (a final rotation layer is added)
     * @return the circuit
     */
    public static QuantumCircuit hardwareEfficient(int n, int layers) {
        QuantumCircuit c = new QuantumCircuit(n);
        for (int l = 0; l <= layers; l++) {
            for (int q = 0; q < n; q++) {
                c.rotation(1 << q, 1 << q, 0.5, c.newParameter());
                c.rotation(0, 1 << q, 0.5, c.newParameter());
            }
            if (l < layers) {
                for (int q = 0; q + 1 < n; q++) {
                    c.cnot(q, q + 1);
                }
                if (n > 2) {
                    c.cnot(n - 1, 0);
                }
            }
        }
        return c;
    }

    /**
     * The product (mean-field) ansatz.
     *
     * @param n qubits
     * @return the circuit
     */
    public static QuantumCircuit product(int n) {
        return hardwareEfficient(n, 0);
    }

    /**
     * The Hamiltonian variational ansatz for a TFIM: from {@code |+...+>}, per layer
     * {@code exp(-i a/2 sum ZZ)} then {@code exp(-i b/2 sum X)}.
     *
     * @param tfim the Ising Hamiltonian (its ZZ and X terms are used)
     * @param layers the layers
     * @return the circuit
     */
    public static QuantumCircuit tfimVariational(PauliHamiltonian tfim, int layers) {
        int n = tfim.qubits();
        QuantumCircuit c = new QuantumCircuit(n);
        for (int q = 0; q < n; q++) {
            c.h(q);
        }
        for (int l = 0; l < layers; l++) {
            int a = c.newParameter();
            int b = c.newParameter();
            for (PauliHamiltonian.Term t : tfim.terms()) {
                if (t.x() == 0) {
                    c.rotation(0, t.z(), 0.5 * Math.signum(t.coefficient()), a);
                }
            }
            for (PauliHamiltonian.Term t : tfim.terms()) {
                if (t.x() != 0) {
                    c.rotation(t.x(), t.z(), 0.5 * Math.signum(t.coefficient()), b);
                }
            }
        }
        return c;
    }

    /**
     * The Hamiltonian variational ansatz for an XXZ chain: singlets on bonds (0,1), (2,3), ...,
     * then per layer the ZZ and the XX+YY exponentials of the odd bonds and then of the even
     * bonds, four angles per layer.
     *
     * @param n sites (even)
     * @param periodic whether the chain is a ring
     * @param layers the layers
     * @return the circuit
     */
    public static QuantumCircuit xxzVariational(int n, boolean periodic, int layers) {
        QuantumCircuit c = new QuantumCircuit(n);
        for (int q = 0; q + 1 < n; q += 2) {
            // |00> -> (|01> - |10>) / sqrt 2
            c.h(q).cnot(q, q + 1).pauli(1 << (q + 1), 0).pauli(0, 1 << q);
        }
        for (int l = 0; l < layers; l++) {
            for (int parity : new int[] {1, 0}) {
                int zz = c.newParameter();
                int xy = c.newParameter();
                for (int i = parity; i < (periodic ? n : n - 1); i += 2) {
                    int m = (1 << i) | (1 << ((i + 1) % n));
                    c.rotation(0, m, 0.5, zz);
                }
                for (int i = parity; i < (periodic ? n : n - 1); i += 2) {
                    int m = (1 << i) | (1 << ((i + 1) % n));
                    c.rotation(m, 0, 0.5, xy);
                    c.rotation(m, m, 0.5, xy);
                }
            }
        }
        return c;
    }

    // ------------------------------------------------------------------ energies

    /**
     * The exact energy.
     *
     * @param theta the parameters
     * @return the energy
     */
    public double energy(double[] theta) {
        evaluations++;
        return circuit.energy(hamiltonian, theta, null);
    }

    /**
     * The exact energy and its adjoint gradient.
     *
     * @param theta the parameters
     * @param gradient receives the gradient
     * @return the energy
     */
    public double energyAndGradient(double[] theta, double[] gradient) {
        evaluations += 3;
        return circuit.energy(hamiltonian, theta, gradient);
    }

    /**
     * A finite-shot energy estimate.
     *
     * @param theta the parameters
     * @param shots shots per Pauli term
     * @param random the noise source
     * @return the estimate
     */
    public double shotEnergy(double[] theta, int shots, SplittableRandom random) {
        evaluations++;
        return hamiltonian.estimate(circuit.run(theta), shots, random);
    }

    /**
     * The fidelity of the prepared state with a reference.
     *
     * @param theta the parameters
     * @param reference the reference state
     * @return {@code |<ref|psi>|^2}
     */
    public double fidelity(double[] theta, StateVector reference) {
        return circuit.run(theta).fidelity(reference);
    }

    /** @return energy evaluations so far */
    public long evaluations() {
        return evaluations;
    }

    /**
     * Small random starting angles, {@code N(0, scale^2)}.
     *
     * @param scale the spread
     * @param seed the seed
     * @return the angles
     */
    public double[] initial(double scale, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] t = new double[circuit.parameterCount()];
        for (int i = 0; i < t.length; i++) {
            t[i] = scale * r.nextGaussian();
        }
        return t;
    }

    // ------------------------------------------------------------------ optimisation

    /**
     * Optimises from a start.
     *
     * @param method the optimizer
     * @param theta0 the start
     * @param iterations steps (Adam, SPSA) or iterations (L-BFGS)
     * @param learningRate Adam's step, or SPSA's {@code a}
     * @param shots SPSA's shots per term
     * @param seed SPSA's seed
     * @return the result
     */
    public Result optimize(Method method, double[] theta0, int iterations, double learningRate, int shots,
            long seed) {
        ParameterObjective f = this::energyAndGradient;
        double[] theta = switch (method) {
            case ADAM -> f.minimize(Optimizer.adam((float) learningRate), theta0, iterations, null);
            case LBFGS -> f.minimizeLbfgs(theta0, iterations);
            case SPSA -> spsa(theta0, iterations, learningRate, 0.1, shots, seed);
        };
        long used = evaluations;
        return new Result(circuit.energy(hamiltonian, theta, null), theta, used);
    }

    private double[] spsa(double[] theta0, int iterations, double a, double c, int shots, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] t = theta0.clone();
        int n = t.length;
        double bigA = 0.1 * iterations;
        double[] plus = new double[n];
        double[] minus = new double[n];
        double[] delta = new double[n];
        for (int k = 0; k < iterations; k++) {
            double ak = a / Math.pow(k + 1 + bigA, 0.602);
            double ck = c / Math.pow(k + 1, 0.101);
            for (int i = 0; i < n; i++) {
                delta[i] = r.nextBoolean() ? 1 : -1;
                plus[i] = t[i] + ck * delta[i];
                minus[i] = t[i] - ck * delta[i];
            }
            double d = (shotEnergy(plus, shots, r) - shotEnergy(minus, shots, r)) / (2 * ck);
            for (int i = 0; i < n; i++) {
                t[i] -= ak * d * delta[i];
            }
        }
        return t;
    }

    /**
     * The variance of {@code dE/dtheta_k} over uniformly random angles in {@code [0, 2 pi)}, the
     * barren-plateau diagnostic.
     *
     * @param parameter which parameter's derivative
     * @param samples random draws
     * @param seed the seed
     * @return the variance
     */
    public double gradientVariance(int parameter, int samples, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] g = new double[circuit.parameterCount()];
        double s = 0;
        double s2 = 0;
        for (int i = 0; i < samples; i++) {
            double[] t = new double[g.length];
            for (int k = 0; k < t.length; k++) {
                t[k] = 2 * Math.PI * r.nextDouble();
            }
            circuit.energy(hamiltonian, t, g);
            s += g[parameter];
            s2 += g[parameter] * g[parameter];
        }
        double m = s / samples;
        return s2 / samples - m * m;
    }

    /**
     * The best of several L-BFGS restarts from small random angles.
     *
     * @param restarts the restarts
     * @param iterations L-BFGS iterations each
     * @param scale the initial-angle spread
     * @param seed the seed
     * @return the best result
     */
    public Result bestOf(int restarts, int iterations, double scale, long seed) {
        Result best = null;
        for (int k = 0; k < restarts; k++) {
            Result res = optimize(Method.LBFGS, initial(scale, seed + 31L * k), iterations, 0, 0, 0);
            if (best == null || res.energy() < best.energy()) {
                best = res;
            }
        }
        return new Result(best.energy(), best.parameters(), evaluations);
    }
}
