package io.github.zzuegg.brainiac.experimental.quantum;

import java.util.ArrayList;
import java.util.List;

/**
 * A parametrised quantum circuit: a list of fixed gates (Hadamard, CNOT, Pauli strings) and
 * Pauli rotations {@code exp(-i s theta_k P)} whose angle is a parameter {@code theta_k} times
 * a fixed scale {@code s}. Several rotations may share a parameter, as in the Hamiltonian
 * variational ansatz where one angle drives all terms of a commuting group.
 *
 * <p>The energy gradient comes from the <em>adjoint method</em> (Jones and Gacon 2020,
 * <a href="https://arxiv.org/abs/2009.02823">arXiv:2009.02823</a>): one forward pass, then a
 * backward sweep that un-applies each gate to the state and to {@code lambda = H psi}, with
 * {@code dE/dtheta_k += 2 s Im <lambda|P|psi>} at every rotation; its cost is a few circuit
 * runs regardless of the parameter count. The parameter-shift rule, what hardware uses, is
 * provided for comparison.
 */
public final class QuantumCircuit {

    private enum Kind { HADAMARD, CNOT, PAULI, ROTATION }

    private record Gate(Kind kind, int a, int b, double scale, int parameter) {
    }

    private final int qubits;
    private final List<Gate> gates = new ArrayList<>();
    private int parameters;

    /**
     * An empty circuit acting on {@code |0...0>}.
     *
     * @param qubits the number of qubits
     */
    public QuantumCircuit(int qubits) {
        this.qubits = qubits;
    }

    /** @return the number of qubits */
    public int qubits() {
        return qubits;
    }

    /** @return the number of parameters */
    public int parameterCount() {
        return parameters;
    }

    /** @return the number of gates */
    public int gateCount() {
        return gates.size();
    }

    /**
     * Allocates a new parameter.
     *
     * @return its index
     */
    public int newParameter() {
        return parameters++;
    }

    /**
     * Appends a Hadamard.
     *
     * @param q the qubit
     * @return this circuit
     */
    public QuantumCircuit h(int q) {
        gates.add(new Gate(Kind.HADAMARD, q, 0, 0, -1));
        return this;
    }

    /**
     * Appends a CNOT.
     *
     * @param control the control
     * @param target the target
     * @return this circuit
     */
    public QuantumCircuit cnot(int control, int target) {
        gates.add(new Gate(Kind.CNOT, control, target, 0, -1));
        return this;
    }

    /**
     * Appends a fixed Pauli string (X, Y or Z gates).
     *
     * @param x the X mask
     * @param z the Z mask
     * @return this circuit
     */
    public QuantumCircuit pauli(int x, int z) {
        gates.add(new Gate(Kind.PAULI, x, z, 0, -1));
        return this;
    }

    /**
     * Appends {@code exp(-i scale theta_parameter P)}.
     *
     * @param x the X mask
     * @param z the Z mask
     * @param scale the angle per unit parameter (1/2 for the usual RX/RY/RZ)
     * @param parameter the parameter index (from {@link #newParameter()})
     * @return this circuit
     */
    public QuantumCircuit rotation(int x, int z, double scale, int parameter) {
        if (parameter < 0 || parameter >= parameters) {
            throw new IllegalArgumentException("allocate the parameter first");
        }
        gates.add(new Gate(Kind.ROTATION, x, z, scale, parameter));
        return this;
    }

    private static void apply(StateVector s, Gate g, double[] theta, boolean inverse) {
        switch (g.kind()) {
            case HADAMARD -> s.h(g.a());
            case CNOT -> s.cnot(g.a(), g.b());
            case PAULI -> s.applyPauli(g.a(), g.b());
            case ROTATION -> s.pauliRotation(g.a(), g.b(), (inverse ? -1 : 1) * g.scale() * theta[g.parameter()]);
            default -> throw new IllegalStateException();
        }
    }

    /**
     * Runs the circuit on {@code |0...0>}.
     *
     * @param theta the parameters
     * @return the final state
     */
    public StateVector run(double[] theta) {
        StateVector s = new StateVector(qubits);
        for (Gate g : gates) {
            apply(s, g, theta, false);
        }
        return s;
    }

    /**
     * The energy and its exact gradient by the adjoint method.
     *
     * @param hamiltonian the observable
     * @param theta the parameters
     * @param gradient receives {@code dE/dtheta} (overwritten), or null
     * @return {@code <psi(theta)|H|psi(theta)>}
     */
    public double energy(PauliHamiltonian hamiltonian, double[] theta, double[] gradient) {
        StateVector psi = run(theta);
        StateVector lambda = hamiltonian.apply(psi);
        double e = psi.inner(lambda)[0];
        if (gradient == null) {
            return e;
        }
        java.util.Arrays.fill(gradient, 0);
        for (int k = gates.size() - 1; k >= 0; k--) {
            Gate g = gates.get(k);
            if (g.kind() == Kind.ROTATION) {
                StateVector p = psi.copy();
                p.applyPauli(g.a(), g.b());
                gradient[g.parameter()] += 2 * g.scale() * lambda.inner(p)[1];
            }
            apply(psi, g, theta, true);
            apply(lambda, g, theta, true);
        }
        return e;
    }

    /**
     * The gradient by the parameter-shift rule, valid when each parameter drives rotations of
     * scale 1/2 used once ({@code dE/dt = (E(t + pi/2) - E(t - pi/2)) / 2}); costs two energy
     * evaluations per parameter.
     *
     * @param hamiltonian the observable
     * @param theta the parameters
     * @return the gradient
     */
    public double[] parameterShift(PauliHamiltonian hamiltonian, double[] theta) {
        double[] g = new double[parameters];
        double[] t = theta.clone();
        for (int k = 0; k < parameters; k++) {
            t[k] = theta[k] + Math.PI / 2;
            double plus = hamiltonian.expectation(run(t));
            t[k] = theta[k] - Math.PI / 2;
            double minus = hamiltonian.expectation(run(t));
            t[k] = theta[k];
            g[k] = (plus - minus) / 2;
        }
        return g;
    }
}
