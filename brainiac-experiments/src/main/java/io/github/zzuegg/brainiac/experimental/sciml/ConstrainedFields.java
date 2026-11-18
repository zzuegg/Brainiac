package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;

import java.util.List;

/**
 * Hard constraints by construction (Richter-Powell, Lipman and Chen, NeurIPS 2022, "Neural
 * conservation laws"): instead of penalising a violated constraint, parameterise the field so
 * it holds exactly for any weights.
 *
 * <ul>
 *   <li>{@link #streamFunction}: a 2-D velocity {@code (u, v) = (d psi/dy, -d psi/dx)} from a
 *       learned stream function {@code psi}, divergence-free for any {@code psi}.
 *   <li>{@link #conservationLaw}: a density and flux {@code (rho, J) = (d phi/dx, -d phi/dt)}
 *       from a learned potential on space-time, which satisfy the continuity equation
 *       {@code rho_t + J_x = 0} exactly (the space-time field is divergence-free).
 * </ul>
 *
 * <p>Both are {@link Layer}s whose output is the derivative of the inner network with respect
 * to the input, computed with {@code Var.gradients(..., createGraph = true)} so they train by
 * ordinary backpropagation and can be differentiated again (for example to measure the
 * divergence). {@link #divergence} gives that measurement for any 2-D field.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2210.01741">arXiv:2210.01741</a>.
 */
public final class ConstrainedFields {

    private ConstrainedFields() {
    }

    private static final class Curl implements Layer {
        private final Layer potential;
        private final int first;
        private final int second;

        Curl(Layer potential, int first, int second) {
            this.potential = potential;
            this.first = first;
            this.second = second;
        }

        @Override
        public Var forward(Var input) {
            Var x = input.requiresGrad() ? input : Var.param(input.value());
            Var psi = potential.forward(x).slice(1, 0, 1);
            Var g = Var.gradients(psi.sum(), List.of(x), true).get(0);
            // (d psi / d x_second, -d psi / d x_first)
            return Var.concat(1, g.slice(1, second, second + 1), g.slice(1, first, first + 1).neg());
        }

        @Override
        public List<Var> parameters() {
            return potential.parameters();
        }

        @Override
        public int outputSize(int inputSize) {
            return 2;
        }

        @Override
        public String toString() {
            return "Curl(" + potential + ")";
        }
    }

    /**
     * A divergence-free 2-D velocity field from a stream function network.
     *
     * @param streamFunction {@code [n, 2] -> [n, 1]} (only the first output is used)
     * @return {@code [n, 2] -> [n, 2]}, the velocity {@code (d psi/dy, -d psi/dx)}
     */
    public static Layer streamFunction(Layer streamFunction) {
        return new Curl(streamFunction, 0, 1);
    }

    /**
     * A density and flux on space-time {@code (x, t)} that satisfy {@code rho_t + J_x = 0}
     * exactly.
     *
     * @param potential {@code [n, 2] -> [n, 1]} over {@code (x, t)}
     * @return {@code [n, 2] -> [n, 2]}, the pair {@code (rho, J) = (d phi/dx, -d phi/dt)}
     */
    public static Layer conservationLaw(Layer potential) {
        // rho = d phi / dx, J = -d phi / dt  =>  rho_t + J_x = phi_xt - phi_tx = 0
        return new Curl(potential, 1, 0);
    }

    /**
     * The divergence {@code d f0/d x0 + d f1/d x1} of a 2-D field (for the conservation law
     * on {@code (x, t)}, ordered as {@code (rho, J)}, use {@link #continuityResidual}).
     *
     * @param field {@code [n, 2] -> [n, 2]}
     * @param points the points, a differentiable input ({@code Var.param})
     * @return {@code [n, 1]}
     */
    public static Var divergence(Layer field, Var points) {
        Var f = field.forward(points);
        Var d0 = Var.gradients(f.slice(1, 0, 1).sum(), List.of(points), true).get(0).slice(1, 0, 1);
        Var d1 = Var.gradients(f.slice(1, 1, 2).sum(), List.of(points), true).get(0).slice(1, 1, 2);
        return d0.add(d1);
    }

    /**
     * The continuity residual {@code rho_t + J_x} of a {@code (rho, J)} field over
     * {@code (x, t)}.
     *
     * @param field {@code [n, 2] -> [n, 2]}
     * @param points the points {@code (x, t)}, a differentiable input
     * @return {@code [n, 1]}
     */
    public static Var continuityResidual(Layer field, Var points) {
        Var f = field.forward(points);
        Var rhoT = Var.gradients(f.slice(1, 0, 1).sum(), List.of(points), true).get(0).slice(1, 1, 2);
        Var jX = Var.gradients(f.slice(1, 1, 2).sum(), List.of(points), true).get(0).slice(1, 0, 1);
        return rhoT.add(jX);
    }
}
