package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * A Lagrangian neural network (Cranmer, Greydanus, Hoyer, Battaglia, Spergel and Ho, 2020):
 * a network learns a scalar Lagrangian {@code L_theta(q, q')} and accelerations follow from
 * the Euler-Lagrange equations,
 *
 * <pre>
 * q'' = (d2L/dq'dq')^-1 (dL/dq - (d2L/dq'dq) q')
 * </pre>
 *
 * <p>so the model works in arbitrary coordinates (angles, positions) without canonical
 * momenta, and conserves the learned energy. The network output is added to a unit kinetic
 * prior {@code |q'|^2 / 2}, so the mass matrix starts at the identity instead of near zero
 * (an untrained or poorly extrapolating {@code d2L/dq'^2} otherwise makes the solve explode).
 * The Hessian blocks come from second-order autograd; the solve is closed-form for
 * {@code d <= 2} degrees of freedom.
 *
 * <p>As a {@link Layer}, {@code forward} maps states {@code [b, 2d]} (positions then
 * velocities) to their time derivatives {@code (q', q'')}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2003.04630">arXiv:2003.04630</a>.
 */
public final class LagrangianNetwork implements Layer {

    private final int dimension;
    private final SimpleMlp lagrangian;

    private LagrangianNetwork(int dimension, int width, int depth, long seed) {
        if (dimension < 1 || dimension > 2) {
            throw new IllegalArgumentException("LagrangianNetwork supports 1 or 2 degrees of freedom");
        }
        this.dimension = dimension;
        int[] sizes = new int[depth + 2];
        sizes[0] = 2 * dimension;
        for (int i = 1; i <= depth; i++) {
            sizes[i] = width;
        }
        sizes[depth + 1] = 1;
        this.lagrangian = SimpleMlp.tanh(seed, sizes);
    }

    /**
     * Creates the network.
     *
     * @param dimension degrees of freedom, 1 or 2
     * @param width hidden width
     * @param depth hidden layers
     * @param seed the seed
     * @return the network
     */
    public static LagrangianNetwork of(int dimension, int width, int depth, long seed) {
        return new LagrangianNetwork(dimension, width, depth, seed);
    }

    @Override
    public Var forward(Var input) {
        Var x = input.requiresGrad() ? input : Var.param(input.value());
        int d = dimension;
        Var kinetic = x.slice(1, d, 2 * d).square().sum(1, true).mul(0.5f);
        Var g = Var.gradients(lagrangian.forward(x).add(kinetic).sum(), List.of(x), true).get(0);   // [b, 2d]
        // Hessian rows for the velocity components: d/dx (dL/dq'_i)
        List<Var> rows = new ArrayList<>();
        for (int i = 0; i < d; i++) {
            rows.add(Var.gradients(g.slice(1, d + i, d + i + 1).sum(), List.of(x), true).get(0));
        }
        Var qdot = x.slice(1, d, 2 * d);
        // rhs_i = dL/dq_i - sum_j d2L/(dq'_i dq_j) q'_j
        Var[] rhs = new Var[d];
        Var[][] a = new Var[d][d];
        for (int i = 0; i < d; i++) {
            Var hqd = rows.get(i).slice(1, 0, d);
            rhs[i] = g.slice(1, i, i + 1).sub(hqd.mul(qdot).sum(1, true));
            for (int j = 0; j < d; j++) {
                a[i][j] = rows.get(i).slice(1, d + j, d + j + 1);
            }
        }
        Var acc;
        if (d == 1) {
            acc = rhs[0].div(a[0][0].add(1e-4f));
        } else {
            Var a00 = a[0][0].add(1e-4f);
            Var a11 = a[1][1].add(1e-4f);
            Var det = a00.mul(a11).sub(a[0][1].mul(a[1][0]));
            Var x0 = a11.mul(rhs[0]).sub(a[0][1].mul(rhs[1])).div(det);
            Var x1 = a00.mul(rhs[1]).sub(a[1][0].mul(rhs[0])).div(det);
            acc = Var.concat(1, x0, x1);
        }
        return Var.concat(1, qdot, acc);
    }

    /**
     * The mean squared error of predicted accelerations.
     *
     * @param states {@code [b, 2d]}
     * @param accelerations observed {@code q''} {@code [b, d]}
     * @return the loss
     */
    public Var loss(Tensor states, Tensor accelerations) {
        return forward(Var.param(states)).slice(1, dimension, 2 * dimension).sub(accelerations).square().mean();
    }

    /**
     * A frozen vector field {@code (q, q') -> (q', q'')} in double precision for rollouts: the
     * gradient is exact and the Hessian is taken by central differences of it.
     *
     * @return the field
     */
    public UnaryOperator<double[]> vectorField() {
        MlpSnapshot s = new MlpSnapshot(lagrangian);
        int d = dimension;
        UnaryOperator<double[]> grad = x -> {
            double[] g = s.inputGradient(x, 0);
            for (int i = 0; i < d; i++) {
                g[d + i] += x[d + i];   // kinetic prior
            }
            return g;
        };
        return x -> {
            double[] g = grad.apply(x);
            double h = 1e-4;
            double[][] hess = new double[d][2 * d];
            for (int i = 0; i < 2 * d; i++) {
                double[] xp = x.clone();
                double[] xm = x.clone();
                xp[i] += h;
                xm[i] -= h;
                double[] gp = grad.apply(xp);
                double[] gm = grad.apply(xm);
                for (int r = 0; r < d; r++) {
                    hess[r][i] = (gp[d + r] - gm[d + r]) / (2 * h);
                }
            }
            double[] rhs = new double[d];
            for (int i = 0; i < d; i++) {
                rhs[i] = g[i];
                for (int j = 0; j < d; j++) {
                    rhs[i] -= hess[i][j] * x[d + j];
                }
            }
            double[] out = new double[2 * d];
            System.arraycopy(x, d, out, 0, d);
            if (d == 1) {
                out[1] = rhs[0] / (hess[0][1] + 1e-4);
            } else {
                double a00 = hess[0][2] + 1e-4;
                double a01 = hess[0][3];
                double a10 = hess[1][2];
                double a11 = hess[1][3] + 1e-4;
                double det = a00 * a11 - a01 * a10;
                out[2] = (a11 * rhs[0] - a01 * rhs[1]) / det;
                out[3] = (a00 * rhs[1] - a10 * rhs[0]) / det;
            }
            return out;
        };
    }

    @Override
    public List<Var> parameters() {
        return lagrangian.parameters();
    }

    @Override
    public int outputSize(int inputSize) {
        return 2 * dimension;
    }

    @Override
    public String toString() {
        return "LagrangianNetwork(d=" + dimension + ")";
    }
}
