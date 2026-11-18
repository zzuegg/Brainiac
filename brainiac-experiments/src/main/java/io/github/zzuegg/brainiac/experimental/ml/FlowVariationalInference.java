package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.MaskedAutoregressiveFlow;
import io.github.zzuegg.brainiac.nn.Flows;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Variational inference with reparameterised gradients: fit {@code q(z)} to an unnormalised
 * posterior {@link LogDensity} by maximising the ELBO {@code E_q[log p(z) - log q(z)]}, with
 * the gradient flowing through {@code z = T(eps)}, {@code eps ~ N(0, I)}, by autograd.
 *
 * <ul>
 *   <li>{@link Family#MEAN_FIELD} and {@link Family#FULL_RANK}: ADVI (Kucukelbir et al. 2017,
 *       <a href="https://arxiv.org/abs/1603.00788">arXiv:1603.00788</a>), a diagonal or
 *       Cholesky-factored Gaussian in unconstrained space;</li>
 *   <li>{@link Family#PLANAR} and {@link Family#RADIAL}: normalising flows (Rezende and
 *       Mohamed 2015, <a href="https://arxiv.org/abs/1505.05770">arXiv:1505.05770</a>) on top of
 *       the mean-field Gaussian, {@code z + u h(w.z + b)} and
 *       {@code z + beta (z - z0) / (alpha + |z - z0|)}, whose log-determinants are closed
 *       form, so {@code q} can bend and skew;</li>
 *   <li>{@link Family#IAF}: an inverse autoregressive flow (Kingma et al. 2016,
 *       <a href="https://arxiv.org/abs/1606.04934">arXiv:1606.04934</a>). It reuses
 *       {@link MaskedAutoregressiveFlow}: the MAF's density direction, run on the noise, is
 *       exactly an IAF sampling pass, with the log-determinant of the same single pass.</li>
 * </ul>
 * The target enters the graph as a custom operation whose gradient is the density's own, so
 * any {@link LogDensity} (a probabilistic program, a closed-form test target) can be fitted.
 * When the target is normalised, {@code -ELBO} is exactly {@code KL(q || p)}.
 */
public final class FlowVariationalInference {

    /** The variational family. */
    public enum Family {
        /** Diagonal Gaussian (mean-field ADVI). */
        MEAN_FIELD,
        /** Gaussian with a full Cholesky factor (full-rank ADVI). */
        FULL_RANK,
        /** Mean-field Gaussian followed by planar flows. */
        PLANAR,
        /** Mean-field Gaussian followed by radial flows. */
        RADIAL,
        /** Mean-field Gaussian followed by an inverse autoregressive flow. */
        IAF
    }

    private final Family family;
    private final int dims;
    private final Var mu;
    private final Var logSigma;
    private final Var lower;
    private final Tensor lowerMask;
    private final List<Var[]> flows = new ArrayList<>();
    private final MaskedAutoregressiveFlow iaf;
    private final List<Var> parameters = new ArrayList<>();

    private FlowVariationalInference(Family family, int dims, int flowCount, long seed) {
        this.family = family;
        this.dims = dims;
        SplittableRandom r = new SplittableRandom(seed);
        mu = Var.param(Tensor.zeros(Shape.of(1, dims)));
        logSigma = Var.param(Tensor.zeros(Shape.of(1, dims)));
        parameters.add(mu);
        parameters.add(logSigma);
        if (family == Family.FULL_RANK) {
            lower = Var.param(Tensor.zeros(Shape.of(dims, dims)));
            float[] m = new float[dims * dims];
            for (int i = 0; i < dims; i++) {
                for (int j = 0; j < i; j++) {
                    m[i * dims + j] = 1;
                }
            }
            lowerMask = Tensor.of(Shape.of(dims, dims), m);
            parameters.add(lower);
        } else {
            lower = null;
            lowerMask = null;
        }
        if (family == Family.PLANAR || family == Family.RADIAL) {
            for (int k = 0; k < flowCount; k++) {
                Var a = Var.param(Tensor.normal(Shape.of(dims, 1), 0, 0.1f, r));
                Var b = Var.param(Tensor.normal(Shape.of(1, dims), 0, 0.1f, r));
                Var c = Var.param(Tensor.zeros(Shape.of(1, 1)));
                flows.add(new Var[] {a, b, c});
                parameters.add(a);
                parameters.add(b);
                parameters.add(c);
            }
        }
        if (family == Family.IAF) {
            iaf = MaskedAutoregressiveFlow.of(dims, Math.max(1, flowCount), Math.max(16, dims), seed);
            parameters.addAll(iaf.parameters());
        } else {
            iaf = null;
        }
    }

    /**
     * Creates a variational family, initialised at the standard normal.
     *
     * @param family the family
     * @param dims the dimension of {@code z}
     * @param flows the number of flow layers (planar, radial, IAF; ignored otherwise)
     * @param seed the initialisation seed
     * @return the family
     */
    public static FlowVariationalInference create(Family family, int dims, int flows, long seed) {
        return new FlowVariationalInference(family, dims, flows, seed);
    }

    /**
     * The trainable parameters.
     *
     * @return the parameters
     */
    public List<Var> parameters() {
        return List.copyOf(parameters);
    }

    /**
     * Draws reparameterised samples with their log density under {@code q}.
     *
     * @param eps standard normal noise {@code [n][dims]}
     * @return {@code {z [n, dims], log q(z) [n]}}
     */
    public Var[] sample(Tensor eps) {
        Var e = Var.of(eps);
        Var logBase = e.square().sum(1, false).mul(-0.5f).sub((float) (0.5 * dims * Math.log(2 * Math.PI)));
        Var z;
        Var logDet;
        if (family == Family.FULL_RANK) {
            Var l = lower.mul(lowerMask);
            z = e.mul(logSigma.exp()).add(e.matmul(l.transpose())).add(mu);
            // the factor is unit-free below the diagonal: L = diag(sigma) + strictly lower
            logDet = logSigma.sum();
        } else {
            z = e.mul(logSigma.exp()).add(mu);
            logDet = logSigma.sum();
        }
        Var logQ = logBase.sub(logDet);
        for (Var[] f : flows) {
            Var[] out = family == Family.PLANAR ? planar(z, f) : radial(z, f);
            z = out[0];
            logQ = logQ.sub(out[1]);
        }
        if (iaf != null) {
            Flows.Result res = iaf.forward(z);
            z = res.output();
            logQ = logQ.sub(res.logDet());
        }
        return new Var[] {z, logQ};
    }

    /** Planar flow with the invertibility reparameterisation of u (Rezende and Mohamed, appendix A.1). */
    private static Var[] planar(Var z, Var[] p) {
        Var w = p[0];   // [d, 1]
        Var u = p[1];   // [1, d]
        Var b = p[2];   // [1, 1]
        Var wu = u.matmul(w);   // [1, 1]
        Var ww = w.square().sum().add(1e-6f);
        Var m = wu.softplus().sub(1f);
        Var uHat = u.add(m.sub(wu).div(ww).mul(w.transpose()));
        Var a = z.matmul(w).add(b);   // [n, 1]
        Var h = a.tanh();
        Var out = z.add(h.matmul(uHat));
        Var dh = h.square().neg().add(1f);   // [n, 1]
        Var psiU = dh.mul(uHat.matmul(w));   // [n, 1]
        Var logDet = psiU.add(1f).abs().add(1e-8f).log().sum(1, false);
        return new Var[] {out, logDet};
    }

    /** Radial flow {@code z + beta h (z - z0)}, {@code h = 1 / (alpha + r)}. */
    private Var[] radial(Var z, Var[] p) {
        Var z0 = p[1];   // [1, d]
        Var alpha = p[2].softplus().add(1e-3f);   // [1, 1]
        Var betaRaw = p[0].sum().reshape(1, 1);
        Var beta = alpha.neg().add(betaRaw.softplus());
        Var diff = z.sub(z0);
        Var r = diff.square().sum(1, true).add(1e-12f).sqrt();   // [n, 1]
        Var h = alpha.add(r).pow(-1f);
        Var bh = beta.mul(h);
        Var out = z.add(diff.mul(bh));
        Var hPrime = h.square().neg();
        Var logDet = bh.add(1f).log().mul((float) (dims - 1)).add(bh.add(1f).add(beta.mul(hPrime).mul(r)).log());
        return new Var[] {out, logDet.sum(1, false)};
    }

    /**
     * The target density as a graph operation over a batch of rows.
     *
     * @param target the density
     * @param z the points {@code [n, d]}
     * @return {@code log p(z)} as {@code [n]}
     */
    public static Var logDensity(LogDensity target, Var z) {
        int n = z.shape().dim(0);
        int d = z.shape().dim(1);
        double[][] grads = new double[n][d];
        return Var.custom(in -> {
            double[] v = in[0].toDoubleArray();
            float[] out = new float[n];
            for (int i = 0; i < n; i++) {
                double[] x = java.util.Arrays.copyOfRange(v, i * d, (i + 1) * d);
                double lp = target.evaluate(x, grads[i]);
                out[i] = (float) Math.max(lp, -1e30);
            }
            return Tensor.of(Shape.of(n), out);
        }, (in, out, g) -> {
            double[] gv = g.toDoubleArray();
            float[] r = new float[n * d];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) {
                    double v = gv[i] * grads[i][j];
                    r[i * d + j] = (float) (Double.isFinite(v) ? v : 0);
                }
            }
            return new Tensor[] {Tensor.of(Shape.of(n, d), r)};
        }, z);
    }

    /**
     * Maximises the ELBO with Adam.
     *
     * @param target the unnormalised posterior
     * @param iterations the optimisation steps
     * @param batch Monte Carlo samples per step
     * @param learningRate the Adam learning rate, for example 0.01
     * @param seed the noise seed
     * @return the ELBO estimate of each step
     */
    public double[] fit(LogDensity target, int iterations, int batch, float learningRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(learningRate);
        double[] history = new double[iterations];
        for (int it = 0; it < iterations; it++) {
            Var[] s = sample(Tensor.normal(Shape.of(batch, dims), 0, 1, r));
            Var elbo = logDensity(target, s[0]).sub(s[1]).mean();
            history[it] = elbo.value().item();
            elbo.neg().backward();
            opt.step(parameters);
        }
        return history;
    }

    /**
     * A Monte Carlo ELBO estimate without gradients.
     *
     * @param target the unnormalised posterior
     * @param n the number of samples
     * @param seed the noise seed
     * @return {@code E_q[log p - log q]}
     */
    public double elbo(LogDensity target, int n, long seed) {
        return Var.noGrad(() -> {
            Var[] s = sample(Tensor.normal(Shape.of(n, dims), 0, 1, new SplittableRandom(seed)));
            double[] z = s[0].value().toDoubleArray();
            double[] lq = s[1].value().toDoubleArray();
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += target.value(java.util.Arrays.copyOfRange(z, i * dims, (i + 1) * dims)) - lq[i];
            }
            return sum / n;
        });
    }

    /**
     * Draws samples from {@code q}.
     *
     * @param n the number of samples
     * @param seed the noise seed
     * @return {@code [n][dims]}
     */
    public double[][] draw(int n, long seed) {
        return Var.noGrad(() -> {
            double[] z = sample(Tensor.normal(Shape.of(n, dims), 0, 1, new SplittableRandom(seed)))[0].value()
                    .toDoubleArray();
            double[][] out = new double[n][dims];
            for (int i = 0; i < n; i++) {
                System.arraycopy(z, i * dims, out[i], 0, dims);
            }
            return out;
        });
    }
}
