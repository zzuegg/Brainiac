package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A deep Gaussian process trained by doubly stochastic variational inference (Salimbeni and
 * Deisenroth 2017, <a href="https://arxiv.org/abs/1705.08933">arXiv:1705.08933</a>):
 * {@code f = f_L o ... o f_1}, each layer a sparse variational GP with its own inducing
 * inputs, the same model {@link SparseVariationalGp} fits for one layer.
 *
 * <p>Each layer uses the whitened parameterisation {@code u = L_zz v},
 * {@code q(v) = N(m, diag(s^2))}, so given its input {@code h} a layer's output is Gaussian
 * with mean {@code A m + h} (an identity mean function for hidden layers) and variance
 * {@code k(h, h) - |A|^2 + |A s|^2}, {@code A = K_hz L_zz^-T}. Training samples through the
 * layers with the reparameterisation trick (the first stochasticity), on minibatches (the
 * second); the ELBO is the expected Gaussian log likelihood at the last layer minus the
 * layers' KL terms {@code 1/2 sum(s^2 + m^2 - 1 - log s^2)}. Composing GPs gives
 * non-stationary, non-Gaussian predictive distributions (a step, input-dependent noise)
 * that a single stationary GP cannot express. Inducing inputs and RBF hyperparameters are fixed
 * per layer; the variational parameters and the noise are learned with Adam through autograd.
 */
public final class DeepGaussianProcess {

    /** One whitened SVGP layer. */
    private static final class GpLayer {
        final Tensor z;            // [M, din]
        final Tensor zSq;          // [1, M]
        final Tensor lInvT;        // [M, M] = L^-T
        final double lengthscale;
        final double variance;
        final boolean identityMean;
        final Var m;               // [M, dout]
        final Var logS;            // [M, dout]

        GpLayer(double[][] inducing, int dout, double lengthscale, double variance, boolean identityMean,
                SplittableRandom r) {
            int mm = inducing.length;
            int din = inducing[0].length;
            this.lengthscale = lengthscale;
            this.variance = variance;
            this.identityMean = identityMean;
            float[] zf = new float[mm * din];
            float[] zs = new float[mm];
            for (int i = 0; i < mm; i++) {
                for (int j = 0; j < din; j++) {
                    zf[i * din + j] = (float) inducing[i][j];
                    zs[i] += (float) (inducing[i][j] * inducing[i][j]);
                }
            }
            z = Tensor.of(Shape.of(mm, din), zf);
            zSq = Tensor.of(Shape.of(1, mm), zs);
            double[][] k = StationaryKernel.rbf(lengthscale, variance).gram(inducing);
            for (int i = 0; i < mm; i++) {
                k[i][i] += 1e-5 * variance;
            }
            double[][] l = MlMatrices.cholesky(k);
            double[][] inv = MlMatrices.triangularSolve(l, MlMatrices.identity(mm), false);   // L^-1
            float[] t = new float[mm * mm];
            for (int i = 0; i < mm; i++) {
                for (int j = 0; j < mm; j++) {
                    t[i * mm + j] = (float) inv[j][i];
                }
            }
            lInvT = Tensor.of(Shape.of(mm, mm), t);
            m = Var.param(Tensor.normal(Shape.of(mm, dout), 0, identityMean ? 0.01f : 0.1f, r));
            logS = Var.param(Tensor.full(Shape.of(mm, dout), identityMean ? -5f : 0f));
        }

        /** Returns {mean, variance} of the layer output for inputs h [B, din]. */
        Var[] marginals(Var h) {
            Var hSq = h.square().sum(1, true);   // [B, 1]
            Var cross = h.matmul(Var.of(z.transpose()));   // [B, M]
            Var sq = hSq.add(Var.of(zSq)).sub(cross.mul(2f)).maximum(0f);
            Var kxz = sq.mul((float) (-0.5 / (lengthscale * lengthscale))).exp().mul((float) variance);
            Var a = kxz.matmul(Var.of(lInvT));   // [B, M]
            Var mean = a.matmul(m);
            if (identityMean) {
                mean = mean.add(h);
            }
            Var a2 = a.square();
            Var var = a2.matmul(logS.mul(2f).exp()).sub(a2.sum(1, true)).add((float) variance).maximum(1e-6f);
            return new Var[] {mean, var};
        }

        Var kl() {
            Var s2 = logS.mul(2f).exp();
            return s2.add(m.square()).sub(1f).sub(logS.mul(2f)).sum().mul(0.5f);
        }
    }

    private final List<GpLayer> layers = new ArrayList<>();
    private final Var logNoise;
    private final double yMean;
    private final double yScale;

    private DeepGaussianProcess(double[][] x, int[] widths, double[] lengthscales, int inducing, double yMean,
            double yScale, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.yMean = yMean;
        this.yScale = yScale;
        int din = x[0].length;
        for (int l = 0; l < widths.length; l++) {
            double[][] z;
            if (l == 0) {
                z = KMeans.fit(x, Math.min(inducing, x.length), KMeans.Init.PLUS_PLUS, 1, 20, seed).centres();
            } else {
                z = new double[inducing][din];
                for (int i = 0; i < inducing; i++) {
                    for (int j = 0; j < din; j++) {
                        z[i][j] = -2.5 + 5.0 * i / (inducing - 1) + (j > 0 ? 0.1 * r.nextGaussian() : 0);
                    }
                }
            }
            boolean last = l == widths.length - 1;
            layers.add(new GpLayer(z, widths[l], lengthscales[l], 1, !last && din == widths[l], r));
            din = widths[l];
        }
        logNoise = Var.param(Tensor.full(Shape.of(1, 1), (float) Math.log(0.1)));
    }

    /**
     * Trains a deep GP on standardised targets.
     *
     * @param x inputs {@code [n][d]}, ideally standardised
     * @param y targets
     * @param widths the output width of each layer; the last is 1
     * @param lengthscales the RBF lengthscale of each layer
     * @param inducing inducing points per layer
     * @param iterations Adam steps
     * @param samples Monte Carlo samples per step
     * @param seed the seed
     * @return the trained model
     */
    public static DeepGaussianProcess fit(double[][] x, double[] y, int[] widths, double[] lengthscales, int inducing,
            int iterations, int samples, long seed) {
        int n = x.length;
        double mean = 0;
        for (double v : y) {
            mean += v / n;
        }
        double sd = 0;
        for (double v : y) {
            sd += (v - mean) * (v - mean) / n;
        }
        sd = Math.sqrt(sd) + 1e-12;
        DeepGaussianProcess g = new DeepGaussianProcess(x, widths, lengthscales, inducing, mean, sd, seed);
        Tensor xs = rows(x, samples);
        float[] yv = new float[n * samples];
        for (int s = 0; s < samples; s++) {
            for (int i = 0; i < n; i++) {
                yv[s * n + i] = (float) ((y[i] - mean) / sd);
            }
        }
        Tensor yt = Tensor.of(Shape.of(n * samples, 1), yv);
        List<Var> ps = g.parameters();
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int it = 0; it < iterations; it++) {
            Var[] last = g.propagate(Var.of(xs), r);
            Var noise = g.logNoise.exp();
            // E_q[log N(y; f, noise)] with f ~ N(mean, var): closed form at the last layer
            Var ll = last[0].sub(yt).square().add(last[1]).div(noise).mul(-0.5f).sub(g.logNoise.mul(0.5f))
                    .sub((float) (0.5 * Math.log(2 * Math.PI)));
            Var elbo = ll.sum().div((float) samples);
            for (GpLayer l : g.layers) {
                elbo = elbo.sub(l.kl());
            }
            elbo.neg().div((float) n).backward();
            opt.step(ps);
        }
        return g;
    }

    private static Tensor rows(double[][] x, int repeats) {
        int n = x.length;
        int d = x[0].length;
        float[] v = new float[n * repeats * d];
        for (int s = 0; s < repeats; s++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) {
                    v[(s * n + i) * d + j] = (float) x[i][j];
                }
            }
        }
        return Tensor.of(Shape.of(n * repeats, d), v);
    }

    /** Samples through every layer but the last; returns the last layer's {mean, variance}. */
    private Var[] propagate(Var h, SplittableRandom r) {
        for (int l = 0; l < layers.size(); l++) {
            Var[] mv = layers.get(l).marginals(h);
            if (l == layers.size() - 1) {
                return mv;
            }
            Tensor eps = Tensor.normal(mv[0].shape(), 0, 1, r);
            h = mv[0].add(mv[1].sqrt().mul(eps));
        }
        throw new IllegalStateException("no layers");
    }

    /**
     * The trainable parameters.
     *
     * @return the variational means, log standard deviations and the log noise
     */
    public List<Var> parameters() {
        List<Var> ps = new ArrayList<>();
        for (GpLayer l : layers) {
            ps.add(l.m);
            ps.add(l.logS);
        }
        ps.add(logNoise);
        return ps;
    }

    /**
     * Monte Carlo predictive samples: for each of {@code samples} propagations, the Gaussian
     * mean and variance (including the noise) of the target at every test input.
     *
     * @param xs test inputs {@code [m][d]}
     * @param samples propagations
     * @param seed the sampling seed
     * @return {@code {means [samples][m], variances [samples][m]}} in target units
     */
    public double[][][] predictive(double[][] xs, int samples, long seed) {
        return Var.noGrad(() -> {
            int m = xs.length;
            Var[] last = propagate(Var.of(rows(xs, samples)), new SplittableRandom(seed));
            double[] mu = last[0].value().toDoubleArray();
            double[] var = last[1].value().toDoubleArray();
            double noise = Math.exp(logNoise.value().item());
            double[][][] out = new double[2][samples][m];
            for (int s = 0; s < samples; s++) {
                for (int i = 0; i < m; i++) {
                    out[0][s][i] = yMean + yScale * mu[s * m + i];
                    out[1][s][i] = yScale * yScale * (var[s * m + i] + noise);
                }
            }
            return out;
        });
    }

    /**
     * The mean negative log predictive density of test targets under the Gaussian-mixture
     * predictive.
     *
     * @param xs test inputs
     * @param ys test targets
     * @param samples propagations
     * @param seed the sampling seed
     * @return the mean NLL
     */
    public double nll(double[][] xs, double[] ys, int samples, long seed) {
        double[][][] p = predictive(xs, samples, seed);
        double total = 0;
        for (int i = 0; i < ys.length; i++) {
            double[] lp = new double[samples];
            for (int s = 0; s < samples; s++) {
                double v = p[1][s][i];
                double d = ys[i] - p[0][s][i];
                lp[s] = -0.5 * d * d / v - 0.5 * Math.log(2 * Math.PI * v);
            }
            total -= McmcDiagnostics.logMeanExp(lp);
        }
        return total / ys.length;
    }

    /**
     * The mean CRPS of test targets, estimated from predictive samples.
     *
     * @param xs test inputs
     * @param ys test targets
     * @param samples propagations (one draw per propagation)
     * @param seed the sampling seed
     * @return the mean CRPS
     */
    public double crps(double[][] xs, double[] ys, int samples, long seed) {
        double[][][] p = predictive(xs, samples, seed);
        SplittableRandom r = new SplittableRandom(seed + 7);
        double total = 0;
        for (int i = 0; i < ys.length; i++) {
            double[] d = new double[samples];
            for (int s = 0; s < samples; s++) {
                d[s] = p[0][s][i] + Math.sqrt(p[1][s][i]) * r.nextGaussian();
            }
            total += sampleCrps(d, ys[i]);
        }
        return total / ys.length;
    }

    /**
     * CRPS from samples: {@code E|X - y| - 1/2 E|X - X'|}.
     *
     * @param draws predictive draws
     * @param y the target
     * @return the CRPS estimate
     */
    public static double sampleCrps(double[] draws, double y) {
        double a = 0;
        double b = 0;
        for (double v : draws) {
            a += Math.abs(v - y);
        }
        for (double v : draws) {
            for (double w : draws) {
                b += Math.abs(v - w);
            }
        }
        int n = draws.length;
        return a / n - 0.5 * b / ((double) n * n);
    }
}
