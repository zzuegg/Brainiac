package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ContinuousFlow;
import io.github.zzuegg.brainiac.experimental.nn.DensityFlow;
import io.github.zzuegg.brainiac.experimental.nn.OdeSolver;
import io.github.zzuegg.brainiac.experimental.nn.RealNvpFlow;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.FlowMatching;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Density estimation on 2-D {@link PointClouds} (two moons, two rings) and on noise-augmented
 * {@link ToyImages}: every model is trained by maximum likelihood for the same number of
 * steps and scored by held-out negative log-likelihood (nats per point, or bits per
 * dimension for images in 8-bit pixel units, as in {@link TarFlowExperiment}). Models:
 * {@code GlowFlow}, {@code SplineFlow} and {@code MaskedAutoregressiveFlow} (wave 6, item
 * 4.4), FFJORD ({@link ContinuousFlow}, 4.5), the exact likelihood of a flow-matching
 * velocity field computed by the same ODE (4.5's comparison), and the baselines
 * {@link RealNvpFlow} and a full-covariance Gaussian mixture fitted by EM.
 */
@SuppressWarnings("exports")
public final class DensityBenchmark {
    private static final int BATCH = 128;
    private static final int HELD_OUT = 1024;

    /** The 2-D targets. */
    public enum Target {
        /** Two interleaved moons. */
        MOONS,
        /** Two concentric rings. */
        RINGS
    }

    private DensityBenchmark() {
    }

    /**
     * Draws points of a target.
     *
     * @param target the target
     * @param n how many
     * @param random the random source
     * @return {@code [n][2]}
     */
    public static float[][] data(Target target, int n, SplittableRandom random) {
        return target == Target.MOONS ? PointClouds.moons(n, 0.05, random) : PointClouds.rings(n, 0.05, random);
    }

    private static Tensor heldOut(Target target, long seed) {
        return Nets.matrix(data(target, HELD_OUT, new SplittableRandom(seed + 1000)));
    }

    /**
     * Trains a flow by maximum likelihood and returns its held-out negative log-likelihood.
     *
     * @param flow the flow
     * @param target the data
     * @param steps optimizer steps of 128 points
     * @param learningRate the Adam learning rate (dropped 3x for the last quarter)
     * @param seed the data seed
     * @return nats per point
     */
    public static double train(DensityFlow flow, Target target, int steps, float learningRate, long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < steps; i++) {
            if (i == steps * 3 / 4) {
                opt.learningRate(learningRate / 3);
            }
            Var loss = flow.logProb(Var.of(Nets.matrix(data(target, BATCH, r)))).mean().neg();
            loss.backward();
            Nets.clip(flow.parameters(), 5f);
            opt.step(flow.parameters());
        }
        return -flow.logProb(Var.of(heldOut(target, seed))).mean().value().item();
    }

    /**
     * The RealNVP baseline.
     *
     * @param target the data
     * @param steps optimizer steps
     * @param seed the seed
     * @return nats per point
     */
    public static double realNvp(Target target, int steps, long seed) {
        RealNvpFlow nvp = new RealNvpFlow(2, 6, 64, new SplittableRandom(seed));
        return train(new DensityFlow() {
            @Override
            public io.github.zzuegg.brainiac.nn.Flows.Result forward(Var x) {
                Var[] zl = nvp.forward(x);
                return new io.github.zzuegg.brainiac.nn.Flows.Result(zl[0], zl[1]);
            }

            @Override
            public Tensor inverse(Tensor z) {
                return nvp.inverse(z);
            }

            @Override
            public int dimension() {
                return 2;
            }

            @Override
            public List<Var> parameters() {
                return nvp.parameters();
            }
        }, target, steps, 2e-3f, seed);
    }

    /**
     * A Gaussian mixture with full covariances fitted by expectation-maximisation on 4096 points.
     *
     * @param target the data
     * @param components the number of components
     * @param seed the seed
     * @return held-out nats per point
     */
    public static double gaussianMixture(Target target, int components, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = data(target, 4096, r);
        int n = x.length;
        double[] w = new double[components];
        double[][] mu = new double[components][2];
        double[][] cov = new double[components][];
        for (int k = 0; k < components; k++) {
            w[k] = 1.0 / components;
            float[] p = x[r.nextInt(n)];
            mu[k][0] = p[0];
            mu[k][1] = p[1];
            cov[k] = new double[] {0.1, 0, 0.1};
        }
        double[][] resp = new double[n][components];
        for (int it = 0; it < 100; it++) {
            for (int i = 0; i < n; i++) {
                double[] lp = new double[components];
                for (int k = 0; k < components; k++) {
                    lp[k] = Math.log(w[k]) + logGauss(x[i], mu[k], cov[k]);
                }
                double lse = logSumExp(lp);
                for (int k = 0; k < components; k++) {
                    resp[i][k] = Math.exp(lp[k] - lse);
                }
            }
            for (int k = 0; k < components; k++) {
                double nk = 1e-9;
                double mx = 0;
                double my = 0;
                for (int i = 0; i < n; i++) {
                    nk += resp[i][k];
                    mx += resp[i][k] * x[i][0];
                    my += resp[i][k] * x[i][1];
                }
                mx /= nk;
                my /= nk;
                double sxx = 0;
                double sxy = 0;
                double syy = 0;
                for (int i = 0; i < n; i++) {
                    double dx = x[i][0] - mx;
                    double dy = x[i][1] - my;
                    sxx += resp[i][k] * dx * dx;
                    sxy += resp[i][k] * dx * dy;
                    syy += resp[i][k] * dy * dy;
                }
                w[k] = nk / n;
                mu[k][0] = mx;
                mu[k][1] = my;
                cov[k] = new double[] {sxx / nk + 1e-4, sxy / nk, syy / nk + 1e-4};
            }
        }
        float[][] test = data(target, HELD_OUT, new SplittableRandom(seed + 1000));
        double nll = 0;
        for (float[] p : test) {
            double[] lp = new double[components];
            for (int k = 0; k < components; k++) {
                lp[k] = Math.log(w[k]) + logGauss(p, mu[k], cov[k]);
            }
            nll -= logSumExp(lp);
        }
        return nll / test.length;
    }

    private static double logGauss(float[] p, double[] mu, double[] c) {
        double det = c[0] * c[2] - c[1] * c[1];
        double dx = p[0] - mu[0];
        double dy = p[1] - mu[1];
        double q = (c[2] * dx * dx - 2 * c[1] * dx * dy + c[0] * dy * dy) / det;
        return -0.5 * q - Math.log(2 * Math.PI) - 0.5 * Math.log(det);
    }

    private static double logSumExp(double[] v) {
        double m = Arrays.stream(v).max().orElseThrow();
        double s = 0;
        for (double x : v) {
            s += Math.exp(x - m);
        }
        return m + Math.log(s);
    }

    /**
     * FFJORD trained by maximum likelihood with the Hutchinson trace, RK4 and a small
     * kinetic-energy penalty, evaluated with the exact trace and an adaptive solver.
     *
     * @param target the data
     * @param steps optimizer steps
     * @param seed the seed
     * @return {@code {held-out nats per point, field evaluations per density evaluation}}
     */
    public static double[] ffjord(Target target, int steps, long seed) {
        MlpDenoiser field = MlpDenoiser.of(2, 0, 64, 2, seed);
        int[] none = new int[BATCH];
        Arrays.fill(none, -1);
        ContinuousFlow train = ContinuousFlow.of(field.bind(none), field.parameters(), 2, OdeSolver.rk4(3), seed)
                .trace(ContinuousFlow.Trace.HUTCHINSON).training(true);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < steps; i++) {
            if (i == steps * 3 / 4) {
                opt.learningRate(1e-3f);
            }
            Var loss = train.logProb(Var.of(Nets.matrix(data(target, BATCH, r)))).mean().neg()
                    .add(train.kineticEnergy().mean().mul(0.05f));
            loss.backward();
            Nets.clip(field.parameters(), 5f);
            opt.step(field.parameters());
        }
        return evaluateField(field, target, seed);
    }

    /**
     * Core flow matching trained for the same steps, then scored by the exact likelihood of
     * its velocity field through the same ODE as FFJORD.
     *
     * @param target the data
     * @param steps optimizer steps
     * @param seed the seed
     * @return {@code {held-out nats per point, field evaluations per density evaluation}}
     */
    public static double[] flowMatching(Target target, int steps, long seed) {
        MlpDenoiser field = MlpDenoiser.of(2, 0, 64, 2, seed);
        int[] none = new int[BATCH];
        Arrays.fill(none, -1);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < steps; i++) {
            if (i == steps * 3 / 4) {
                opt.learningRate(1e-3f);
            }
            Var loss = FlowMatching.loss(field.bind(none), Nets.matrix(data(target, BATCH, r)), r);
            loss.backward();
            Nets.clip(field.parameters(), 5f);
            opt.step(field.parameters());
        }
        return evaluateField(field, target, seed);
    }

    private static double[] evaluateField(MlpDenoiser field, Target target, long seed) {
        int[] none = new int[HELD_OUT];
        Arrays.fill(none, -1);
        ContinuousFlow eval = ContinuousFlow.of(field.bind(none), field.parameters(), 2,
                OdeSolver.dopri5(1e-4f, 1e-4f), seed);
        double nll = -eval.logProb(Var.of(heldOut(target, seed))).mean().value().item();
        return new double[] {nll, eval.evaluations()};
    }
}
