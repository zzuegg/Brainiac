package io.github.zzuegg.brainiac.experimental.sbi;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Prior;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Simulator;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.FlowMatching;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Continuous-time posterior estimators.
 * <ul>
 *   <li>{@link #fmpe}: flow-matching posterior estimation (Dax et al., NeurIPS 2023,
 *       <a href="https://arxiv.org/abs/2305.17161">arXiv:2305.17161</a>): a conditional
 *       velocity field {@code v(theta_t, t, x)} trained with core {@link FlowMatching#loss} on
 *       straight paths between standardised parameters and noise, sampled by integrating the
 *       ODE with Heun steps; no architectural constraint, so multimodal posteriors are
 *       easy;</li>
 *   <li>{@link #npse}: neural posterior score estimation (Sharrock et al., ICML 2024; Geffner,
 *       Papamakarios and Mnih, ICML 2023): a variance-preserving diffusion whose network
 *       predicts the noise given {@code (theta_t, t, x)}; samples come from the reverse SDE.
 *       {@link #sampleComposed} combines the scores of {@code n} i.i.d. observations as
 *       {@code sum_i s(theta_t | x_i) - (n - 1) s_prior(theta_t)}, where the diffused prior
 *       score of the standardised prior is approximated by {@code -theta_t} (exact for a Gaussian
 *       prior, since a VP diffusion keeps unit variance).</li>
 * </ul>
 */
public final class FlowMatchingPosterior {

    private final SimpleMlp net;
    private final double[] tMean;
    private final double[] tSd;
    private final double[] xMean;
    private final double[] xSd;
    private final boolean score;
    private final long simulations;

    private FlowMatchingPosterior(SimpleMlp net, double[] tMean, double[] tSd, double[] xMean, double[] xSd,
            boolean score, long simulations) {
        this.net = net;
        this.tMean = tMean;
        this.tSd = tSd;
        this.xMean = xMean;
        this.xSd = xSd;
        this.score = score;
        this.simulations = simulations;
    }

    private static double[][][] simulate(Prior prior, Simulator sim, int n, SplittableRandom r) {
        double[][] t = new double[n][];
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) {
            t[i] = prior.sample(r);
            x[i] = sim.simulate(t[i], r);
        }
        return new double[][][] {t, x};
    }

    private Denoiser.Conditional model() {
        return (noisy, time, condition) -> {
            int b = noisy.shape().dim(0);
            Tensor tt = time.reshape(b, 1);
            Var in = Var.concat(1, noisy, Var.of(tt), Var.of(tt.mul((float) Math.PI).sin()),
                    Var.of(tt.add(1e-3f).log().mul(0.25f)), Var.of(condition));
            return net.forward(in);
        };
    }

    /**
     * Trains a flow-matching posterior estimator on prior simulations.
     *
     * @param prior the prior
     * @param sim the simulator
     * @param simulations the simulation budget
     * @param hidden hidden width
     * @param iterations Adam steps on minibatches of 128
     * @param seed the seed
     * @return the estimator
     */
    public static FlowMatchingPosterior fmpe(Prior prior, Simulator sim, int simulations, int hidden, int iterations,
            long seed) {
        return fit(prior, sim, simulations, hidden, iterations, seed, false);
    }

    /**
     * Trains a score-based (VP diffusion) posterior estimator on prior simulations.
     *
     * @param prior the prior
     * @param sim the simulator
     * @param simulations the simulation budget
     * @param hidden hidden width
     * @param iterations Adam steps on minibatches of 128
     * @param seed the seed
     * @return the estimator
     */
    public static FlowMatchingPosterior npse(Prior prior, Simulator sim, int simulations, int hidden, int iterations,
            long seed) {
        return fit(prior, sim, simulations, hidden, iterations, seed, true);
    }

    private static FlowMatchingPosterior fit(Prior prior, Simulator sim, int simulations, int hidden, int iterations,
            long seed, boolean score) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][][] d = simulate(prior, sim, simulations, r);
        double[][] t = d[0];
        double[][] x = d[1];
        double[] tm = SbiArrays.mean(t);
        double[] ts = SbiArrays.sd(t, tm);
        double[] xm = SbiArrays.mean(x);
        double[] xs = SbiArrays.sd(x, xm);
        int dt = tm.length;
        SimpleMlp net = SimpleMlp.of(seed, Var::silu, dt + 3 + xm.length, hidden, hidden, hidden, dt);
        FlowMatchingPosterior q = new FlowMatchingPosterior(net, tm, ts, xm, xs, score, simulations);
        Denoiser.Conditional model = q.model();
        Optimizer opt = Optimizer.adam(2e-3f);
        int b = Math.min(256, simulations);
        for (int it = 0; it < iterations; it++) {
            int[] idx = SbiArrays.batch(simulations, b, r);
            Tensor tb = SbiArrays.standardisedRows(t, idx, tm, ts);
            Tensor xb = SbiArrays.standardisedRows(x, idx, xm, xs);
            Var loss = score ? q.scoreLoss(model, tb, xb, r) : FlowMatching.loss(model.given(xb), tb, r);
            loss.backward();
            opt.step(net.parameters());
        }
        return q;
    }

    private static final double BETA_MIN = 0.1;
    private static final double BETA_MAX = 12;
    private static final double T_MIN = 1e-3;

    private static double beta(double t) {
        return BETA_MIN + t * (BETA_MAX - BETA_MIN);
    }

    private static double alpha(double t) {
        return Math.exp(-0.5 * (BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t * t));
    }

    private Var scoreLoss(Denoiser.Conditional model, Tensor theta0, Tensor x, SplittableRandom r) {
        int b = theta0.shape().dim(0);
        int dt = tMean.length;
        float[] time = new float[b];
        float[] a = new float[b];
        float[] s = new float[b];
        for (int i = 0; i < b; i++) {
            double t = T_MIN + (1 - T_MIN) * r.nextDouble();
            time[i] = (float) t;
            double al = alpha(t);
            a[i] = (float) al;
            s[i] = (float) Math.sqrt(1 - al * al);
        }
        Tensor eps = Tensor.normal(Shape.of(b, dt), 0, 1, r);
        Tensor xt = theta0.mul(Tensor.of(Shape.of(b, 1), a)).add(eps.mul(Tensor.of(Shape.of(b, 1), s)));
        Var pred = model.apply(Var.of(xt), Tensor.of(Shape.of(b), time), x);
        return pred.sub(Var.of(eps)).square().mean();
    }

    /**
     * Draws posterior samples for one observation.
     *
     * @param x the observation
     * @param n the number of samples
     * @param steps integration steps
     * @param random the random source
     * @return {@code [n][d]}
     */
    public double[][] sample(double[] x, int n, int steps, SplittableRandom random) {
        return sampleComposed(new double[][] {x}, n, steps, random);
    }

    /**
     * Draws posterior samples given several i.i.d. observations. For a flow-matching model only
     * one observation is supported; a score model composes the per-observation scores.
     *
     * @param xs the observations {@code [k][dx]}
     * @param n the number of samples
     * @param steps integration steps
     * @param random the random source
     * @return {@code [n][d]}
     */
    public double[][] sampleComposed(double[][] xs, int n, int steps, SplittableRandom random) {
        int dt = tMean.length;
        Denoiser.Conditional model = model();
        Tensor noise = Tensor.normal(Shape.of(n, dt), 0, 1, random);
        if (!score) {
            if (xs.length != 1) {
                throw new IllegalArgumentException("flow matching takes one observation; use a score model");
            }
            Tensor cond = SbiArrays.repeated(xs[0], n, xMean, xSd);
            Tensor out = Var.noGrad(() -> Var.of(FlowMatching.sample(model.given(cond), noise, steps,
                    FlowMatching.Solver.HEUN, null))).value();
            return SbiArrays.unstandardise(SbiArrays.rows(out), tMean, tSd);
        }
        Tensor[] conds = new Tensor[xs.length];
        for (int i = 0; i < xs.length; i++) {
            conds[i] = SbiArrays.repeated(xs[i], n, xMean, xSd);
        }
        Tensor th = noise;
        double dtStep = (1 - T_MIN) / steps;
        for (int k = 0; k < steps; k++) {
            double t = 1 - k * dtStep;
            Tensor sc = composedScore(model, th, t, conds);
            float bt = (float) beta(t);
            Tensor drift = th.mul(0.5f * bt).add(sc.mul(bt));
            th = th.add(drift.mul((float) dtStep));
            if (k < steps - 1) {
                th = th.add(Tensor.normal(Shape.of(n, dt), 0, 1, random).mul((float) Math.sqrt(bt * dtStep)));
            }
        }
        if (conds.length > 1) {
            // The composed diffused score is only exact as t -> 0 (at intermediate times it is
            // too concentrated, Geffner et al. 2023), so the reverse SDE ends too narrow; finish
            // with unadjusted Langevin at a small time, whose target is the composed posterior
            // smoothed by a tiny noise level. Step size from the score's signal-to-noise ratio
            // (Song et al. 2021 predictor-corrector).
            for (int k = 0; k < steps; k++) {
                Tensor s2 = composedScore(model, th, LANGEVIN_TIME, conds);
                Tensor z = Tensor.normal(Shape.of(n, dt), 0, 1, random);
                double sn = Math.sqrt(s2.mul(s2).sum().item() / n);
                double zn = Math.sqrt(z.mul(z).sum().item() / n);
                double eps = 2 * Math.pow(CORRECTOR_SNR * zn / Math.max(sn, 1e-12), 2);
                th = th.add(s2.mul((float) eps)).add(z.mul((float) Math.sqrt(2 * eps)));
            }
        }
        return SbiArrays.unstandardise(SbiArrays.rows(th), tMean, tSd);
    }

    private static final double LANGEVIN_TIME = 0.01;

    private static final double CORRECTOR_SNR = 0.16;

    /** {@code sum_i s(theta_t | x_i) - (n - 1) s_prior(theta_t)} with {@code s_prior = -theta_t}. */
    private Tensor composedScore(Denoiser.Conditional model, Tensor th, double t, Tensor[] conds) {
        int n = th.shape().dim(0);
        double al = alpha(t);
        float sigma = (float) Math.sqrt(1 - al * al);
        float[] tv = new float[n];
        java.util.Arrays.fill(tv, (float) t);
        Tensor time = Tensor.of(Shape.of(n), tv);
        Tensor sc = null;
        for (Tensor c : conds) {
            Tensor epsHat = Var.noGrad(() -> model.apply(Var.of(th), time, c)).value();
            Tensor s = epsHat.mul(-1f / sigma);
            sc = sc == null ? s : sc.add(s);
        }
        if (conds.length > 1) {
            sc = sc.add(th.mul((float) (conds.length - 1)));
        }
        return sc;
    }

    /**
     * The simulations used.
     *
     * @return the count
     */
    public long simulations() {
        return simulations;
    }
}
