package io.github.zzuegg.brainiac.experimental.design;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite;
import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite.DesignProblem;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Sequential design for location finding ({@link SimulatorSuite#locationFinding}) with
 * {@code T} measurements.
 * <ul>
 *   <li>{@link Strategy#DAD}: deep adaptive design (Foster et al., ICML 2021,
 *       <a href="https://arxiv.org/abs/2103.02438">arXiv:2103.02438</a>): a policy maps the
 *       history, encoded by a shared MLP and sum-pooled (permutation invariant), to the next
 *       design; it is trained offline by maximising the sequential PCE bound
 *       {@code log p(h_T | theta_0) - log (1/(L+1)) sum_l p(h_T | theta_l)} by
 *       backpropagation through reparameterised rollouts; deployment is one forward pass per
 *       step;</li>
 *   <li>{@link Strategy#STEP_DAD}: semi-amortised Step-DAD (Hedman et al., ICML 2025,
 *       <a href="https://arxiv.org/abs/2507.14057">arXiv:2507.14057</a>): at a chosen step, a
 *       copy of the policy is fine-tuned on the remaining steps with {@code theta_0} and the
 *       contrasts drawn from the importance-sampling posterior given the history so far;</li>
 *   <li>{@link Strategy#STATIC}: a fixed design sequence trained with the same bound (the
 *       policy sees only the step index);</li>
 *   <li>{@link Strategy#MYOPIC}: greedy re-optimisation of the one-step EIG (nested Monte
 *       Carlo over importance-sampling posterior particles) on a candidate grid at every step,
 *       the classical expensive approach;</li>
 *   <li>{@link Strategy#UNCERTAINTY}: uncertainty sampling, the design where the posterior
 *       predictive mean signal varies most;</li>
 *   <li>{@link Strategy#RANDOM}: uniform designs.</li>
 * </ul>
 */
public final class AdaptiveDesign {

    /** How designs are chosen. */
    public enum Strategy {
        /** Uniform random designs. */
        RANDOM,
        /** A trained, non-adaptive design sequence. */
        STATIC,
        /** The deep adaptive design policy. */
        DAD,
        /** DAD with posterior fine-tuning during the experiment. */
        STEP_DAD,
        /** Greedy one-step EIG re-optimisation. */
        MYOPIC,
        /** Maximal predictive variance. */
        UNCERTAINTY
    }

    private static final int EMBED = 16;
    private static final int HIDDEN = 32;
    private static final float BOX = 4f;

    private final int sources;
    private final int horizon;
    private final boolean adaptive;
    private final SimpleMlp encoder;
    private final SimpleMlp emitter;
    private final long seed;

    private AdaptiveDesign(int sources, int horizon, boolean adaptive, long seed) {
        this.sources = sources;
        this.horizon = horizon;
        this.adaptive = adaptive;
        this.seed = seed;
        this.encoder = SimpleMlp.of(seed, Var::silu, 3, HIDDEN, EMBED);
        this.emitter = SimpleMlp.of(seed + 1, Var::silu, adaptive ? EMBED : horizon, HIDDEN, 2);
    }

    private List<Var> parameters() {
        List<Var> p = new ArrayList<>(emitter.parameters());
        if (adaptive) {
            p.addAll(encoder.parameters());
        }
        return p;
    }

    private AdaptiveDesign copy() {
        AdaptiveDesign c = new AdaptiveDesign(sources, horizon, adaptive, seed);
        List<Var> from = parameters();
        List<Var> to = c.parameters();
        for (int i = 0; i < from.size(); i++) {
            to.get(i).assign(from.get(i).value());
        }
        return c;
    }

    /**
     * The trained policy's horizon.
     *
     * @return {@code T}
     */
    public int horizon() {
        return horizon;
    }

    /**
     * Whether the policy adapts to the history (DAD) or is a static sequence.
     *
     * @return true for DAD
     */
    public boolean adaptive() {
        return adaptive;
    }

    private static Var meanSignal(Tensor[] tx, Tensor[] ty, Var xiX, Var xiY) {
        Var s = null;
        for (int k = 0; k < tx.length; k++) {
            Var dx = Var.of(tx[k]).sub(xiX);
            Var dy = Var.of(ty[k]).sub(xiY);
            Var term = dx.square().add(dy.square()).add(1e-4f).pow(-1f);
            s = s == null ? term : s.add(term);
        }
        return s.add(0.1f).log();
    }

    private Var input(Var pooled, int t, int batch) {
        if (adaptive) {
            return pooled == null ? Var.of(Tensor.zeros(Shape.of(batch, EMBED))) : pooled;
        }
        float[] oh = new float[batch * horizon];
        for (int b = 0; b < batch; b++) {
            oh[b * horizon + t] = 1;
        }
        return Var.of(Tensor.of(Shape.of(batch, horizon), oh));
    }

    private Var encode(Var xi, Var y) {
        return encoder.forward(Var.concat(1, xi.div(BOX), y.div(4f)));
    }

    /** Draws {@code theta} for a batch: column 0 from {@code truth}, the rest from {@code contrast}. */
    private interface ThetaSource {
        double[] draw(boolean truth, SplittableRandom r);
    }

    /**
     * One sPCE gradient step over steps {@code from..T-1}, with a fixed pooled prefix.
     */
    private double step(Optimizer opt, int batch, int contrasts, int from, double[][] prefixDesigns,
            double[] prefixYs, ThetaSource source, SplittableRandom r) {
        int cols = contrasts + 1;
        Tensor[] tx = new Tensor[sources];
        Tensor[] ty = new Tensor[sources];
        float[][] fx = new float[sources][batch * cols];
        float[][] fy = new float[sources][batch * cols];
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < cols; c++) {
                double[] th = source.draw(c == 0, r);
                for (int k = 0; k < sources; k++) {
                    fx[k][b * cols + c] = (float) th[2 * k];
                    fy[k][b * cols + c] = (float) th[2 * k + 1];
                }
            }
        }
        for (int k = 0; k < sources; k++) {
            tx[k] = Tensor.of(Shape.of(batch, cols), fx[k]);
            ty[k] = Tensor.of(Shape.of(batch, cols), fy[k]);
        }
        Var pooled = null;
        if (adaptive && prefixDesigns != null) {
            for (int t = 0; t < prefixYs.length; t++) {
                float[] xi = new float[batch * 2];
                float[] y = new float[batch];
                for (int b = 0; b < batch; b++) {
                    xi[2 * b] = (float) prefixDesigns[t][0];
                    xi[2 * b + 1] = (float) prefixDesigns[t][1];
                    y[b] = (float) prefixYs[t];
                }
                Var e = encode(Var.of(Tensor.of(Shape.of(batch, 2), xi)), Var.of(Tensor.of(Shape.of(batch, 1), y)));
                pooled = pooled == null ? e : pooled.add(e);
            }
        }
        Var ll = null;
        float sigma = (float) SimulatorSuite.LOCATION_NOISE;
        for (int t = from; t < horizon; t++) {
            Var xi = emitter.forward(input(pooled, t, batch)).tanh().mul(BOX);
            Var mu = meanSignal(tx, ty, xi.slice(1, 0, 1), xi.slice(1, 1, 2));
            float[] eps = new float[batch];
            for (int b = 0; b < batch; b++) {
                eps[b] = (float) (sigma * r.nextGaussian());
            }
            Var y = mu.slice(1, 0, 1).add(Var.of(Tensor.of(Shape.of(batch, 1), eps)));
            Var term = mu.sub(y).div(sigma).square().mul(-0.5f);
            ll = ll == null ? term : ll.add(term);
            if (adaptive) {
                Var e = encode(xi, y);
                pooled = pooled == null ? e : pooled.add(e);
            }
        }
        Var spce = ll.slice(1, 0, 1).reshape(batch).sub(ll.logsumexp(1, false)).mean();
        Var loss = spce.neg();
        loss.backward();
        opt.step(parameters());
        return spce.value().toDoubleArray()[0] + Math.log(cols);
    }

    private static double[] priorDraw(int sources, SplittableRandom r) {
        double[] t = new double[2 * sources];
        for (int i = 0; i < t.length; i++) {
            t[i] = r.nextGaussian();
        }
        return t;
    }

    /**
     * Trains a policy offline by the sequential PCE bound.
     *
     * @param sources {@code K}
     * @param horizon {@code T}
     * @param adaptive true for DAD, false for a static sequence
     * @param iterations Adam steps
     * @param batch rollouts per step
     * @param contrasts contrastive samples {@code L}
     * @param seed the seed
     * @return the policy
     */
    public static AdaptiveDesign train(int sources, int horizon, boolean adaptive, int iterations, int batch,
            int contrasts, long seed) {
        AdaptiveDesign p = new AdaptiveDesign(sources, horizon, adaptive, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 7);
        for (int it = 0; it < iterations; it++) {
            p.step(opt, batch, contrasts, 0, null, null, (truth, rr) -> priorDraw(sources, rr), r);
        }
        return p;
    }

    /**
     * The policy's next design for one history.
     *
     * @param designs past designs
     * @param ys past outcomes
     * @return the next design
     */
    public double[] next(double[][] designs, double[] ys) {
        int t = ys.length;
        Tensor out = Var.noGrad(() -> {
            Var pooled = null;
            if (adaptive) {
                for (int s = 0; s < t; s++) {
                    Var e = encode(Var.of(Tensor.of(Shape.of(1, 2), (float) designs[s][0], (float) designs[s][1])),
                            Var.of(Tensor.of(Shape.of(1, 1), (float) ys[s])));
                    pooled = pooled == null ? e : pooled.add(e);
                }
            }
            return emitter.forward(input(pooled, Math.min(t, horizon - 1), 1)).tanh().mul(BOX);
        }).value();
        double[] v = out.toDoubleArray();
        return new double[] {v[0], v[1]};
    }

    /**
     * A finished experiment.
     *
     * @param designs the designs
     * @param ys the outcomes
     * @param secondsPerStep mean decision time per step
     */
    public record Rollout(double[][] designs, double[] ys, double secondsPerStep) {
    }

    /**
     * Settings of the instance-time strategies.
     *
     * @param particles prior draws for the importance-sampling posterior
     * @param candidates grid points per axis for MYOPIC and UNCERTAINTY
     * @param outer NMC outer samples for MYOPIC
     * @param inner NMC inner samples for MYOPIC
     * @param refineAt the step at which STEP_DAD fine-tunes (0 for none)
     * @param refineIterations STEP_DAD fine-tuning steps
     */
    public record Settings(int particles, int candidates, int outer, int inner, int refineAt, int refineIterations) {
    }

    /**
     * Runs one experiment.
     *
     * @param strategy the strategy
     * @param policy the trained policy (DAD or static; ignored by the other strategies)
     * @param theta0 the true source positions
     * @param horizon {@code T}
     * @param settings the instance-time settings
     * @param random the random source (measurement noise and sampling)
     * @return the rollout
     */
    public static Rollout run(Strategy strategy, AdaptiveDesign policy, double[] theta0, int horizon, Settings settings,
            SplittableRandom random) {
        int k = theta0.length / 2;
        DesignProblem problem = SimulatorSuite.locationFinding(k);
        double[][] designs = new double[horizon][];
        double[] ys = new double[horizon];
        double[][] pool = new double[settings.particles()][];
        double[] logw = new double[pool.length];
        for (int i = 0; i < pool.length; i++) {
            pool[i] = priorDraw(k, random);
        }
        AdaptiveDesign active = policy;
        long nanos = 0;
        for (int t = 0; t < horizon; t++) {
            long start = System.nanoTime();
            double[][] hd = java.util.Arrays.copyOf(designs, t);
            double[] hy = java.util.Arrays.copyOf(ys, t);
            double[] xi = switch (strategy) {
                case RANDOM -> new double[] {BOX * (2 * random.nextDouble() - 1), BOX * (2 * random.nextDouble() - 1)};
                case STATIC, DAD -> policy.next(hd, hy);
                case STEP_DAD -> {
                    if (t == settings.refineAt() && t > 0) {
                        double[][] post = resample(pool, logw, 512, random);
                        active = policy.copy();
                        Optimizer opt = Optimizer.adam(1e-3f);
                        for (int it = 0; it < settings.refineIterations(); it++) {
                            active.step(opt, 32, 63, t, hd, hy, (truth, rr) -> post[rr.nextInt(post.length)], random);
                        }
                    }
                    yield active.next(hd, hy);
                }
                case MYOPIC -> myopic(problem, pool, logw, settings, random);
                case UNCERTAINTY -> uncertainty(pool, logw, settings, random);
            };
            nanos += System.nanoTime() - start;
            designs[t] = xi;
            ys[t] = problem.simulate(theta0, xi, random);
            for (int i = 0; i < pool.length; i++) {
                logw[i] += problem.logLikelihood(ys[t], pool[i], xi);
            }
        }
        return new Rollout(designs, ys, nanos * 1e-9 / horizon);
    }

    private static double[][] resample(double[][] pool, double[] logw, int n, SplittableRandom r) {
        double m = Double.NEGATIVE_INFINITY;
        for (double v : logw) {
            m = Math.max(m, v);
        }
        double[] cum = new double[pool.length];
        double s = 0;
        for (int i = 0; i < pool.length; i++) {
            s += Math.exp(logw[i] - m);
            cum[i] = s;
        }
        double[][] out = new double[n][];
        for (int j = 0; j < n; j++) {
            double u = r.nextDouble() * s;
            int idx = java.util.Arrays.binarySearch(cum, u);
            idx = idx >= 0 ? idx : Math.min(pool.length - 1, -idx - 1);
            out[j] = pool[idx];
        }
        return out;
    }

    private static double[][] candidates(int perAxis) {
        double[][] c = new double[perAxis * perAxis][];
        for (int i = 0; i < perAxis; i++) {
            for (int j = 0; j < perAxis; j++) {
                c[i * perAxis + j] = new double[] {-BOX + 2 * BOX * i / (perAxis - 1),
                    -BOX + 2 * BOX * j / (perAxis - 1)};
            }
        }
        return c;
    }

    private static double[] myopic(DesignProblem problem, double[][] pool, double[] logw, Settings s,
            SplittableRandom r) {
        double[][] post = resample(pool, logw, s.outer() + s.inner(), r);
        double[] best = null;
        double bv = Double.NEGATIVE_INFINITY;
        double[] l = new double[s.inner()];
        for (double[] xi : candidates(s.candidates())) {
            double eig = 0;
            for (int o = 0; o < s.outer(); o++) {
                double[] t = post[o];
                double y = problem.simulate(t, xi, r);
                double m = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < s.inner(); i++) {
                    l[i] = problem.logLikelihood(y, post[s.outer() + i], xi);
                    m = Math.max(m, l[i]);
                }
                double z = 0;
                for (double v : l) {
                    z += Math.exp(v - m);
                }
                eig += problem.logLikelihood(y, t, xi) - (m + Math.log(z / s.inner()));
            }
            if (eig > bv) {
                bv = eig;
                best = xi;
            }
        }
        return best;
    }

    private static double[] uncertainty(double[][] pool, double[] logw, Settings s, SplittableRandom r) {
        double[][] post = resample(pool, logw, s.inner(), r);
        double[] best = null;
        double bv = Double.NEGATIVE_INFINITY;
        for (double[] xi : candidates(s.candidates())) {
            double m = 0;
            double v = 0;
            for (double[] t : post) {
                m += SimulatorSuite.locationMean(t, xi) / post.length;
            }
            for (double[] t : post) {
                double d = SimulatorSuite.locationMean(t, xi) - m;
                v += d * d;
            }
            if (v > bv) {
                bv = v;
                best = xi;
            }
        }
        return best;
    }

    /**
     * The importance-sampling posterior mean of the source positions after a rollout.
     *
     * @param rollout the rollout
     * @param sources {@code K}
     * @param particles prior draws
     * @param random the random source
     * @return the posterior mean
     */
    public static double[] posteriorMean(Rollout rollout, int sources, int particles, SplittableRandom random) {
        DesignProblem problem = SimulatorSuite.locationFinding(sources);
        double[] mean = new double[2 * sources];
        double[][] th = new double[particles][];
        double[] lw = new double[particles];
        double m = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < particles; i++) {
            th[i] = priorDraw(sources, random);
            for (int t = 0; t < rollout.ys().length; t++) {
                lw[i] += problem.logLikelihood(rollout.ys()[t], th[i], rollout.designs()[t]);
            }
            m = Math.max(m, lw[i]);
        }
        double z = 0;
        for (int i = 0; i < particles; i++) {
            double w = Math.exp(lw[i] - m);
            z += w;
            for (int c = 0; c < mean.length; c++) {
                mean[c] += w * th[i][c];
            }
        }
        for (int c = 0; c < mean.length; c++) {
            mean[c] /= z;
        }
        return mean;
    }
}
