package io.github.zzuegg.brainiac.experimental.sbi;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Prior;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Simulator;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Neural ratio estimation: a classifier {@code f(theta, x)} trained to tell joint pairs from
 * marginal pairs learns the likelihood-to-evidence ratio {@code log r = log p(x|theta)/p(x)},
 * so {@code log p(theta|x) = f + log p(theta)} is sampled by {@link SliceSampler}.
 * <ul>
 *   <li>{@link Variant#NRE_A}: binary cross-entropy, joint against shuffled pairs (Hermans,
 *       Begy and Louppe, ICML 2020, <a href="https://arxiv.org/abs/1903.04057">
 *       arXiv:1903.04057</a>);</li>
 *   <li>{@link Variant#NRE_C}: contrastive NRE (Miller, Weniger and Forré, NeurIPS 2022): a
 *       {@code K+1}-way classifier over {@code K} candidate parameters, class 0 meaning all are
 *       marginal; {@code q(y=k) = gamma r_k / (K + gamma sum_j r_j)} with {@code gamma = 1}, which
 *       makes the ratio exactly normalised at the optimum;</li>
 *   <li>{@link Variant#BNRE}: balanced NRE (Delaunoy et al., NeurIPS 2022,
 *       <a href="https://arxiv.org/abs/2208.13624">arXiv:2208.13624</a>): NRE-A plus
 *       {@code lambda (E_joint[sigmoid f] + E_marginal[sigmoid f] - 1)^2}, which pushes the
 *       classifier towards conservative (wider) posteriors.</li>
 * </ul>
 */
public final class NeuralRatio {

    /** The training objective. */
    public enum Variant {
        /** Binary joint-versus-marginal classification. */
        NRE_A,
        /** Contrastive multiclass classification over K candidates. */
        NRE_C,
        /** NRE-A with the balancing regulariser. */
        BNRE
    }

    private final SimpleMlp net;
    private final double[] tMean;
    private final double[] tSd;
    private final double[] xMean;
    private final double[] xSd;
    private final long simulations;

    private NeuralRatio(SimpleMlp net, double[] tMean, double[] tSd, double[] xMean, double[] xSd, long simulations) {
        this.net = net;
        this.tMean = tMean;
        this.tSd = tSd;
        this.xMean = xMean;
        this.xSd = xSd;
        this.simulations = simulations;
    }

    /**
     * Trains a ratio estimator on prior simulations.
     *
     * @param variant the objective
     * @param prior the prior
     * @param sim the simulator
     * @param simulations the simulation budget
     * @param contrastive {@code K} for NRE-C (ignored otherwise), for example 5
     * @param lambda the balancing weight for BNRE, for example 100
     * @param hidden hidden width
     * @param iterations Adam steps on minibatches of 128
     * @param seed the seed
     * @return the estimator
     */
    public static NeuralRatio train(Variant variant, Prior prior, Simulator sim, int simulations, int contrastive,
            double lambda, int hidden, int iterations, long seed) {
        Objects.requireNonNull(variant, "variant");
        SplittableRandom r = new SplittableRandom(seed);
        double[][] t = new double[simulations][];
        double[][] x = new double[simulations][];
        for (int i = 0; i < simulations; i++) {
            t[i] = prior.sample(r);
            x[i] = sim.simulate(t[i], r);
        }
        double[] tm = SbiArrays.mean(t);
        double[] ts = SbiArrays.sd(t, tm);
        double[] xm = SbiArrays.mean(x);
        double[] xs = SbiArrays.sd(x, xm);
        SimpleMlp net = SimpleMlp.of(seed, Var::silu, tm.length + xm.length, hidden, hidden, 1);
        NeuralRatio nr = new NeuralRatio(net, tm, ts, xm, xs, simulations);
        Optimizer opt = Optimizer.adam(1e-3f);
        int b = Math.min(128, simulations);
        int k = variant == Variant.NRE_C ? Math.max(2, contrastive) : 1;
        for (int it = 0; it < iterations; it++) {
            int[] idx = SbiArrays.batch(simulations, b, r);
            Tensor tb = SbiArrays.standardisedRows(t, idx, tm, ts);
            Tensor xb = SbiArrays.standardisedRows(x, idx, xm, xs);
            Var loss;
            if (variant == Variant.NRE_C) {
                // logits for each x_i against K candidates: joint set (candidate 0 is theta_i)
                // and marginal set (all candidates from other rows)
                Var[] joint = new Var[k];
                Var[] marg = new Var[k];
                for (int c = 0; c < k; c++) {
                    joint[c] = nr.logit(roll(tb, c), xb);
                    marg[c] = nr.logit(roll(tb, c + 1), xb);
                }
                Var fj = Var.concat(1, joint);
                Var fm = Var.concat(1, marg);
                float logK = (float) Math.log(k);
                Var normJ = logNormaliser(fj, logK, b);
                Var normM = logNormaliser(fm, logK, b);
                Var lossJoint = fj.slice(1, 0, 1).reshape(b).sub(normJ).mean().neg();
                Var lossMarg = normM.neg().add(logK).mean().neg();
                loss = lossJoint.add(lossMarg).mul(0.5f);
            } else {
                Var fj = nr.logit(tb, xb).reshape(b);
                Var fm = nr.logit(roll(tb, 1), xb).reshape(b);
                // -log sigmoid(fj) - log(1 - sigmoid(fm)) = softplus(-fj) + softplus(fm)
                loss = fj.neg().softplus().mean().add(fm.softplus().mean());
                if (variant == Variant.BNRE) {
                    Var bal = fj.sigmoid().mean().add(fm.sigmoid().mean()).sub(1f);
                    loss = loss.add(bal.square().mul((float) lambda));
                }
            }
            loss.backward();
            opt.step(net.parameters());
        }
        return nr;
    }

    /** {@code log(K + sum_j exp f_j)} per row, {@code f [B, K]}. */
    private static Var logNormaliser(Var f, float logK, int b) {
        Tensor lk = Tensor.full(io.github.zzuegg.brainiac.tensor.Shape.of(b, 1), logK);
        return Var.concat(1, Var.of(lk), f).logsumexp(1, false);
    }

    private static Tensor roll(Tensor t, int shift) {
        int n = t.shape().dim(0);
        int s = shift % n;
        if (s == 0) {
            return t;
        }
        return Tensor.concat(0, t.slice(0, s, n), t.slice(0, 0, s));
    }

    private Var logit(Tensor theta, Tensor x) {
        return net.forward(Var.concat(1, Var.of(theta), Var.of(x)));
    }

    /**
     * The estimated log ratio for several parameters and one observation.
     *
     * @param thetas parameters {@code [n][d]}
     * @param x the observation
     * @return {@code log r(theta_i, x)}
     */
    public double[] logRatios(double[][] thetas, double[] x) {
        Tensor tb = SbiArrays.standardised(thetas, tMean, tSd);
        Tensor xb = SbiArrays.repeated(x, thetas.length, xMean, xSd);
        return Var.noGrad(() -> logit(tb, xb)).value().toDoubleArray();
    }

    /**
     * The estimated log ratio.
     *
     * @param theta the parameters
     * @param x the observation
     * @return {@code log r(theta, x)}
     */
    public double logRatio(double[] theta, double[] x) {
        return logRatios(new double[][] {theta}, x)[0];
    }

    /**
     * Posterior samples by slice sampling {@code log r + log prior}.
     *
     * @param prior the prior
     * @param x the observation
     * @param n the number of samples
     * @param random the random source
     * @return {@code [n][d]}
     */
    public double[][] posterior(Prior prior, double[] x, int n, SplittableRandom random) {
        double[][] cand = new double[500][];
        for (int i = 0; i < cand.length; i++) {
            cand[i] = prior.sample(random);
        }
        return SliceSampler.sample(th -> {
            double lp = prior.logDensity(th);
            return Double.isFinite(lp) ? lp + logRatio(th, x) : lp;
        }, cand, 4, 60, n, 0.5, random);
    }

    /**
     * Posterior samples by sampling-importance-resampling from prior draws with weights
     * {@code r(theta, x)}: cheap and batched, adequate when the posterior is not much narrower
     * than the prior.
     *
     * @param prior the prior
     * @param x the observation
     * @param n the number of samples
     * @param pool prior draws to resample from
     * @param random the random source
     * @return {@code [n][d]}
     */
    public double[][] posteriorSir(Prior prior, double[] x, int n, int pool, SplittableRandom random) {
        double[][] cand = new double[pool][];
        for (int i = 0; i < pool; i++) {
            cand[i] = prior.sample(random);
        }
        double[] lw = logRatios(cand, x);
        int[] pick = io.github.zzuegg.brainiac.experimental.ml.ProbabilisticProgram.systematic(lw, random);
        double[][] out = new double[n][];
        for (int i = 0; i < n; i++) {
            out[i] = cand[pick[random.nextInt(pick.length)]].clone();
        }
        return out;
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
