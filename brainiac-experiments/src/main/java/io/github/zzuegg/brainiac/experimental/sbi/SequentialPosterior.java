package io.github.zzuegg.brainiac.experimental.sbi;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Prior;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Simulator;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Sequential neural posterior estimation for one observation {@code x_o}: later rounds simulate
 * from parameters near the posterior instead of the prior.
 * <ul>
 *   <li>{@link Method#APT_ATOMIC}: automatic posterior transformation (SNPE-C; Greenberg,
 *       Nonnenmacher and Macke, ICML 2019, <a href="https://arxiv.org/abs/1905.07488">
 *       arXiv:1905.07488</a>) proposes from the current {@code q(theta | x_o)} and trains with the
 *       atomic loss: for each pair, the posterior is renormalised over a set of {@code A}
 *       contrasting parameters, {@code -log softmax_j(log q(theta_j | x) - log p(theta_j))}, so
 *       no importance weights are needed; it can leak mass outside the prior support;</li>
 *   <li>{@link Method#TSNPE}: truncated sequential NPE (Deistler, Goncalves and Macke, NeurIPS
 *       2022, <a href="https://arxiv.org/abs/2210.04815">arXiv:2210.04815</a>) proposes from the
 *       prior truncated to the current highest-probability region
 *       {@code {theta : log q(theta | x_o) >= the epsilon-quantile of log q at its own samples}},
 *       found by rejection; maximum likelihood stays valid because the truncation contains
 *       the posterior's support.</li>
 * </ul>
 */
public final class SequentialPosterior {

    /** The proposal and loss. */
    public enum Method {
        /** APT with the atomic loss. */
        APT_ATOMIC,
        /** Truncated sequential NPE. */
        TSNPE
    }

    /**
     * The outcome.
     *
     * @param density the final {@code q(theta | x)}
     * @param samples posterior samples at {@code x_o}
     * @param simulations simulations spent
     * @param leakage share of the final {@code q(theta | x_o)} mass outside the prior support
     * @param diverged whether a round produced non-finite densities
     */
    public record Result(ConditionalDensity density, double[][] samples, long simulations, double leakage,
            boolean diverged) {
    }

    private SequentialPosterior() {
    }

    /**
     * Runs sequential estimation.
     *
     * @param method APT or TSNPE
     * @param prior the prior
     * @param sim the simulator
     * @param observed {@code x_o}
     * @param rounds the number of rounds
     * @param perRound simulations per round
     * @param config the MDN settings (iterations per round)
     * @param atoms APT's contrasting set size, for example 10
     * @param epsilon TSNPE's excluded mass, for example 1e-3
     * @param samples posterior samples to return
     * @param seed the seed
     * @return the result
     */
    public static Result run(Method method, Prior prior, Simulator sim, double[] observed, int rounds, int perRound,
            ConditionalDensity.Config config, int atoms, double epsilon, int samples, long seed) {
        Objects.requireNonNull(method, "method");
        SplittableRandom r = new SplittableRandom(seed);
        List<double[]> ts = new ArrayList<>();
        List<double[]> xs = new ArrayList<>();
        ConditionalDensity q = null;
        boolean diverged = false;
        for (int round = 0; round < rounds; round++) {
            double[][] proposal;
            if (round == 0) {
                proposal = new double[perRound][];
                for (int i = 0; i < perRound; i++) {
                    proposal[i] = prior.sample(r);
                }
            } else if (method == Method.APT_ATOMIC) {
                proposal = inSupport(prior, q, observed, perRound, r);
            } else {
                proposal = truncatedPrior(prior, q, observed, perRound, epsilon, r);
            }
            for (double[] t : proposal) {
                ts.add(t);
                xs.add(sim.simulate(t, r));
            }
            double[][] t = ts.toArray(new double[0][]);
            double[][] x = xs.toArray(new double[0][]);
            if (q == null) {
                q = ConditionalDensity.fit(ConditionalDensity.Kind.MDN, t, x, config, seed);
            } else if (method == Method.TSNPE) {
                q.train(t, x, config.iterations(), seed + round);
            } else {
                atomic(q, prior, t, x, config, atoms, seed + round);
            }
            double check = q.logDensity(q.sample(observed, 1, r)[0], observed);
            diverged |= !Double.isFinite(check);
        }
        double[][] raw = q.sample(observed, samples, r);
        int outside = 0;
        for (double[] v : raw) {
            if (!Double.isFinite(prior.logDensity(v))) {
                outside++;
            }
        }
        double[][] post = inSupport(prior, q, observed, samples, r);
        return new Result(q, post, ts.size(), outside / (double) samples, diverged);
    }

    /** Samples of {@code q(. | x_o)} restricted to the prior support (by rejection, capped). */
    private static double[][] inSupport(Prior prior, ConditionalDensity q, double[] xo, int n, SplittableRandom r) {
        double[][] out = new double[n][];
        int k = 0;
        for (int attempt = 0; attempt < 20 && k < n; attempt++) {
            for (double[] v : q.sample(xo, 2 * n, r)) {
                if (k < n && Double.isFinite(prior.logDensity(v))) {
                    out[k++] = v;
                }
            }
        }
        while (k < n) {
            out[k++] = prior.sample(r);
        }
        return out;
    }

    private static double[][] truncatedPrior(Prior prior, ConditionalDensity q, double[] xo, int n, double epsilon,
            SplittableRandom r) {
        double[][] own = q.sample(xo, 2000, r);
        double[][] cond = new double[own.length][];
        Arrays.fill(cond, xo);
        double[] lq = q.logDensities(own, cond);
        double[] sorted = lq.clone();
        Arrays.sort(sorted);
        double threshold = sorted[(int) Math.floor(epsilon * (sorted.length - 1))];
        double[][] out = new double[n][];
        int k = 0;
        for (int batch = 0; batch < 100 && k < n; batch++) {
            double[][] cand = new double[2000][];
            for (int i = 0; i < cand.length; i++) {
                cand[i] = prior.sample(r);
            }
            double[] l = q.logDensities(cand, padTo(cond, cand.length, xo));
            for (int i = 0; i < cand.length && k < n; i++) {
                if (l[i] >= threshold) {
                    out[k++] = cand[i];
                }
            }
        }
        while (k < n) {   // the region is tiny: fall back to the estimate's own samples
            out[k] = own[k % own.length];
            k++;
        }
        return out;
    }

    private static double[][] padTo(double[][] cond, int n, double[] xo) {
        if (cond.length == n) {
            return cond;
        }
        double[][] c = new double[n][];
        Arrays.fill(c, xo);
        return c;
    }

    private static void atomic(ConditionalDensity q, Prior prior, double[][] t, double[][] x,
            ConditionalDensity.Config config, int atoms, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = t.length;
        double[] logPrior = new double[n];
        for (int i = 0; i < n; i++) {
            logPrior[i] = prior.logDensity(t[i]);
        }
        int b = Math.min(config.batch(), n);
        int a = Math.min(atoms, b);
        for (int it = 0; it < config.iterations(); it++) {
            int[] idx = SbiArrays.batch(n, b, r);
            double[][] tb = new double[b][];
            double[][] xb = new double[b][];
            float[] lp = new float[b];
            for (int i = 0; i < b; i++) {
                tb[i] = t[idx[i]];
                xb[i] = x[idx[i]];
                lp[i] = (float) logPrior[idx[i]];
            }
            Tensor tt = q.standardiseY(tb);
            Tensor xt = q.standardiseC(xb);
            Tensor lpt = Tensor.of(Shape.of(b, 1), lp);
            Var[] logits = new Var[a];
            for (int j = 0; j < a; j++) {
                Var lq = q.logProbStandardised(Var.of(roll(tt, j)), xt).reshape(b, 1);
                logits[j] = lq.sub(roll(lpt, j));
            }
            Var all = Var.concat(1, logits);
            Var loss = all.slice(1, 0, 1).reshape(b).sub(all.logsumexp(1, false)).mean().neg();
            q.step(loss);
        }
    }

    private static Tensor roll(Tensor t, int shift) {
        int n = t.shape().dim(0);
        int s = shift % n;
        return s == 0 ? t : Tensor.concat(0, t.slice(0, s, n), t.slice(0, 0, s));
    }
}
