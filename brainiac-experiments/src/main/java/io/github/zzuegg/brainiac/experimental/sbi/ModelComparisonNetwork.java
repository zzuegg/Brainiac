package io.github.zzuegg.brainiac.experimental.sbi;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Prior;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Simulator;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Amortised Bayesian model comparison from simulations of the candidate models.
 * <ul>
 *   <li>{@link Loss#CLASSIFIER}: a softmax classifier trained by cross-entropy on simulations
 *       drawn with equal model priors learns posterior model probabilities {@code p(M | x)}
 *       (Radev et al., IEEE TNNLS 2023); log Bayes factors are logit differences, which
 *       saturate where the classes barely overlap;</li>
 *   <li>{@link Loss#EVIDENCE_NETWORK}: evidence networks (Jeffrey and Wandelt 2023,
 *       <a href="https://arxiv.org/abs/2305.11241">arXiv:2305.11241</a>) for two models: the
 *       exponential loss {@code E[exp((1/2 - m) f(x))]} is minimised exactly by
 *       {@code f = log p(x | M1) / p(x | M0)}; the output passes through a leaky parity-odd
 *       power {@code f = o + a o |o|} ("l-POP") so large log Bayes factors need only moderate
 *       activations.</li>
 * </ul>
 * {@link #calibrationError} is the expected calibration error of the predicted probabilities.
 */
public final class ModelComparisonNetwork {

    /** The training objective. */
    public enum Loss {
        /** Cross-entropy classification over K models. */
        CLASSIFIER,
        /** Exponential evidence-network loss for two models. */
        EVIDENCE_NETWORK
    }

    private static final float POP = 0.1f;

    private final Loss loss;
    private final SimpleMlp net;
    private final double[] xMean;
    private final double[] xSd;
    private final int models;

    private ModelComparisonNetwork(Loss loss, SimpleMlp net, double[] xMean, double[] xSd, int models) {
        this.loss = loss;
        this.net = net;
        this.xMean = xMean;
        this.xSd = xSd;
        this.models = models;
    }

    /**
     * Trains on {@code simulations} draws, the model index uniform.
     *
     * @param loss the objective (EVIDENCE_NETWORK needs exactly two models)
     * @param priors the models' priors
     * @param simulators the models' simulators
     * @param simulations the budget
     * @param hidden hidden width
     * @param iterations Adam steps on batches of 128
     * @param seed the seed
     * @return the network
     */
    public static ModelComparisonNetwork train(Loss loss, List<Prior> priors, List<Simulator> simulators,
            int simulations, int hidden, int iterations, long seed) {
        Objects.requireNonNull(loss, "loss");
        int k = priors.size();
        if (loss == Loss.EVIDENCE_NETWORK && k != 2) {
            throw new IllegalArgumentException("evidence networks compare two models");
        }
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[simulations][];
        int[] m = new int[simulations];
        for (int i = 0; i < simulations; i++) {
            m[i] = r.nextInt(k);
            x[i] = simulators.get(m[i]).simulate(priors.get(m[i]).sample(r), r);
        }
        double[] xm = SbiArrays.mean(x);
        double[] xs = SbiArrays.sd(x, xm);
        int out = loss == Loss.CLASSIFIER ? k : 1;
        SimpleMlp net = SimpleMlp.of(seed, Var::silu, xm.length, hidden, hidden, out);
        ModelComparisonNetwork mc = new ModelComparisonNetwork(loss, net, xm, xs, k);
        Optimizer opt = Optimizer.adam(1e-3f);
        int b = Math.min(128, simulations);
        for (int it = 0; it < iterations; it++) {
            int[] idx = SbiArrays.batch(simulations, b, r);
            Var o = net.forward(Var.of(SbiArrays.standardisedRows(x, idx, xm, xs)));
            Var l;
            if (loss == Loss.CLASSIFIER) {
                float[] oneHot = new float[b * k];
                for (int i = 0; i < b; i++) {
                    oneHot[i * k + m[idx[i]]] = 1;
                }
                Var picked = o.mul(Var.of(Tensor.of(Shape.of(b, k), oneHot))).sum(1, false);
                l = o.logsumexp(1, false).sub(picked).mean();
            } else {
                float[] sign = new float[b];
                for (int i = 0; i < b; i++) {
                    sign[i] = m[idx[i]] == 0 ? 0.5f : -0.5f;
                }
                Var f = pop(o.reshape(b));
                l = f.mul(Var.of(Tensor.of(Shape.of(b), sign))).clamp(-30f, 30f).exp().mean();
            }
            l.backward();
            opt.step(net.parameters());
        }
        return mc;
    }

    private static Var pop(Var o) {
        return o.add(o.mul(o.abs()).mul(POP));
    }

    /**
     * Log Bayes factors of model 1 against model 0 for several observations.
     *
     * @param xs observations
     * @return {@code log p(x | M1) - log p(x | M0)}
     */
    public double[] logBayesFactors(double[][] xs) {
        Tensor in = SbiArrays.standardised(xs, xMean, xSd);
        double[] o = Var.noGrad(() -> {
            Var v = net.forward(Var.of(in));
            return loss == Loss.CLASSIFIER ? v : pop(v);
        }).value().toDoubleArray();
        double[] out = new double[xs.length];
        for (int i = 0; i < xs.length; i++) {
            out[i] = loss == Loss.CLASSIFIER ? o[i * models + 1] - o[i * models] : o[i];
        }
        return out;
    }

    /**
     * Posterior model probabilities under equal model priors.
     *
     * @param xs observations
     * @return {@code [n][K]}
     */
    public double[][] probabilities(double[][] xs) {
        double[][] p = new double[xs.length][models];
        if (loss == Loss.EVIDENCE_NETWORK) {
            double[] lbf = logBayesFactors(xs);
            for (int i = 0; i < xs.length; i++) {
                p[i][1] = 1 / (1 + Math.exp(-lbf[i]));
                p[i][0] = 1 - p[i][1];
            }
            return p;
        }
        double[] o = Var.noGrad(() -> net.forward(Var.of(SbiArrays.standardised(xs, xMean, xSd)))).value()
                .toDoubleArray();
        for (int i = 0; i < xs.length; i++) {
            double[] row = java.util.Arrays.copyOfRange(o, i * models, (i + 1) * models);
            double z = SbiArrays.logSumExp(row);
            for (int c = 0; c < models; c++) {
                p[i][c] = Math.exp(row[c] - z);
            }
        }
        return p;
    }

    /**
     * Expected calibration error of the top-class probability over 10 equal-width bins.
     *
     * @param probabilities predicted probabilities {@code [n][K]}
     * @param labels true model indices
     * @return the ECE
     */
    public static double calibrationError(double[][] probabilities, int[] labels) {
        double[] conf = new double[10];
        double[] acc = new double[10];
        int[] count = new int[10];
        for (int i = 0; i < labels.length; i++) {
            int best = 0;
            for (int c = 1; c < probabilities[i].length; c++) {
                if (probabilities[i][c] > probabilities[i][best]) {
                    best = c;
                }
            }
            double p = probabilities[i][best];
            int bin = Math.min(9, (int) (p * 10));
            conf[bin] += p;
            acc[bin] += best == labels[i] ? 1 : 0;
            count[bin]++;
        }
        double e = 0;
        for (int b = 0; b < 10; b++) {
            if (count[b] > 0) {
                e += Math.abs(acc[b] - conf[b]) / labels.length;
            }
        }
        return e;
    }
}
