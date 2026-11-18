package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.experimental.ml.NormalTails;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * The likelihood-ratio membership inference attack, LiRA (Carlini et al., "Membership
 * Inference Attacks From First Principles", IEEE S&amp;P 2022,
 * <a href="https://arxiv.org/abs/2112.03570">arXiv:2112.03570</a>). The attacker trains many
 * shadow models, each on a random half of a data pool, so every example is "in" for about half
 * of them and "out" for the rest. For each example it fits Gaussians to the shadow models'
 * logit-scaled confidence {@code φ = log(p / (1 - p))} of the true label, in the in and out
 * groups, and scores the target model by the likelihood ratio of its {@code φ}:
 *
 * <ul>
 *   <li>online: {@code log N(φ; μ_in, σ_in²) - log N(φ; μ_out, σ_out²)};</li>
 *   <li>offline (no shadow model trained on the example): the one-sided test
 *       {@code -log P(Z > φ)} under the out Gaussian.</li>
 * </ul>
 *
 * <p>With few shadow models the variances are pooled over examples ("fixed variance"), as
 * the paper recommends below 64 models. Attacks are judged by their true-positive rate at very
 * low false-positive rates ({@link #tprAtFpr}), where a global loss threshold is nearly useless:
 * per-example calibration is what makes LiRA strong.
 */
public final class LikelihoodRatioAttack {

    /** Trains one model on a subset of the pool and reports its confidences. */
    public interface ShadowTrainer {

        /**
         * Trains a model on the pool's members.
         *
         * @param members which pool examples are in the training set
         * @param seed the training seed
         * @return the trained model's probability of the true label, for every pool example
         */
        double[] confidences(boolean[] members, long seed);
    }

    /**
     * Attack quality.
     *
     * @param auc the area under the ROC curve
     * @param tprAt01 the true-positive rate at 0.1% false positives
     * @param tprAt1 the true-positive rate at 1% false positives
     */
    public record Quality(double auc, double tprAt01, double tprAt1) {
    }

    private final double[][] phi;
    private final boolean[][] in;
    private final boolean fixedVariance;

    private LikelihoodRatioAttack(double[][] phi, boolean[][] in, boolean fixedVariance) {
        this.phi = phi;
        this.in = in;
        this.fixedVariance = fixedVariance;
    }

    /**
     * Trains the shadow models.
     *
     * @param poolSize the number of pool examples
     * @param shadows the number of shadow models
     * @param trainer trains one model
     * @param fixedVariance pool the variances over examples
     * @param seed the seed of the random halves
     * @return the fitted attack
     */
    public static LikelihoodRatioAttack fit(int poolSize, int shadows, ShadowTrainer trainer, boolean fixedVariance,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] phi = new double[shadows][];
        boolean[][] in = new boolean[shadows][poolSize];
        for (int s = 0; s < shadows; s++) {
            // balanced halves: each example in for exactly half the models when shadows is even
            for (int i = 0; i < poolSize; i++) {
                in[s][i] = s % 2 == 0 ? r.nextBoolean() : !in[s - 1][i];
            }
            phi[s] = logitScale(trainer.confidences(in[s], seed * 1000 + s));
        }
        return new LikelihoodRatioAttack(phi, in, fixedVariance);
    }

    /** @return the number of shadow models */
    public int shadows() {
        return phi.length;
    }

    /**
     * @param s a shadow model
     * @return its membership mask
     */
    public boolean[] members(int s) {
        return in[s].clone();
    }

    /**
     * @param s a shadow model
     * @return its logit-scaled confidences
     */
    public double[] confidences(int s) {
        return phi[s].clone();
    }

    /**
     * Logit scaling, numerically stable near 0 and 1.
     *
     * @param p probabilities of the true label
     * @return {@code log(p / (1 - p))}
     */
    public static double[] logitScale(double[] p) {
        double[] out = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            double q = Math.min(Math.max(p[i], 1e-15), 1 - 1e-15);
            out[i] = Math.log(q) - Math.log1p(-q);
        }
        return out;
    }

    /**
     * Scores a target model.
     *
     * @param target the target's probabilities of the true label
     * @param online the online (both Gaussians) or offline (out Gaussian only) test
     * @return membership scores, higher meaning member
     */
    public double[] score(double[] target, boolean online) {
        return score(logitScale(target), online, -1);
    }

    /**
     * Scores logit-scaled confidences, leaving one shadow model out of the statistics.
     *
     * @param phiTarget the target's logit-scaled confidences
     * @param online online or offline
     * @param exclude a shadow model to leave out (the target itself), or -1
     * @return the scores
     */
    double[] score(double[] phiTarget, boolean online, int exclude) {
        int n = phiTarget.length;
        double[] muIn = new double[n];
        double[] muOut = new double[n];
        double[] varIn = new double[n];
        double[] varOut = new double[n];
        double pooledIn = 0;
        double pooledOut = 0;
        int countIn = 0;
        int countOut = 0;
        for (int i = 0; i < n; i++) {
            double si = 0;
            double so = 0;
            double si2 = 0;
            double so2 = 0;
            int ni = 0;
            int no = 0;
            for (int s = 0; s < phi.length; s++) {
                if (s == exclude) {
                    continue;
                }
                double v = phi[s][i];
                if (in[s][i]) {
                    si += v;
                    si2 += v * v;
                    ni++;
                } else {
                    so += v;
                    so2 += v * v;
                    no++;
                }
            }
            muIn[i] = ni > 0 ? si / ni : 0;
            muOut[i] = no > 0 ? so / no : 0;
            varIn[i] = ni > 1 ? Math.max(si2 / ni - muIn[i] * muIn[i], 0) * ni / (ni - 1) : 0;
            varOut[i] = no > 1 ? Math.max(so2 / no - muOut[i] * muOut[i], 0) * no / (no - 1) : 0;
            pooledIn += varIn[i] * Math.max(ni - 1, 0);
            pooledOut += varOut[i] * Math.max(no - 1, 0);
            countIn += Math.max(ni - 1, 0);
            countOut += Math.max(no - 1, 0);
        }
        double fixedIn = pooledIn / Math.max(countIn, 1);
        double fixedOut = pooledOut / Math.max(countOut, 1);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double sdIn = Math.sqrt(Math.max(fixedVariance ? fixedIn : varIn[i], 1e-6));
            double sdOut = Math.sqrt(Math.max(fixedVariance ? fixedOut : varOut[i], 1e-6));
            double zOut = (phiTarget[i] - muOut[i]) / sdOut;
            if (online) {
                double zIn = (phiTarget[i] - muIn[i]) / sdIn;
                out[i] = (-0.5 * zIn * zIn - Math.log(sdIn)) - (-0.5 * zOut * zOut - Math.log(sdOut));
            } else {
                out[i] = -NormalTails.logSf(zOut);
            }
        }
        return out;
    }

    /**
     * The standard leave-one-out evaluation: every shadow model in turn is the target,
     * attacked with the statistics of the others.
     *
     * @param online online or offline LiRA
     * @return the attack quality over all (model, example) pairs
     */
    public Quality evaluate(boolean online) {
        int n = phi[0].length;
        double[] scores = new double[phi.length * n];
        boolean[] labels = new boolean[scores.length];
        for (int s = 0; s < phi.length; s++) {
            double[] sc = score(phi[s], online, s);
            for (int i = 0; i < n; i++) {
                scores[s * n + i] = sc[i];
                labels[s * n + i] = in[s][i];
            }
        }
        return quality(scores, labels);
    }

    /**
     * The global-threshold baseline on the same models: score by confidence alone (the loss
     * attack, since the loss is a decreasing function of the confidence).
     *
     * @return the baseline's quality
     */
    public Quality evaluateLossThreshold() {
        int n = phi[0].length;
        double[] scores = new double[phi.length * n];
        boolean[] labels = new boolean[scores.length];
        for (int s = 0; s < phi.length; s++) {
            for (int i = 0; i < n; i++) {
                scores[s * n + i] = phi[s][i];
                labels[s * n + i] = in[s][i];
            }
        }
        return quality(scores, labels);
    }

    /**
     * @param scores membership scores
     * @param members the truth
     * @return AUC and TPR at 0.1% and 1% FPR
     */
    public static Quality quality(double[] scores, boolean[] members) {
        return new Quality(auc(scores, members), tprAtFpr(scores, members, 0.001), tprAtFpr(scores, members, 0.01));
    }

    /**
     * The ROC AUC by ranks (ties count half).
     *
     * @param scores the scores
     * @param positive the truth
     * @return the AUC
     */
    public static double auc(double[] scores, boolean[] positive) {
        Integer[] idx = order(scores);
        double rankSum = 0;
        long pos = 0;
        int i = 0;
        while (i < idx.length) {
            int j = i;
            while (j + 1 < idx.length && scores[idx[j + 1]] == scores[idx[i]]) {
                j++;
            }
            double rank = (i + j) / 2.0 + 1;
            for (int k = i; k <= j; k++) {
                if (positive[idx[k]]) {
                    rankSum += rank;
                    pos++;
                }
            }
            i = j + 1;
        }
        long neg = scores.length - pos;
        return pos == 0 || neg == 0 ? 0.5 : (rankSum - pos * (pos + 1) / 2.0) / ((double) pos * neg);
    }

    /**
     * The true-positive rate at a false-positive rate: the threshold is the smallest one that
     * lets at most {@code fpr} of the negatives through.
     *
     * @param scores the scores
     * @param positive the truth
     * @param fpr the false-positive rate
     * @return the TPR
     */
    public static double tprAtFpr(double[] scores, boolean[] positive, double fpr) {
        int negatives = 0;
        int positives = 0;
        for (boolean p : positive) {
            negatives += p ? 0 : 1;
            positives += p ? 1 : 0;
        }
        double[] neg = new double[negatives];
        int k = 0;
        for (int i = 0; i < scores.length; i++) {
            if (!positive[i]) {
                neg[k++] = scores[i];
            }
        }
        Arrays.sort(neg);
        int allowed = (int) Math.floor(fpr * negatives);
        double threshold = allowed >= negatives ? Double.NEGATIVE_INFINITY : neg[negatives - 1 - allowed];
        int tp = 0;
        for (int i = 0; i < scores.length; i++) {
            if (positive[i] && scores[i] > threshold) {
                tp++;
            }
        }
        return positives == 0 ? 0 : (double) tp / positives;
    }

    private static Integer[] order(double[] scores) {
        Integer[] idx = new Integer[scores.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(scores[a], scores[b]));
        return idx;
    }
}
