package io.github.zzuegg.brainiac.experimental.metric;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Pair-based metric learning losses with mining, each a core {@link Loss} over an embedding
 * batch {@code [n, d]} and its class ids. Which pairs a loss sees matters more than its form,
 * so the mining is part of each loss and happens on the current (detached) distances.
 *
 * <ul>
 *   <li>{@link #contrastive}: the classic siamese loss (Hadsell et al., 2006) on every pair:
 *       {@code d^2} for positives, {@code relu(m - d)^2} for negatives.</li>
 *   <li>{@link #triplet}: FaceNet's triplet loss {@code relu(d(a,p) - d(a,n) + m)} on
 *       L2-normalised embeddings (Schroff et al., CVPR 2015) with one triplet per anchor,
 *       mined at random, hardest-first, or <em>semi-hard</em>: the closest negative that is
 *       still farther than the positive (inside the margin), FaceNet's choice, which avoids the
 *       collapse that the hardest negatives cause early in training.</li>
 *   <li>{@link #multiSimilarity}: the multi-similarity loss (Wang et al., CVPR 2019) on
 *       cosines: pairs are mined by comparing each with the hardest pair of the other kind
 *       (a negative is kept if it is more similar than the least similar positive minus
 *       {@code eps}, a positive if it is less similar than the most similar negative plus
 *       {@code eps}), then weighted by a soft log-sum-exp
 *       {@code 1/alpha log(1 + sum_P e^{-alpha (S - lambda)}) + 1/beta log(1 + sum_N e^{beta (S - lambda)})}.</li>
 * </ul>
 *
 * <p>Musgrave et al. (ECCV 2020) found that under one fair protocol these losses land within a
 * few points of each other; the experiment uses the same embedding size, steps and data for all.
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1503.03832">arXiv:1503.03832</a>,
 * <a href="https://arxiv.org/abs/1904.06627">arXiv:1904.06627</a>,
 * <a href="https://arxiv.org/abs/2003.08505">arXiv:2003.08505</a>.
 */
public final class PairLosses {

    /** How the triplet loss picks a negative (and positive) for each anchor. */
    public enum Mining {
        /** A random positive and a random negative. */
        RANDOM,
        /** The farthest positive and the closest negative ("batch hard"). */
        HARD,
        /** A random positive and the closest negative farther than it (FaceNet). */
        SEMI_HARD
    }

    private PairLosses() {
    }

    /**
     * The contrastive loss over all pairs of the batch.
     *
     * @param margin the distance negatives are pushed beyond, such as 1
     * @return the loss
     */
    public static Loss contrastive(float margin) {
        return (z, target) -> {
            int n = z.shape().dim(0);
            int[] y = MetricMath.labels(target);
            float[] pos = new float[n * n];
            float[] neg = new float[n * n];
            int pairs = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        (y[i] == y[j] ? pos : neg)[i * n + j] = 1;
                        pairs++;
                    }
                }
            }
            Var d = MetricMath.distances(SelfSupervisedLosses.normalize(z));
            var dev = z.value().device();
            Var p = d.square().mul(Tensor.of(Shape.of(n, n), pos, dev));
            Var q = d.neg().add(margin).relu().square().mul(Tensor.of(Shape.of(n, n), neg, dev));
            return p.add(q).sum().div(Math.max(1, pairs));
        };
    }

    /**
     * The triplet loss with one mined triplet per anchor that has a positive and a negative.
     *
     * @param margin the margin on normalised embeddings, 0.2 in FaceNet
     * @param mining how to mine
     * @param seed the seed of random choices
     * @return the loss (stateful: its random stream advances per call)
     */
    public static Loss triplet(float margin, Mining mining, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        return (z, target) -> {
            int n = z.shape().dim(0);
            int[] y = MetricMath.labels(target);
            Var d = MetricMath.distances(SelfSupervisedLosses.normalize(z));
            float[][] dv = MetricMath.matrix(d);
            float[] pm = new float[n * n];
            float[] nm = new float[n * n];
            int anchors = 0;
            for (int i = 0; i < n; i++) {
                int p = pick(dv[i], y, i, true, mining == Mining.HARD ? 1 : 0, random);
                if (p < 0) {
                    continue;
                }
                int q = switch (mining) {
                    case RANDOM -> pick(dv[i], y, i, false, 0, random);
                    case HARD -> pick(dv[i], y, i, false, -1, random);
                    case SEMI_HARD -> semiHard(dv[i], y, i, dv[i][p], margin);
                };
                if (q < 0) {
                    continue;
                }
                pm[i * n + p] = 1;
                nm[i * n + q] = 1;
                anchors++;
            }
            if (anchors == 0) {
                return z.mul(0f).sum();
            }
            var dev = z.value().device();
            Var dap = d.mul(Tensor.of(Shape.of(n, n), pm, dev)).sum(1, false);
            Var dan = d.mul(Tensor.of(Shape.of(n, n), nm, dev)).sum(1, false);
            float[] has = new float[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    has[i] += pm[i * n + j];
                }
            }
            return dap.sub(dan).add(margin).relu().mul(Tensor.of(Shape.of(n), has, dev)).sum().div(anchors);
        };
    }

    /** Picks a same-class (or other-class) index: 0 random, 1 farthest, -1 closest; -1 if none. */
    private static int pick(float[] d, int[] y, int i, boolean same, int mode, SplittableRandom random) {
        int best = -1;
        int seen = 0;
        for (int j = 0; j < d.length; j++) {
            if (j == i || (y[j] == y[i]) != same) {
                continue;
            }
            seen++;
            if (mode == 0) {
                if (random.nextInt(seen) == 0) {
                    best = j;
                }
            } else if (best < 0 || (mode > 0 ? d[j] > d[best] : d[j] < d[best])) {
                best = j;
            }
        }
        return best;
    }

    /** The closest negative farther than the positive; else the closest negative overall. */
    private static int semiHard(float[] d, int[] y, int i, float dap, float margin) {
        int best = -1;
        int hardest = -1;
        for (int j = 0; j < d.length; j++) {
            if (j == i || y[j] == y[i]) {
                continue;
            }
            if (hardest < 0 || d[j] < d[hardest]) {
                hardest = j;
            }
            if (d[j] > dap && d[j] < dap + margin && (best < 0 || d[j] < d[best])) {
                best = j;
            }
        }
        return best >= 0 ? best : hardest;
    }

    /**
     * The multi-similarity loss with its pair mining.
     *
     * @param alpha the positive scale, 2 in the paper
     * @param beta the negative scale, 50 in the paper
     * @param lambda the similarity offset, 0.5 (1 in the paper's CUB setting)
     * @param epsilon the mining margin, 0.1 in the paper
     * @return the loss
     */
    public static Loss multiSimilarity(float alpha, float beta, float lambda, float epsilon) {
        return (z, target) -> {
            int n = z.shape().dim(0);
            int[] y = MetricMath.labels(target);
            Var s = MetricMath.cosine(z, z);
            float[][] sv = MetricMath.matrix(s);
            float[] pm = new float[n * n];
            float[] nm = new float[n * n];
            int anchors = 0;
            for (int i = 0; i < n; i++) {
                float minPos = Float.POSITIVE_INFINITY;
                float maxNeg = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < n; j++) {
                    if (j == i) {
                        continue;
                    }
                    if (y[j] == y[i]) {
                        minPos = Math.min(minPos, sv[i][j]);
                    } else {
                        maxNeg = Math.max(maxNeg, sv[i][j]);
                    }
                }
                boolean any = false;
                for (int j = 0; j < n; j++) {
                    if (j == i) {
                        continue;
                    }
                    if (y[j] == y[i] && sv[i][j] - epsilon < maxNeg) {
                        pm[i * n + j] = 1;
                        any = true;
                    } else if (y[j] != y[i] && sv[i][j] + epsilon > minPos) {
                        nm[i * n + j] = 1;
                        any = true;
                    }
                }
                anchors += any ? 1 : 0;
            }
            var dev = z.value().device();
            Var posTerm = s.sub(lambda).mul(-alpha).exp().mul(Tensor.of(Shape.of(n, n), pm, dev)).sum(1, false)
                    .add(1f).log().div(alpha);
            Var negTerm = s.sub(lambda).mul(beta).exp().mul(Tensor.of(Shape.of(n, n), nm, dev)).sum(1, false)
                    .add(1f).log().div(beta);
            return posTerm.add(negTerm).sum().div(Math.max(1, anchors));
        };
    }
}
