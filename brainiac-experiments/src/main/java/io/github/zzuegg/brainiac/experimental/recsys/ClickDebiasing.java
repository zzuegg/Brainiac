package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Unbiased learning to rank from click logs. Clicks confound relevance with position: under the
 * position-based model (PBM) {@code P(click at rank k) = theta_k * gamma(d)}, the examination
 * probability {@code theta_k} times the relevance (attractiveness) {@code gamma}. A ranker trained
 * on raw clicks learns to reproduce the logging ranker's order.
 * <ul>
 *   <li>{@link #ipwLoss} (Joachims et al., WSDM 2017, <a href="https://arxiv.org/abs/1608.04468">
 *       arXiv:1608.04468</a>): counterfactual LTR; each click is weighted by
 *       {@code 1 / theta_k} in a listwise softmax loss, which is unbiased for the relevance-based
 *       loss when the propensities are right (here: known, or from a swap experiment).</li>
 *   <li>{@link #regressionEm} (Wang et al., WSDM 2018,
 *       <a href="https://doi.org/10.1145/3159652.3159732">doi:10.1145/3159652.3159732</a>): fit the
 *       PBM itself; the E-step computes the posteriors of the hidden "examined" and "relevant"
 *       variables of every unclicked impression, the M-step re-estimates {@code theta_k} in closed
 *       form and regresses relevance on features (the ranker) against the soft labels.</li>
 *   <li>{@link #dualLearning} (Ai et al., SIGIR 2018, <a href="https://arxiv.org/abs/1804.05938">
 *       arXiv:1804.05938</a>): the dual learning algorithm. A ranker and a propensity model are
 *       trained jointly, each with listwise softmax losses on the clicks inversely weighted by the
 *       other: the ranker by the estimated propensity ratio {@code theta_1 / theta_k}, the
 *       propensity model by the ranker's relevance ratio.</li>
 *   <li>{@link #cascadeLoss}: the cascade-model correction: only results at or above the last click
 *       count as examined, and the ranker is fitted to clicks among those (pointwise); right under a
 *       cascade, wrong under a PBM.</li>
 * </ul>
 * The ranker is any core {@link Model} from document features to one score.
 */
public final class ClickDebiasing {

    private ClickDebiasing() {
    }

    /**
     * Logged impressions: {@code k} ranked results per session.
     *
     * @param features {@code [sessions][k][d]} the shown documents' features, top first
     * @param clicks {@code [sessions][k]} 0/1
     */
    public record Impressions(double[][][] features, int[][] clicks) {

        /** @return the number of sessions */
        public int sessions() {
            return clicks.length;
        }

        /** @return results per session */
        public int depth() {
            return clicks[0].length;
        }

        /** @return the feature dimension */
        public int dim() {
            return features[0][0].length;
        }
    }

    /**
     * The listwise softmax loss with per-position click weights,
     * {@code -sum_k w_k c_k log softmax(s)_k} averaged over sessions.
     *
     * @param scores {@code [sessions, k]} ranker scores
     * @param clicks {@code [sessions, k]} 0/1 clicks
     * @param weights per-position weights, such as {@code 1 / theta_k} (or all 1 for naive clicks)
     * @return the scalar loss
     */
    public static Var ipwLoss(Var scores, Tensor clicks, double[] weights) {
        int b = scores.shape().dim(0);
        int k = scores.shape().dim(1);
        float[] w = new float[k];
        for (int p = 0; p < k; p++) {
            w[p] = (float) weights[p];
        }
        Var logp = scores.sub(scores.logsumexp(1, true));
        return logp.mul(clicks).mul(Tensor.of(Shape.of(1, k), w)).sum().div(-b);
    }

    /**
     * Scores a batch of sessions.
     *
     * @param ranker the ranker
     * @param log the impressions
     * @param sessions the sessions to score
     * @param training the training flag
     * @return {@code [sessions, k]} scores
     */
    public static Var score(Model ranker, Impressions log, int[] sessions, boolean training) {
        int k = log.depth();
        int d = log.dim();
        float[] f = new float[sessions.length * k * d];
        for (int i = 0; i < sessions.length; i++) {
            for (int p = 0; p < k; p++) {
                for (int j = 0; j < d; j++) {
                    f[(i * k + p) * d + j] = (float) log.features()[sessions[i]][p][j];
                }
            }
        }
        return ranker.forward(Var.of(Tensor.of(Shape.of(sessions.length * k, d), f)), training)
                .reshape(sessions.length, k);
    }

    /**
     * @param log the impressions
     * @param sessions sessions
     * @return their clicks as {@code [sessions, k]}
     */
    public static Tensor clicks(Impressions log, int[] sessions) {
        int k = log.depth();
        float[] c = new float[sessions.length * k];
        for (int i = 0; i < sessions.length; i++) {
            for (int p = 0; p < k; p++) {
                c[i * k + p] = log.clicks()[sessions[i]][p];
            }
        }
        return Tensor.of(Shape.of(sessions.length, k), c);
    }

    private static int[] batch(int n, int size, SplittableRandom r) {
        int[] b = new int[size];
        for (int i = 0; i < size; i++) {
            b[i] = r.nextInt(n);
        }
        return b;
    }

    /**
     * Trains a ranker with the IPW listwise loss.
     *
     * @param ranker the ranker
     * @param log impressions
     * @param propensity examination per rank (all 1 for naive training)
     * @param steps Adam steps
     * @param batchSessions sessions per step
     * @param lr learning rate
     * @param seed the batch seed
     */
    public static void trainIpw(Model ranker, Impressions log, double[] propensity, int steps, int batchSessions,
            float lr, long seed) {
        double[] w = new double[propensity.length];
        for (int p = 0; p < w.length; p++) {
            w[p] = 1 / Math.max(propensity[p], 1e-3);
        }
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int[] b = batch(log.sessions(), batchSessions, r);
            ipwLoss(score(ranker, log, b, true), clicks(log, b), w).backward();
            opt.step(ranker.parameters());
        }
    }

    /**
     * Regression EM for the position-based model. The ranker outputs the relevance logit.
     *
     * @param ranker the relevance model, trained in place
     * @param log impressions
     * @param iterations EM iterations
     * @param stepsPerIteration Adam steps of the M-step regression
     * @param batchSessions sessions per step
     * @param lr learning rate
     * @param seed the seed
     * @return the estimated examination probability per rank, normalised so rank 1 is 1
     */
    public static double[] regressionEm(Model ranker, Impressions log, int iterations, int stepsPerIteration,
            int batchSessions, float lr, long seed) {
        int n = log.sessions();
        int k = log.depth();
        double[] theta = new double[k];
        java.util.Arrays.fill(theta, 0.5);
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int[] all = new int[n];
        for (int i = 0; i < n; i++) {
            all[i] = i;
        }
        for (int it = 0; it < iterations; it++) {
            float[] gamma = Var.noGrad(() -> score(ranker, log, all, false).sigmoid()).value().toArray();
            double[] examined = new double[k];
            float[] soft = new float[n * k];
            for (int i = 0; i < n; i++) {
                for (int p = 0; p < k; p++) {
                    double g = gamma[i * k + p];
                    if (log.clicks()[i][p] == 1) {
                        examined[p] += 1;
                        soft[i * k + p] = 1;
                    } else {
                        double denom = 1 - theta[p] * g;
                        examined[p] += theta[p] * (1 - g) / denom;
                        soft[i * k + p] = (float) ((1 - theta[p]) * g / denom);
                    }
                }
            }
            for (int p = 0; p < k; p++) {
                theta[p] = Math.clamp(examined[p] / n, 1e-3, 1);
            }
            Tensor targets = Tensor.of(Shape.of(n, k), soft);
            for (int s = 0; s < stepsPerIteration; s++) {
                int[] b = batch(n, batchSessions, r);
                Var logit = score(ranker, log, b, true);
                float[] t = new float[b.length * k];
                float[] tv = targets.toArray();
                for (int i = 0; i < b.length; i++) {
                    System.arraycopy(tv, b[i] * k, t, i * k, k);
                }
                logit.softplus().sub(logit.mul(Tensor.of(Shape.of(b.length, k), t))).mean().backward();
                opt.step(ranker.parameters());
            }
        }
        double[] out = new double[k];
        for (int p = 0; p < k; p++) {
            out[p] = theta[p] / theta[0];
        }
        return out;
    }

    /**
     * The dual learning algorithm: trains the ranker and a per-rank propensity model jointly.
     *
     * @param ranker the ranker, trained in place
     * @param log impressions
     * @param steps Adam steps
     * @param batchSessions sessions per step
     * @param lr learning rate
     * @param seed the seed
     * @return the estimated examination per rank, normalised so rank 1 is 1
     */
    public static double[] dualLearning(Model ranker, Impressions log, int steps, int batchSessions, float lr,
            long seed) {
        int k = log.depth();
        Var propensityLogits = Var.param(Tensor.zeros(Shape.of(1, k)));
        Optimizer rankOpt = Optimizer.adam(lr);
        Optimizer propOpt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int[] b = batch(log.sessions(), batchSessions, r);
            Tensor c = clicks(log, b);
            Var scores = score(ranker, log, b, true);
            Var logRel = scores.sub(scores.logsumexp(1, true));
            Var logExam = propensityLogits.sub(propensityLogits.logsumexp(1, true));
            // ranker weights: theta_1 / theta_k from the propensity model (no gradient)
            float[] exam = logExam.value().toArray();
            double[] wr = new double[k];
            for (int p = 0; p < k; p++) {
                wr[p] = Math.min(20, Math.exp(exam[0] - exam[p]));
            }
            Var rankLoss = logRel.mul(c).mul(Tensor.of(Shape.of(1, k), toFloat(wr))).sum().div(-b.length);
            // propensity weights: rel_1 / rel_k per session from the ranker (no gradient)
            float[] rel = logRel.value().toArray();
            float[] wp = new float[b.length * k];
            for (int i = 0; i < b.length; i++) {
                for (int p = 0; p < k; p++) {
                    wp[i * k + p] = (float) Math.min(20, Math.exp(rel[i * k] - rel[i * k + p]));
                }
            }
            Var propLoss = logExam.mul(c).mul(Tensor.of(Shape.of(b.length, k), wp)).sum().div(-b.length);
            rankLoss.backward();
            rankOpt.step(ranker.parameters());
            propLoss.backward();
            propOpt.step(java.util.List.of(propensityLogits));
        }
        float[] logits = propensityLogits.value().toArray();
        double[] out = new double[k];
        for (int p = 0; p < k; p++) {
            out[p] = Math.exp(logits[p] - logits[0]);
        }
        return out;
    }

    /**
     * The cascade correction: a pointwise logistic loss on the impressions at or above each
     * session's last click (sessions without clicks count as fully examined).
     *
     * @param scores {@code [sessions, k]} relevance logits
     * @param clicks {@code [sessions, k]} 0/1
     * @return the scalar loss
     */
    public static Var cascadeLoss(Var scores, Tensor clicks) {
        int b = scores.shape().dim(0);
        int k = scores.shape().dim(1);
        float[] c = clicks.toArray();
        float[] examined = new float[b * k];
        for (int i = 0; i < b; i++) {
            int last = k - 1;
            for (int p = k - 1; p >= 0; p--) {
                if (c[i * k + p] > 0) {
                    last = p;
                    break;
                }
            }
            for (int p = 0; p <= last; p++) {
                examined[i * k + p] = 1;
            }
        }
        Tensor mask = Tensor.of(Shape.of(b, k), examined);
        float total = Math.max(1, mask.sum().item());
        return scores.softplus().sub(scores.mul(clicks)).mul(mask).sum().div(total);
    }

    private static float[] toFloat(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return f;
    }
}
