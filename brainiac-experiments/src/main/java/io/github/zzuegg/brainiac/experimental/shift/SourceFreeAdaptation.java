package io.github.zzuegg.brainiac.experimental.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Source-free domain adaptation by SHOT (Liang et al., ICML 2020,
 * <a href="https://arxiv.org/abs/2002.08546">arXiv:2002.08546</a>): only the trained source
 * model is available. The classifier head is frozen ("hypothesis transfer") and the feature
 * extractor is adapted on unlabelled target data with
 *
 * <ul>
 *   <li>information maximisation ({@link #informationMaximization}): low entropy per example
 *       and high entropy of the batch-mean prediction, which prevents collapse onto one class;
 *       and</li>
 *   <li>self-supervised pseudo-labels ({@link #centroidPseudoLabels}): class centroids in
 *       feature space weighted by the predicted probabilities, labels by nearest centroid in
 *       cosine distance, refined by one k-means step; weighted by {@code beta}.</li>
 * </ul>
 */
public final class SourceFreeAdaptation {

    private final FeatureClassifier model;
    private final float beta;
    private final Optimizer optimizer;

    /**
     * Prepares the adaptation.
     *
     * @param model the source model; its body is adapted in place, its head frozen
     * @param beta the pseudo-label weight, 0.3 in the paper (0 = information maximisation only)
     * @param learningRate the Adam step size
     */
    public SourceFreeAdaptation(FeatureClassifier model, float beta, float learningRate) {
        this.model = model;
        this.beta = beta;
        this.optimizer = Optimizer.adam(learningRate);
    }

    /**
     * Information maximisation.
     *
     * @param logits {@code [n, classes]}
     * @return mean per-example entropy minus the entropy of the mean prediction
     */
    public static Var informationMaximization(Var logits) {
        Var logP = logits.sub(logits.logsumexp(1, true));
        Var p = logP.exp();
        Var condEnt = p.mul(logP).sum(1, false).neg().mean();
        Var mean = p.mean(0, false);
        Var margEnt = mean.mul(mean.add(1e-6f).log()).sum().neg();
        return condEnt.sub(margEnt);
    }

    /**
     * SHOT's pseudo-labels.
     *
     * @param features {@code [n, d]}
     * @param probs the predicted probabilities {@code [n, classes]}
     * @return the label of each row
     */
    public static int[] centroidPseudoLabels(Tensor features, Tensor probs) {
        int n = features.shape().dim(0);
        int d = features.shape().dim(1);
        int k = probs.shape().dim(1);
        double[] f = features.toDoubleArray();
        for (int i = 0; i < n; i++) {   // append 1 and normalise, as SHOT does
            double s = 1;
            for (int j = 0; j < d; j++) {
                s += f[i * d + j] * f[i * d + j];
            }
            s = Math.sqrt(s);
            for (int j = 0; j < d; j++) {
                f[i * d + j] /= s;
            }
        }
        double[] p = probs.toDoubleArray();
        double[][] centroids = new double[k][d];
        double[] mass = new double[k];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < k; c++) {
                mass[c] += p[i * k + c];
                for (int j = 0; j < d; j++) {
                    centroids[c][j] += p[i * k + c] * f[i * d + j];
                }
            }
        }
        for (int c = 0; c < k; c++) {
            for (int j = 0; j < d; j++) {
                centroids[c][j] /= Math.max(1e-8, mass[c]);
            }
        }
        int[] labels = nearest(f, n, d, centroids);
        // one k-means refinement with hard assignments
        double[][] hard = new double[k][d];
        int[] count = new int[k];
        for (int i = 0; i < n; i++) {
            count[labels[i]]++;
            for (int j = 0; j < d; j++) {
                hard[labels[i]][j] += f[i * d + j];
            }
        }
        for (int c = 0; c < k; c++) {
            for (int j = 0; j < d; j++) {
                hard[c][j] = count[c] > 0 ? hard[c][j] / count[c] : centroids[c][j];
            }
        }
        return nearest(f, n, d, hard);
    }

    private static int[] nearest(double[] f, int n, int d, double[][] centroids) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            double best = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < centroids.length; c++) {
                double dot = 0;
                double norm = 1e-12;
                for (int j = 0; j < d; j++) {
                    dot += f[i * d + j] * centroids[c][j];
                    norm += centroids[c][j] * centroids[c][j];
                }
                double cos = dot / Math.sqrt(norm);
                if (cos > best) {
                    best = cos;
                    out[i] = c;
                }
            }
        }
        return out;
    }

    /**
     * One epoch over the target data: pseudo-labels from the current model, then minibatch
     * steps on the body.
     *
     * @param target unlabelled target inputs
     * @param batch the batch size
     * @param random the shuffling source
     * @return the mean loss
     */
    public double epoch(Tensor target, int batch, SplittableRandom random) {
        int n = target.shape().dim(0);
        int[] pseudo = beta > 0 ? centroidPseudoLabels(model.features(target),
                softmax(model.logits(target))) : null;
        int[] order = random.ints(0, Integer.MAX_VALUE).limit(n).toArray();
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        java.util.Arrays.sort(idx, (a, b) -> Integer.compare(order[a], order[b]));
        double total = 0;
        int steps = 0;
        for (int from = 0; from + 2 <= n; from += batch) {
            int to = Math.min(n, from + batch);
            float[] rows = new float[to - from];
            float[] ys = new float[to - from];
            for (int i = from; i < to; i++) {
                rows[i - from] = idx[i];
                ys[i - from] = pseudo == null ? 0 : pseudo[idx[i]];
            }
            Tensor x = target.gather(0, Tensor.of(Shape.of(rows.length), rows));
            Var logits = model.logits(Var.of(x), true);
            Var loss = informationMaximization(logits);
            if (pseudo != null) {
                loss = loss.add(Functional.crossEntropy(logits, Tensor.of(Shape.of(ys.length), ys)).mul(beta));
            }
            loss.backward();
            for (Var p : model.head().parameters()) {
                p.zeroGrad();
            }
            total += loss.value().item();
            steps++;
            optimizer.step(model.body().parameters());
        }
        return total / Math.max(1, steps);
    }

    private static Tensor softmax(Tensor logits) {
        return logits.sub(logits.logsumexp(1, true)).exp();
    }
}
