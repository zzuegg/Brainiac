package io.github.zzuegg.brainiac.experimental.ood;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.KnnIndex;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Selective classification: a classifier that may abstain, judged by its risk–coverage curve.
 * Every method yields a confidence score; thresholding it trades coverage for risk.
 *
 * <ul>
 *   <li>{@link #softmaxResponse}: the maximum softmax probability, the hard-to-beat
 *       baseline.</li>
 *   <li>{@link SelectiveNet}: a selection head {@code g} trained jointly with the prediction
 *       head under a target coverage {@code c}: {@code alpha (r(f, g) + lambda max(0, c -
 *       mean g)^2) + (1 - alpha) CE(h)}, where {@code r = mean(CE g) / mean(g)} is the
 *       selective risk and {@code h} an auxiliary head on all data (Geifman and El-Yaniv, ICML
 *       2019, <a href="https://arxiv.org/abs/1901.09192">arXiv:1901.09192</a>).</li>
 *   <li>{@link #gamblerLoss}: an extra "reserve" output trained with the doubling-rate loss
 *       {@code -log(p_y + p_reserve / o)} for a payoff {@code o} in {@code (1, classes)} (Liu
 *       et al., NeurIPS 2019, <a href="https://arxiv.org/abs/1907.00208">arXiv:1907.00208</a>).</li>
 *   <li>{@link LikelihoodRatio}: post-hoc Neyman–Pearson rejection by the ratio of the input
 *       densities of correctly and wrongly classified examples: Δ-MDS (difference of
 *       tied-Gaussian Mahalanobis scores fitted to correct and to wrong examples), Δ-KNN
 *       (difference of mean log k-NN distances) and their sums with the relative logit
 *       {@code l1 - l2} ({@link #relativeLogit}) (2025,
 *       <a href="https://arxiv.org/abs/2505.15008">arXiv:2505.15008</a>).</li>
 * </ul>
 */
public final class SelectiveClassifier {

    private SelectiveClassifier() {
    }

    /**
     * @param logits {@code [n, classes]}
     * @return the maximum softmax probability
     */
    public static double[] softmaxResponse(Tensor logits) {
        return LogitScores.msp(logits);
    }

    /**
     * @param logits {@code [n, classes]}
     * @return the gap between the two largest logits (RLog)
     */
    public static double[] relativeLogit(Tensor logits) {
        int c = logits.shape().dim(1);
        double[] v = logits.toDoubleArray();
        int n = v.length / c;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double a = Double.NEGATIVE_INFINITY;
            double b = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < c; j++) {
                double x = v[i * c + j];
                if (x > a) {
                    b = a;
                    a = x;
                } else if (x > b) {
                    b = x;
                }
            }
            out[i] = a - b;
        }
        return out;
    }

    /**
     * The gambler's loss.
     *
     * @param logits {@code [n, classes + 1]}, the last column the reservation
     * @param labels class ids {@code [n]}
     * @param payoff {@code o}, between 1 and {@code classes}
     * @return {@code -mean log(p_y + p_reserve / o)}
     */
    public static Var gamblerLoss(Var logits, Tensor labels, float payoff) {
        int n = logits.shape().dim(0);
        int c = logits.shape().dim(1);
        Var p = logits.sub(logits.logsumexp(1, true)).exp();
        float[] ids = labels.toArray();
        float[] oh = new float[n * c];
        for (int i = 0; i < n; i++) {
            oh[i * c + (int) ids[i]] = 1;
            oh[i * c + c - 1] = 1 / payoff;
        }
        return p.mul(Tensor.of(Shape.of(n, c), oh)).sum(1, false).add(1e-7f).log().mean().neg();
    }

    /**
     * @param logits {@code [n, classes + 1]} of a gambler's model
     * @return the confidence {@code 1 - p_reserve}
     */
    public static double[] gamblerConfidence(Tensor logits) {
        int c = logits.shape().dim(1);
        Tensor p = logits.sub(logits.logsumexp(1, true)).exp();
        double[] v = p.toDoubleArray();
        double[] out = new double[v.length / c];
        for (int i = 0; i < out.length; i++) {
            out[i] = 1 - v[i * c + c - 1];
        }
        return out;
    }

    /**
     * @param logits {@code [n, classes + 1]} of a gambler's model
     * @return the predicted class among the real classes
     */
    public static int[] gamblerPredict(Tensor logits) {
        int c = logits.shape().dim(1);
        return logits.slice(1, 0, c - 1).argmax(1).toIntArray();
    }

    /** SelectiveNet's selection and auxiliary heads on top of a feature extractor. */
    public static final class SelectiveNet {

        private final Dense predict;
        private final Dense hidden;
        private final Dense select;
        private final Dense auxiliary;
        private final float coverage;

        /**
         * Creates the heads.
         *
         * @param features the feature size
         * @param classes the number of classes
         * @param coverage the target coverage {@code c}
         * @param seed the seed
         */
        public SelectiveNet(int features, int classes, float coverage, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            this.predict = new Dense(features, classes, Activation.IDENTITY, Init.XAVIER, r);
            this.hidden = new Dense(features, 16, Activation.RELU, Init.HE, r);
            this.select = new Dense(16, 1, Activation.IDENTITY, Init.XAVIER, r);
            this.auxiliary = new Dense(features, classes, Activation.IDENTITY, Init.XAVIER, r);
            this.coverage = coverage;
        }

        /** @return the parameters of all three heads */
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(predict.parameters());
            p.addAll(hidden.parameters());
            p.addAll(select.parameters());
            p.addAll(auxiliary.parameters());
            return p;
        }

        /**
         * @param features {@code [n, d]}
         * @return the prediction logits
         */
        public Var logits(Var features) {
            return predict.forward(features);
        }

        /**
         * @param features {@code [n, d]}
         * @return the selection probability {@code g} {@code [n]}
         */
        public Var selection(Var features) {
            return select.forward(hidden.forward(features)).sigmoid().reshape(features.shape().dim(0));
        }

        /**
         * SelectiveNet's training loss.
         *
         * @param features {@code [n, d]} (with a graph to the body)
         * @param labels class ids
         * @param lambda the coverage penalty weight, 32 in the paper
         * @param alpha the weight of the selective part, 0.5 in the paper
         * @return the loss
         */
        public Var loss(Var features, Tensor labels, float lambda, float alpha) {
            Var ce = Functional.crossEntropyPerRow(logits(features), labels);
            Var g = selection(features);
            Var phi = g.mean();
            Var risk = ce.mul(g).mean().div(phi);
            Var penalty = phi.neg().add(coverage).relu().square().mul(lambda);
            Var aux = Functional.crossEntropy(auxiliary.forward(features), labels);
            return risk.add(penalty).mul(alpha).add(aux.mul(1 - alpha));
        }
    }

    /** Post-hoc likelihood-ratio rejection from correct and wrong examples' feature densities. */
    public static final class LikelihoodRatio {

        private final TiedGaussian correct;
        private final TiedGaussian wrong;
        private final KnnIndex correctIndex;
        private final KnnIndex wrongIndex;
        private final int k;

        private LikelihoodRatio(TiedGaussian correct, TiedGaussian wrong, KnnIndex correctIndex, KnnIndex wrongIndex,
                int k) {
            this.correct = correct;
            this.wrong = wrong;
            this.correctIndex = correctIndex;
            this.wrongIndex = wrongIndex;
            this.k = k;
        }

        /**
         * Fits both densities on held-out (or training) examples.
         *
         * @param features {@code [n, d]}
         * @param labels the true classes
         * @param predicted the classifier's predictions
         * @param classes the number of classes
         * @param k the neighbour count of Δ-KNN, such as 25
         * @return the rejector
         */
        public static LikelihoodRatio fit(Tensor features, int[] labels, int[] predicted, int classes, int k) {
            int n = labels.length;
            int d = features.shape().dim(1);
            float[] f = features.toArray();
            List<Integer> ok = new ArrayList<>();
            List<Integer> bad = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                (labels[i] == predicted[i] ? ok : bad).add(i);
            }
            if (bad.isEmpty()) {
                throw new IllegalArgumentException("need at least one wrongly classified example");
            }
            KnnIndex ci = KnnIndex.create(d, 1, KnnIndex.Metric.L2);
            KnnIndex wi = KnnIndex.create(d, 1, KnnIndex.Metric.L2);
            for (int i : ok) {
                ci.add(java.util.Arrays.copyOfRange(f, i * d, (i + 1) * d), new float[1]);
            }
            for (int i : bad) {
                wi.add(java.util.Arrays.copyOfRange(f, i * d, (i + 1) * d), new float[1]);
            }
            return new LikelihoodRatio(gaussian(f, d, ok, predicted, classes),
                    gaussian(f, d, bad, predicted, classes), ci, wi, k);
        }

        /** Groups by predicted class: the class the selector will be asked about. */
        private static TiedGaussian gaussian(float[] f, int d, List<Integer> rows, int[] predicted, int classes) {
            float[] v = new float[rows.size() * d];
            int[] y = new int[rows.size()];
            for (int i = 0; i < rows.size(); i++) {
                System.arraycopy(f, rows.get(i) * d, v, i * d, d);
                y[i] = predicted[rows.get(i)];
            }
            return TiedGaussian.fit(Tensor.of(Shape.of(rows.size(), d), v), y, classes);
        }

        /**
         * @param features {@code [n, d]}
         * @return Δ-MDS: correct-set minus wrong-set Mahalanobis score
         */
        public double[] deltaMahalanobis(Tensor features) {
            double[] a = correct.score(features);
            double[] b = wrong.score(features);
            for (int i = 0; i < a.length; i++) {
                a[i] -= b[i];
            }
            return a;
        }

        /**
         * @param features {@code [n, d]}
         * @return Δ-KNN: mean log distance to wrong neighbours minus to correct neighbours
         */
        public double[] deltaKnn(Tensor features) {
            int d = features.shape().dim(1);
            float[] f = features.toArray();
            int n = f.length / d;
            double[] out = new double[n];
            for (int i = 0; i < n; i++) {
                float[] q = java.util.Arrays.copyOfRange(f, i * d, (i + 1) * d);
                out[i] = meanLogDistance(wrongIndex, q) - meanLogDistance(correctIndex, q);
            }
            return out;
        }

        private double meanLogDistance(KnnIndex index, float[] q) {
            KnnIndex.Hit[] hits = index.search(q, k);
            double s = 0;
            for (KnnIndex.Hit h : hits) {
                s += 0.5 * Math.log(Math.max(1e-12, -h.score()));
            }
            return s / hits.length;
        }

        /**
         * A score plus a weighted second one, the weight balancing their spreads on a reference
         * set (the paper's rule for combining a distance score with RLog).
         *
         * @param first the first score on the evaluation inputs
         * @param second the second score on the evaluation inputs
         * @param firstReference the first score on reference inputs
         * @param secondReference the second score on reference inputs
         * @return {@code first + lambda second}, {@code lambda = sd(first) / sd(second)}
         */
        public static double[] combine(double[] first, double[] second, double[] firstReference,
                double[] secondReference) {
            double lambda = sd(firstReference) / Math.max(1e-12, sd(secondReference));
            double[] out = new double[first.length];
            for (int i = 0; i < out.length; i++) {
                out[i] = first[i] + lambda * second[i];
            }
            return out;
        }

        private static double sd(double[] v) {
            double m = 0;
            for (double x : v) {
                m += x / v.length;
            }
            double s = 0;
            for (double x : v) {
                s += (x - m) * (x - m) / v.length;
            }
            return Math.sqrt(s);
        }
    }
}
