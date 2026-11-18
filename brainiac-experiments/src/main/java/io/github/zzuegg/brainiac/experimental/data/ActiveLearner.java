package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.Calibration;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Pool-based active learning: starting from a few labels, every round retrains the model on
 * the labelled set and asks an oracle for the labels of a batch of pool points chosen by an
 * {@link Acquisition}:
 *
 * <ul>
 *   <li>{@link Acquisition#RANDOM}; {@link Acquisition#ENTROPY} (predictive entropy) and
 *       {@link Acquisition#BALD} (mutual information under MC dropout, Gal et al. 2017), the
 *       uncertainty scores of {@code Calibration}, taking the top of the batch;</li>
 *   <li>{@link Acquisition#CORE_SET}: greedy {@code k}-centre over penultimate embeddings,
 *       covering the pool (Sener and Savarese, ICLR 2018,
 *       <a href="https://arxiv.org/abs/1708.00489">arXiv:1708.00489</a>);</li>
 *   <li>{@link Acquisition#BADGE}: k-means++ seeding over gradient embeddings
 *       {@code (p - onehot(ŷ)) ⊗ h}, the last layer's gradient for the predicted label, whose
 *       norm is the uncertainty and whose direction gives diversity (Ash et al., ICLR 2020,
 *       <a href="https://arxiv.org/abs/1906.03671">arXiv:1906.03671</a>);</li>
 *   <li>{@link Acquisition#LEVERAGE}: greedy ridge-leverage selection on the embeddings
 *       ({@link DataValuation#leverageSelect}).</li>
 * </ul>
 */
public final class ActiveLearner {

    /** How a batch is chosen. */
    public enum Acquisition {
        /** uniformly at random */
        RANDOM,
        /** highest predictive entropy */
        ENTROPY,
        /** highest BALD mutual information */
        BALD,
        /** greedy k-centre on embeddings */
        CORE_SET,
        /** k-means++ on gradient embeddings */
        BADGE,
        /** greedy leverage on embeddings */
        LEVERAGE
    }

    /** A classifier the loop can retrain and query. */
    public interface Learner {

        /**
         * Trains from scratch on labelled data.
         *
         * @param x the inputs
         * @param y the labels
         */
        void fit(float[][] x, int[] y);

        /**
         * @param x inputs
         * @return class probabilities
         */
        double[][] probabilities(float[][] x);

        /**
         * @param x inputs
         * @param samples stochastic forward passes
         * @return {@code [samples][n][classes]} probabilities with dropout on
         */
        double[][][] stochasticProbabilities(float[][] x, int samples);

        /**
         * @param x inputs
         * @return penultimate-layer embeddings
         */
        double[][] embeddings(float[][] x);
    }

    /**
     * A one-hidden-layer ReLU MLP trained by full-batch Adam, with MC dropout on the hidden
     * layer for BALD.
     */
    public static final class MlpLearner implements Learner {
        private final int inputs;
        private final int hidden;
        private final int classes;
        private final int steps;
        private final SplittableRandom random;
        private SimpleMlp net;
        private int fits;

        /**
         * Creates the learner.
         *
         * @param inputs the input width
         * @param hidden hidden units
         * @param classes classes
         * @param steps Adam steps per fit
         * @param seed the seed
         */
        public MlpLearner(int inputs, int hidden, int classes, int steps, long seed) {
            this.inputs = inputs;
            this.hidden = hidden;
            this.classes = classes;
            this.steps = steps;
            this.random = new SplittableRandom(seed);
        }

        @Override
        public void fit(float[][] x, int[] y) {
            net = SimpleMlp.relu(random.nextLong() + fits++, inputs, hidden, classes);
            Tensor tx = Tensor.of(x);
            float[] ids = new float[y.length];
            for (int i = 0; i < y.length; i++) {
                ids[i] = y[i];
            }
            Tensor ty = Tensor.of(Shape.of(y.length), ids);
            Optimizer adam = Optimizer.adam(0.01f);
            for (int s = 0; s < steps; s++) {
                Var h = net.layer(0, Var.of(tx));
                h = h.mul(Var.of(dropoutMask(y.length)));
                Functional.crossEntropy(net.layer(1, h), ty).backward();
                adam.step(net.parameters());
            }
        }

        private Tensor dropoutMask(int n) {
            float[] m = new float[n * hidden];
            for (int i = 0; i < m.length; i++) {
                m[i] = random.nextDouble() < 0.25 ? 0 : 1 / 0.75f;
            }
            return Tensor.of(Shape.of(n, hidden), m);
        }

        @Override
        public double[][] probabilities(float[][] x) {
            return softmax(net.predict(Tensor.of(x)));
        }

        @Override
        public double[][][] stochasticProbabilities(float[][] x, int samples) {
            double[][][] out = new double[samples][][];
            Tensor tx = Tensor.of(x);
            for (int s = 0; s < samples; s++) {
                Var h = net.layer(0, Var.of(tx)).mul(Var.of(dropoutMask(x.length)));
                out[s] = softmax(net.layer(1, h).value());
            }
            return out;
        }

        @Override
        public double[][] embeddings(float[][] x) {
            Tensor h = net.layer(0, Var.of(Tensor.of(x))).value();
            return rows(h);
        }

        private static double[][] softmax(Tensor logits) {
            double[][] l = rows(logits);
            for (double[] row : l) {
                double max = Arrays.stream(row).max().orElse(0);
                double z = 0;
                for (int c = 0; c < row.length; c++) {
                    row[c] = Math.exp(row[c] - max);
                    z += row[c];
                }
                for (int c = 0; c < row.length; c++) {
                    row[c] /= z;
                }
            }
            return l;
        }

        private static double[][] rows(Tensor t) {
            int n = t.shape().dim(0);
            int d = t.shape().dim(1);
            float[] v = t.toArray();
            double[][] out = new double[n][d];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) {
                    out[i][j] = v[i * d + j];
                }
            }
            return out;
        }
    }

    private final Supplier<Learner> learners;
    private final Acquisition acquisition;
    private final SplittableRandom random;

    private ActiveLearner(Supplier<Learner> learners, Acquisition acquisition, long seed) {
        this.learners = Objects.requireNonNull(learners);
        this.acquisition = Objects.requireNonNull(acquisition);
        this.random = new SplittableRandom(seed);
    }

    /**
     * Creates the loop.
     *
     * @param learners makes a fresh learner each round
     * @param acquisition the acquisition function
     * @param seed the seed
     * @return the loop
     */
    public static ActiveLearner of(Supplier<Learner> learners, Acquisition acquisition, long seed) {
        return new ActiveLearner(learners, acquisition, seed);
    }

    /**
     * Runs the loop.
     *
     * @param poolX the pool's inputs
     * @param poolY the oracle's labels
     * @param testX test inputs
     * @param testY test labels
     * @param initial labels to start with (drawn at random)
     * @param batch labels per round
     * @param budget the total number of labels
     * @return test accuracy after each fit, from the initial set to the full budget
     */
    public double[] run(float[][] poolX, int[] poolY, float[][] testX, int[] testY, int initial, int batch,
            int budget) {
        int n = poolX.length;
        boolean[] labelled = new boolean[n];
        List<Integer> chosen = new ArrayList<>();
        while (chosen.size() < initial) {
            int i = random.nextInt(n);
            if (!labelled[i]) {
                labelled[i] = true;
                chosen.add(i);
            }
        }
        List<Double> curve = new ArrayList<>();
        while (true) {
            Learner learner = learners.get();
            float[][] lx = new float[chosen.size()][];
            int[] ly = new int[chosen.size()];
            for (int k = 0; k < ly.length; k++) {
                lx[k] = poolX[chosen.get(k)];
                ly[k] = poolY[chosen.get(k)];
            }
            learner.fit(lx, ly);
            curve.add(accuracy(learner.probabilities(testX), testY));
            if (chosen.size() >= budget) {
                break;
            }
            int take = Math.min(batch, budget - chosen.size());
            for (int i : select(learner, poolX, labelled, chosen, take)) {
                labelled[i] = true;
                chosen.add(i);
            }
        }
        return curve.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * Chooses a batch.
     *
     * @param learner the current model
     * @param poolX the pool
     * @param labelled which pool points are labelled
     * @param chosen the labelled indices
     * @param k the batch size
     * @return the chosen unlabelled indices
     */
    public int[] select(Learner learner, float[][] poolX, boolean[] labelled, List<Integer> chosen, int k) {
        int n = poolX.length;
        int[] candidates = new int[n - chosen.size()];
        for (int i = 0, c = 0; i < n; i++) {
            if (!labelled[i]) {
                candidates[c++] = i;
            }
        }
        return switch (acquisition) {
            case RANDOM -> {
                int[] shuffled = candidates.clone();
                for (int i = shuffled.length - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    int t = shuffled[i];
                    shuffled[i] = shuffled[j];
                    shuffled[j] = t;
                }
                yield Arrays.copyOf(shuffled, k);
            }
            case ENTROPY -> {
                double[][] p = learner.probabilities(poolX);
                double[] score = new double[n];
                for (int i : candidates) {
                    score[i] = Calibration.entropy(p[i]);
                }
                yield top(score, candidates, k);
            }
            case BALD -> {
                double[][][] mc = learner.stochasticProbabilities(poolX, 10);
                double[] score = new double[n];
                for (int i : candidates) {
                    double[][] members = new double[mc.length][];
                    for (int s = 0; s < mc.length; s++) {
                        members[s] = mc[s][i];
                    }
                    score[i] = Calibration.mutualInformation(members);
                }
                yield top(score, candidates, k);
            }
            case CORE_SET -> kCentre(learner.embeddings(poolX), chosen, candidates, k);
            case BADGE -> kMeansPlusPlus(gradientEmbeddings(learner, poolX), candidates, k);
            case LEVERAGE -> {
                double[][] e = learner.embeddings(poolX);
                double trace = 0;
                for (int i : candidates) {
                    for (double v : e[i]) {
                        trace += v * v;
                    }
                }
                double lambda = Math.max(1e-6, 0.01 * trace / e[0].length);
                yield DataValuation.leverageSelect(e, chosen.stream().mapToInt(Integer::intValue).toArray(),
                        candidates, k, lambda);
            }
        };
    }

    private static int[] top(double[] score, int[] candidates, int k) {
        Integer[] idx = Arrays.stream(candidates).boxed().toArray(Integer[]::new);
        Arrays.sort(idx, (a, b) -> Double.compare(score[b], score[a]));
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            out[i] = idx[i];
        }
        return out;
    }

    /** Greedy k-centre: repeatedly the candidate farthest from every labelled or picked point. */
    private static int[] kCentre(double[][] e, List<Integer> chosen, int[] candidates, int k) {
        double[] nearest = new double[e.length];
        Arrays.fill(nearest, Double.POSITIVE_INFINITY);
        for (int c : chosen) {
            for (int i : candidates) {
                nearest[i] = Math.min(nearest[i], squared(e[i], e[c]));
            }
        }
        int[] out = new int[k];
        for (int p = 0; p < k; p++) {
            int best = candidates[0];
            for (int i : candidates) {
                best = nearest[i] > nearest[best] ? i : best;
            }
            out[p] = best;
            for (int i : candidates) {
                nearest[i] = Math.min(nearest[i], squared(e[i], e[best]));
            }
        }
        return out;
    }

    /** BADGE's gradient embedding {@code (p - onehot(argmax p)) ⊗ h}. */
    static double[][] gradientEmbeddings(Learner learner, float[][] x) {
        double[][] p = learner.probabilities(x);
        double[][] h = learner.embeddings(x);
        double[][] g = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            int yhat = Calibration.argmax(p[i]);
            int classes = p[i].length;
            int width = h[i].length;
            g[i] = new double[classes * width];
            for (int c = 0; c < classes; c++) {
                double d = p[i][c] - (c == yhat ? 1 : 0);
                for (int j = 0; j < width; j++) {
                    g[i][c * width + j] = d * h[i][j];
                }
            }
        }
        return g;
    }

    /** k-means++ seeding over the candidates: the first by largest norm, then D² sampling. */
    private int[] kMeansPlusPlus(double[][] g, int[] candidates, int k) {
        int[] out = new int[k];
        double[] d2 = new double[g.length];
        int first = candidates[0];
        for (int i : candidates) {
            first = norm2(g[i]) > norm2(g[first]) ? i : first;
        }
        out[0] = first;
        for (int i : candidates) {
            d2[i] = squared(g[i], g[first]);
        }
        boolean[] taken = new boolean[g.length];
        taken[first] = true;
        for (int p = 1; p < k; p++) {
            double total = 0;
            for (int i : candidates) {
                total += taken[i] ? 0 : d2[i];
            }
            int pick = -1;
            if (total <= 0) {
                for (int i : candidates) {
                    if (!taken[i]) {
                        pick = i;
                        break;
                    }
                }
            } else {
                double u = random.nextDouble() * total;
                for (int i : candidates) {
                    if (taken[i]) {
                        continue;
                    }
                    u -= d2[i];
                    pick = i;
                    if (u <= 0) {
                        break;
                    }
                }
            }
            out[p] = pick;
            taken[pick] = true;
            for (int i : candidates) {
                d2[i] = Math.min(d2[i], squared(g[i], g[pick]));
            }
        }
        return out;
    }

    private static double norm2(double[] a) {
        double s = 0;
        for (double v : a) {
            s += v * v;
        }
        return s;
    }

    private static double squared(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double e = a[i] - b[i];
            s += e * e;
        }
        return s;
    }

    /**
     * @param p probabilities
     * @param y labels
     * @return accuracy
     */
    static double accuracy(double[][] p, int[] y) {
        int right = 0;
        for (int i = 0; i < y.length; i++) {
            right += Calibration.argmax(p[i]) == y[i] ? 1 : 0;
        }
        return (double) right / y.length;
    }

    /**
     * The normalised area under a learning curve (mean accuracy over rounds).
     *
     * @param curve accuracies
     * @return their mean
     */
    public static double area(double[] curve) {
        return Arrays.stream(curve).average().orElse(0);
    }
}
