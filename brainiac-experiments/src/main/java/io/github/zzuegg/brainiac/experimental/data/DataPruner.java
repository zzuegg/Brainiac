package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Data pruning by example difficulty, scored early in training:
 *
 * <ul>
 *   <li><b>EL2N</b> (Paul et al. 2021, "Deep Learning on a Data Diet",
 *       <a href="https://arxiv.org/abs/2107.07075">arXiv:2107.07075</a>): the L2 norm of
 *       {@code softmax(logits) - onehot(label)}, averaged over a few early checkpoints (or
 *       seeds). Easy, redundant examples score near 0; hard or mislabelled ones near
 *       {@code sqrt(2)}.</li>
 *   <li><b>Forgetting events</b> (Toneva et al. 2019,
 *       <a href="https://arxiv.org/abs/1812.05159">arXiv:1812.05159</a>): how often an
 *       example goes from correctly to incorrectly classified between recordings; examples
 *       never learned count as most forgotten.</li>
 * </ul>
 * Then {@link #keep} drops the easiest share (redundant) and optionally the very hardest
 * share (likely label noise) and returns the indices to train on. Sorscher et al. 2022
 * (<a href="https://arxiv.org/abs/2206.14486">arXiv:2206.14486</a>) show that with enough
 * data this beats random pruning and can beat power-law scaling.
 *
 * <p>Call {@link #record} with the logits for every example (in a fixed order) at a few
 * points early in training.
 */
public final class DataPruner {

    private final int examples;
    private final double[] el2nSum;
    private final int[] forgetting;
    private final boolean[] wasCorrect;
    private final boolean[] everLearned;
    private int recordings;

    /**
     * @param examples the dataset size
     */
    public DataPruner(int examples) {
        if (examples < 1) {
            throw new IllegalArgumentException("examples must be positive");
        }
        this.examples = examples;
        this.el2nSum = new double[examples];
        this.forgetting = new int[examples];
        this.wasCorrect = new boolean[examples];
        this.everLearned = new boolean[examples];
    }

    /**
     * Records the model's predictions for every example.
     *
     * @param logits {@code [examples, classes]}
     * @param labels {@code [examples]} class ids
     */
    public void record(Tensor logits, Tensor labels) {
        int classes = logits.shape().dim(1);
        if (logits.shape().dim(0) != examples) {
            throw new IllegalArgumentException("expected logits for " + examples + " examples");
        }
        float[] z = logits.toArray();
        float[] y = labels.toArray();
        for (int i = 0; i < examples; i++) {
            int label = Math.round(y[i]);
            double max = Double.NEGATIVE_INFINITY;
            int best = 0;
            for (int c = 0; c < classes; c++) {
                if (z[i * classes + c] > max) {
                    max = z[i * classes + c];
                    best = c;
                }
            }
            double sum = 0;
            for (int c = 0; c < classes; c++) {
                sum += Math.exp(z[i * classes + c] - max);
            }
            double err = 0;
            for (int c = 0; c < classes; c++) {
                double p = Math.exp(z[i * classes + c] - max) / sum - (c == label ? 1 : 0);
                err += p * p;
            }
            el2nSum[i] += Math.sqrt(err);
            boolean correct = best == label;
            if (recordings > 0 && wasCorrect[i] && !correct) {
                forgetting[i]++;
            }
            wasCorrect[i] = correct;
            everLearned[i] |= correct;
        }
        recordings++;
    }

    /** @return the recordings so far */
    public int recordings() {
        return recordings;
    }

    /** @return the mean EL2N score of every example (higher is harder) */
    public double[] el2n() {
        double[] out = new double[examples];
        for (int i = 0; i < examples; i++) {
            out[i] = recordings == 0 ? 0 : el2nSum[i] / recordings;
        }
        return out;
    }

    /**
     * @return the forgetting events of every example; never-learned examples get one more
     *     than the most forgotten learned example
     */
    public double[] forgettingScores() {
        int max = 0;
        for (int f : forgetting) {
            max = Math.max(max, f);
        }
        double[] out = new double[examples];
        for (int i = 0; i < examples; i++) {
            out[i] = everLearned[i] ? forgetting[i] : max + 1;
        }
        return out;
    }

    /**
     * The examples to keep: sorted by score, the hardest {@code dropHardest} share removed,
     * then the easiest removed until {@code keepFraction} of the dataset is left.
     *
     * @param scores a difficulty per example, higher is harder
     * @param keepFraction the share of the dataset kept, in {@code (0, 1]}
     * @param dropHardest the share of the dataset dropped as likely noise, at the top
     * @return the kept indices, ascending
     */
    public static int[] keep(double[] scores, double keepFraction, double dropHardest) {
        int n = scores.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));
        int hardestKept = n - (int) Math.round(dropHardest * n);
        int count = Math.min(hardestKept, Math.max(1, (int) Math.round(keepFraction * n)));
        int[] kept = new int[count];
        for (int i = 0; i < count; i++) {
            kept[i] = order[hardestKept - count + i];
        }
        Arrays.sort(kept);
        return kept;
    }

    /**
     * A random subset, the baseline pruning.
     *
     * @param n the dataset size
     * @param keepFraction the share kept
     * @param seed the seed
     * @return the kept indices, ascending
     */
    public static int[] random(int n, double keepFraction, long seed) {
        int[] all = new int[n];
        for (int i = 0; i < n; i++) {
            all[i] = i;
        }
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = all[i];
            all[i] = all[j];
            all[j] = t;
        }
        int[] kept = Arrays.copyOf(all, Math.max(1, (int) Math.round(keepFraction * n)));
        Arrays.sort(kept);
        return kept;
    }

    /**
     * Picks rows of a tensor.
     *
     * @param data {@code [n, ...]}
     * @param rows the row indices
     * @return {@code [rows.length, ...]}
     */
    public static Tensor select(Tensor data, int[] rows) {
        int n = data.shape().dim(0);
        int width = Math.toIntExact(data.size() / n);
        float[] all = data.toArray();
        float[] out = new float[rows.length * width];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(all, rows[i] * width, out, i * width, width);
        }
        int[] dims = data.shape().toArray();
        dims[0] = rows.length;
        return Tensor.of(Shape.of(dims), out);
    }
}
