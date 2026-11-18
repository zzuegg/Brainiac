package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GluFeedForward;
import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Structured pruning of transformer language models: removing whole coupled structures (an
 * attention head with its rows of the output projection, a feed-forward neuron with its row of
 * the down projection, a whole attention module) so the result is a smaller dense model that
 * needs no sparse kernels. Three published recipes, on the same {@link Group} abstraction:
 * <ul>
 * <li>{@link #taylorScores}: LLM-Pruner (Ma et al. 2023, <a href="https://arxiv.org/abs/2305.11627">arXiv:2305.11627</a>),
 *     dependency groups scored by the first-order Taylor estimate of the loss change
 *     {@code |sum_group w * dL/dw|} on calibration data, then removed and recovered by
 *     fine-tuning;</li>
 * <li>{@link #faspScores} with {@link #restoreConsumer}: FASP (2025, <a href="https://arxiv.org/abs/2501.09412">arXiv:2501.09412</a>),
 *     which links a layer's removed columns to the next layer's rows, scores each structure by the
 *     consumer row's norm times its input activation's norm (a Wanda-style metric), removes it,
 *     and restores the consumer by the closed-form least-squares update
 *     {@code W' = argmin ||X_kept W' - X W||};</li>
 * <li>{@link #activationScores} plus whole-module removal: 2SSP (Sandri et al. 2025,
 *     <a href="https://arxiv.org/abs/2501.17771">arXiv:2501.17771</a>), which first prunes
 *     feed-forward neurons by their output-activation magnitude (width) and then whole attention
 *     modules (depth), balancing the two by the target sparsity.</li>
 * </ul>
 * Removing a group zeroes its slices; {@link #params} counts what a dense re-packing saves.
 */
public final class StructuredLmPruner {

    private StructuredLmPruner() {
    }

    /**
     * A contiguous range of rows ({@code axis 0}) or columns ({@code axis 1}) of a matrix.
     *
     * @param weight the matrix {@code [in, out]}
     * @param axis 0 for rows, 1 for columns
     * @param start the first index
     * @param end the last index, exclusive
     */
    public record Slice(Var weight, int axis, int start, int end) {

        /** @return the number of weights in the slice */
        public int size() {
            return (end - start) * weight.shape().dim(1 - axis);
        }
    }

    /**
     * A coupled structure that is removed as a whole.
     *
     * @param name a label, such as {@code "b0.head2"}
     * @param kind the structure type, such as {@code "head"}, {@code "neuron"} or {@code "attention"}
     * @param slices every slice of the structure
     * @param consumer the slice of rows of the matrix consuming the structure's output (one of
     *     {@code slices}), used by FASP's scoring and least-squares restore
     */
    public record Group(String name, String kind, List<Slice> slices, Slice consumer) {

        /** @return the number of weights the group holds */
        public int params() {
            int n = 0;
            for (Slice s : slices) {
                n += s.size();
            }
            return n;
        }
    }

    /**
     * One group per attention head: its query, key and value columns and its output rows
     * (multi-head attention without grouped keys and values).
     *
     * @param attention the attention layer
     * @param prefix a label prefix
     * @return the groups
     */
    public static List<Group> heads(MultiHeadAttention attention, String prefix) {
        MultiHeadAttention.Settings s = attention.settings();
        if (s.kvHeads() != s.heads()) {
            throw new IllegalArgumentException("grouped-query attention couples heads; not supported");
        }
        List<Var> w = attention.parameters();
        List<Group> g = new ArrayList<>();
        for (int h = 0; h < s.heads(); h++) {
            Slice out = new Slice(w.get(3), 0, h * s.valueSize(), (h + 1) * s.valueSize());
            g.add(new Group(prefix + "head" + h, "head", List.of(
                    new Slice(w.get(0), 1, h * s.headSize(), (h + 1) * s.headSize()),
                    new Slice(w.get(1), 1, h * s.headSize(), (h + 1) * s.headSize()),
                    new Slice(w.get(2), 1, h * s.valueSize(), (h + 1) * s.valueSize()), out), out));
        }
        return g;
    }

    /**
     * One group per hidden neuron of a feed-forward layer: its up (and gate) column and its down row.
     *
     * @param ffn the feed-forward layer
     * @param prefix a label prefix
     * @return the groups
     */
    public static List<Group> neurons(GluFeedForward ffn, String prefix) {
        List<Var> p = ffn.parameters();   // up, down, gate
        int hidden = p.get(0).shape().dim(1);
        List<Group> g = new ArrayList<>();
        for (int j = 0; j < hidden; j++) {
            Slice down = new Slice(p.get(1), 0, j, j + 1);
            List<Slice> slices = new ArrayList<>(List.of(new Slice(p.get(0), 1, j, j + 1), down));
            if (p.size() > 2) {
                slices.add(new Slice(p.get(2), 1, j, j + 1));
            }
            g.add(new Group(prefix + "n" + j, "neuron", slices, down));
        }
        return g;
    }

    /**
     * A whole attention module as one group (2SSP's depth stage).
     *
     * @param attention the attention layer
     * @param prefix a label prefix
     * @return the group
     */
    public static Group module(MultiHeadAttention attention, String prefix) {
        List<Slice> slices = new ArrayList<>();
        for (Var w : attention.parameters()) {
            if (w.shape().rank() == 2) {
                slices.add(new Slice(w, 0, 0, w.shape().dim(0)));
            }
        }
        return new Group(prefix + "attention", "attention", slices, slices.getLast());
    }

    /**
     * LLM-Pruner's group importance {@code |sum w * g|}.
     *
     * @param groups the groups
     * @param grads the loss gradient of every weight in the groups
     * @return one score per group
     */
    public static double[] taylorScores(List<Group> groups, Map<Var, Tensor> grads) {
        double[] s = new double[groups.size()];
        for (int i = 0; i < s.length; i++) {
            double sum = 0;
            for (Slice sl : groups.get(i).slices()) {
                float[] w = sl.weight().value().toArray();
                float[] g = grads.get(sl.weight()).toArray();
                sum += dotSlice(sl, w, g);
            }
            s[i] = Math.abs(sum);
        }
        return s;
    }

    /**
     * FASP's score: the norm of the consumer rows times the norm of their input activations.
     *
     * @param groups the groups
     * @param consumerInputs the input {@code X [rows, in]} of every consumer matrix
     * @return one score per group
     */
    public static double[] faspScores(List<Group> groups, Map<Var, Tensor> consumerInputs) {
        double[] s = new double[groups.size()];
        for (int i = 0; i < s.length; i++) {
            Slice c = groups.get(i).consumer();
            float[] w = c.weight().value().toArray();
            int out = c.weight().shape().dim(1);
            double wn = 0;
            for (int r = c.start(); r < c.end(); r++) {
                for (int j = 0; j < out; j++) {
                    wn += w[r * out + j] * w[r * out + j];
                }
            }
            s[i] = Math.sqrt(wn) * columnNorm(consumerInputs.get(c.weight()), c.start(), c.end(), false);
        }
        return s;
    }

    /**
     * 2SSP's width score: the mean absolute activation of the structure's output.
     *
     * @param groups the groups
     * @param consumerInputs the input of every consumer matrix
     * @return one score per group
     */
    public static double[] activationScores(List<Group> groups, Map<Var, Tensor> consumerInputs) {
        double[] s = new double[groups.size()];
        for (int i = 0; i < s.length; i++) {
            Slice c = groups.get(i).consumer();
            s[i] = columnNorm(consumerInputs.get(c.weight()), c.start(), c.end(), true);
        }
        return s;
    }

    /**
     * Chooses the lowest-scoring share of the groups.
     *
     * @param scores one per group
     * @param share the share of groups to remove
     * @return which to remove
     */
    public static boolean[] lowestShare(double[] scores, double share) {
        return WoodFisherPruning.lowest(scores, (int) Math.round(share * scores.length));
    }

    /**
     * Zeroes the chosen groups.
     *
     * @param groups the groups
     * @param remove which to remove
     */
    public static void remove(List<Group> groups, boolean[] remove) {
        for (int i = 0; i < groups.size(); i++) {
            if (remove[i]) {
                for (Slice s : groups.get(i).slices()) {
                    zero(s);
                }
            }
        }
    }

    /**
     * @param groups the groups
     * @param remove which were removed
     * @return the weights removed
     */
    public static long params(List<Group> groups, boolean[] remove) {
        long n = 0;
        for (int i = 0; i < groups.size(); i++) {
            n += remove[i] ? groups.get(i).params() : 0;
        }
        return n;
    }

    /**
     * FASP's closed-form restore: refits a consumer matrix on its kept input rows so the layer
     * output on calibration data stays as close as possible to the unpruned output.
     *
     * @param consumer the consumer weight {@code [in, out]}, already zeroed in removed rows
     * @param original the consumer weight before pruning
     * @param input the calibration input {@code X [rows, in]} before pruning
     * @param kept which input rows survive
     * @param ridge a ridge as a share of the mean diagonal of {@code X^T X}, for example 1e-3
     */
    public static void restoreConsumer(Var consumer, Tensor original, Tensor input, boolean[] kept, double ridge) {
        int in = consumer.shape().dim(0);
        int out = consumer.shape().dim(1);
        double[] x = input.toDoubleArray();
        int rows = x.length / in;
        double[] w = original.toDoubleArray();
        int[] k = new int[in];
        int m = 0;
        for (int i = 0; i < in; i++) {
            if (kept[i]) {
                k[m++] = i;
            }
        }
        double[][] gram = new double[m][m];
        double[][] rhs = new double[m][out];
        double[] y = new double[out];
        for (int r = 0; r < rows; r++) {
            Arrays.fill(y, 0);
            for (int i = 0; i < in; i++) {
                double xv = x[r * in + i];
                if (xv != 0) {
                    for (int j = 0; j < out; j++) {
                        y[j] += xv * w[i * out + j];
                    }
                }
            }
            for (int a = 0; a < m; a++) {
                double xa = x[r * in + k[a]];
                for (int b = 0; b < m; b++) {
                    gram[a][b] += xa * x[r * in + k[b]];
                }
                for (int j = 0; j < out; j++) {
                    rhs[a][j] += xa * y[j];
                }
            }
        }
        double mean = 0;
        for (int a = 0; a < m; a++) {
            mean += gram[a][a] / Math.max(1, m);
        }
        for (int a = 0; a < m; a++) {
            gram[a][a] += ridge * mean + 1e-12;
        }
        double[] nw = new double[in * out];
        for (int j = 0; j < out; j++) {
            double[] b = new double[m];
            for (int a = 0; a < m; a++) {
                b[a] = rhs[a][j];
            }
            double[] sol = CurvatureProducts.solveSpd(gram, b);
            for (int a = 0; a < m; a++) {
                nw[k[a] * out + j] = sol[a];
            }
        }
        consumer.assign(Tensor.of(consumer.shape(), WoodFisherPruning.toFloat(nw)));
    }

    /**
     * @param groups groups that all share one consumer matrix
     * @param remove which are removed
     * @param inputs the consumer's input width
     * @return which consumer rows survive
     */
    public static boolean[] keptRows(List<Group> groups, boolean[] remove, int inputs) {
        boolean[] kept = new boolean[inputs];
        Arrays.fill(kept, true);
        for (int i = 0; i < groups.size(); i++) {
            if (remove[i]) {
                Slice c = groups.get(i).consumer();
                for (int r = c.start(); r < c.end(); r++) {
                    kept[r] = false;
                }
            }
        }
        return kept;
    }

    private static double dotSlice(Slice s, float[] w, float[] g) {
        int cols = s.weight().shape().dim(1);
        int rows = s.weight().shape().dim(0);
        double sum = 0;
        if (s.axis() == 0) {
            for (int r = s.start(); r < s.end(); r++) {
                for (int j = 0; j < cols; j++) {
                    sum += w[r * cols + j] * (double) g[r * cols + j];
                }
            }
        } else {
            for (int r = 0; r < rows; r++) {
                for (int j = s.start(); j < s.end(); j++) {
                    sum += w[r * cols + j] * (double) g[r * cols + j];
                }
            }
        }
        return sum;
    }

    private static void zero(Slice s) {
        float[] w = s.weight().value().toArray();
        int cols = s.weight().shape().dim(1);
        int rows = s.weight().shape().dim(0);
        if (s.axis() == 0) {
            Arrays.fill(w, s.start() * cols, s.end() * cols, 0f);
        } else {
            for (int r = 0; r < rows; r++) {
                Arrays.fill(w, r * cols + s.start(), r * cols + s.end(), 0f);
            }
        }
        s.weight().assign(Tensor.of(s.weight().shape(), w));
    }

    private static double columnNorm(Tensor x, int from, int to, boolean meanAbs) {
        int in = x.shape().dim(x.shape().rank() - 1);
        float[] v = x.toArray();
        int rows = v.length / in;
        double s = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = from; c < to; c++) {
                double a = v[r * in + c];
                s += meanAbs ? Math.abs(a) : a * a;
            }
        }
        return meanAbs ? s / (rows * (double) (to - from)) : Math.sqrt(s);
    }
}
