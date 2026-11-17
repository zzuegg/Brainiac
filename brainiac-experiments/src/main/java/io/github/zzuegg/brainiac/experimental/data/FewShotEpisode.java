package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * One few-shot episode: a small labelled support set to learn from and a query set to be
 * scored on, with labels relabelled to {@code 0..ways-1} for the episode. Meta-learners
 * (prototypical and matching networks, MAML and its variants, SNAIL, LEO, CNAPs) all consume
 * this one shape of data.
 *
 * @param support support inputs {@code [support rows, features]}
 * @param supportLabels the episode class of each support row
 * @param query query inputs {@code [queries, features]}
 * @param queryLabels the episode class of each query row
 * @param ways the number of classes in the episode
 */
public record FewShotEpisode(Tensor support, int[] supportLabels, Tensor query, int[] queryLabels, int ways) {

    /**
     * Checks the shapes.
     *
     * @param support support inputs
     * @param supportLabels support classes
     * @param query query inputs
     * @param queryLabels query classes
     * @param ways the classes in the episode
     */
    public FewShotEpisode {
        if (support.shape().dim(0) != supportLabels.length || query.shape().dim(0) != queryLabels.length) {
            throw new IllegalArgumentException("labels do not match the rows");
        }
    }

    /** @return the input features per row */
    public int features() {
        return support.shape().dim(1);
    }

    /**
     * The one-hot support labels.
     *
     * @return {@code [support rows, ways]}
     */
    public Tensor supportOneHot() {
        return oneHot(supportLabels, ways);
    }

    /**
     * The query labels as float class ids, the form {@code Functional.crossEntropy} takes.
     *
     * @return {@code [queries]}
     */
    public Tensor queryIds() {
        return ids(queryLabels);
    }

    /**
     * The support labels as float class ids.
     *
     * @return {@code [support rows]}
     */
    public Tensor supportIds() {
        return ids(supportLabels);
    }

    /**
     * The share of predictions equal to the query labels.
     *
     * @param predicted one class per query
     * @return the accuracy
     */
    public double accuracy(int[] predicted) {
        int ok = 0;
        for (int i = 0; i < predicted.length; i++) {
            ok += predicted[i] == queryLabels[i] ? 1 : 0;
        }
        return ok / (double) predicted.length;
    }

    /**
     * One-hot rows.
     *
     * @param labels the classes
     * @param classes the number of classes
     * @return {@code [labels.length, classes]}
     */
    public static Tensor oneHot(int[] labels, int classes) {
        float[] out = new float[labels.length * classes];
        for (int i = 0; i < labels.length; i++) {
            out[i * classes + labels[i]] = 1;
        }
        return Tensor.of(Shape.of(labels.length, classes), out);
    }

    /**
     * The row-wise argmax of a score matrix.
     *
     * @param scores {@code [rows, classes]}
     * @return the best class of each row
     */
    public static int[] argmax(Tensor scores) {
        float[] s = scores.toArray();
        int k = scores.shape().dim(-1);
        int[] out = new int[s.length / k];
        for (int i = 0; i < out.length; i++) {
            int best = 0;
            for (int c = 1; c < k; c++) {
                if (s[i * k + c] > s[i * k + best]) {
                    best = c;
                }
            }
            out[i] = best;
        }
        return out;
    }

    private static Tensor ids(int[] labels) {
        float[] out = new float[labels.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = labels[i];
        }
        return Tensor.of(Shape.of(out.length), out);
    }
}
