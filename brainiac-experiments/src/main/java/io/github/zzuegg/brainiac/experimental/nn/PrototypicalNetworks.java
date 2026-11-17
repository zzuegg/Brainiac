package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Prototypical networks for few-shot classification: embed the few labelled support
 * examples, average each class's embeddings into a prototype, and classify a query by a
 * softmax over negative squared distances to the prototypes. Training on many small
 * episodes teaches an embedding in which this nearest-mean rule works for classes never seen
 * in training.
 *
 * <p>Paper: Snell, Swersky, Zemel, "Prototypical Networks for Few-shot Learning" (2017),
 * <a href="https://arxiv.org/abs/1703.05175">arXiv:1703.05175</a>.
 */
public final class PrototypicalNetworks {

    private PrototypicalNetworks() {
    }

    /**
     * The class prototypes.
     *
     * @param support support embeddings {@code [n, dim]}
     * @param labels the class of each support row, in {@code [0, classes)}
     * @param classes the number of classes in the episode
     * @return {@code [classes, dim]}, the mean embedding of each class
     */
    public static Var prototypes(Var support, int[] labels, int classes) {
        int n = support.shape().dim(0);
        float[] assign = new float[classes * n];
        int[] counts = new int[classes];
        for (int label : labels) {
            counts[label]++;
        }
        for (int i = 0; i < n; i++) {
            assign[labels[i] * n + i] = 1f / counts[labels[i]];
        }
        return Var.of(Tensor.of(Shape.of(classes, n), assign, support.value().device())).matmul(support);
    }

    /**
     * The logits {@code -||q - c_k||^2} of each query for each prototype.
     *
     * @param query query embeddings {@code [m, dim]}
     * @param prototypes {@code [classes, dim]}
     * @return {@code [m, classes]}
     */
    public static Var logits(Var query, Var prototypes) {
        Var qq = query.square().sum(-1, true);
        Var pp = prototypes.square().sum(-1, true).transpose();
        return query.matmul(prototypes.transpose()).mul(2f).sub(qq).sub(pp);
    }

    /**
     * The episode loss: cross-entropy of the queries' distance softmax.
     *
     * @param support support embeddings {@code [n, dim]}
     * @param supportLabels support classes
     * @param query query embeddings {@code [m, dim]}
     * @param queryLabels query classes
     * @param classes the classes in the episode
     * @return the loss
     */
    public static Var loss(Var support, int[] supportLabels, Var query, int[] queryLabels, int classes) {
        float[] ids = new float[queryLabels.length];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = queryLabels[i];
        }
        return Functional.crossEntropy(logits(query, prototypes(support, supportLabels, classes)),
                Tensor.of(Shape.of(ids.length), ids, query.value().device()));
    }

    /**
     * Classifies queries by their nearest prototype.
     *
     * @param query query embeddings {@code [m, dim]}
     * @param prototypes {@code [classes, dim]}
     * @return the predicted class of each query
     */
    public static int[] classify(Var query, Var prototypes) {
        float[] l = logits(query, prototypes).value().toArray();
        int k = prototypes.shape().dim(0);
        int[] out = new int[query.shape().dim(0)];
        for (int i = 0; i < out.length; i++) {
            int best = 0;
            for (int c = 1; c < k; c++) {
                if (l[i * k + c] > l[i * k + best]) {
                    best = c;
                }
            }
            out[i] = best;
        }
        return out;
    }
}
