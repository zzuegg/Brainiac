package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Matching networks for few-shot classification: a query is labelled by attention over the
 * embedded support set, {@code p(y | q) = sum_i a(q, s_i) onehot(y_i)}, where {@code a} is a
 * softmax over the cosine similarities of the query to every support embedding (scaled by a
 * temperature). Unlike {@link PrototypicalNetworks} it keeps every support example instead of
 * one mean per class, so it behaves like a differentiable, trained nearest-neighbour rule.
 * Episodic training of the embedding with this loss makes the rule work on unseen classes.
 *
 * <p>The paper's optional full-context embeddings (an LSTM over the support set) are left out;
 * the "simple" embedding variant is what this implements.
 *
 * <p>Paper: Vinyals, Blundell, Lillicrap, Kavukcuoglu, Wierstra, "Matching Networks for One
 * Shot Learning" (2016), <a href="https://arxiv.org/abs/1606.04080">arXiv:1606.04080</a>.
 */
public final class MatchingNetwork {

    private MatchingNetwork() {
    }

    /**
     * The attention weights of every query over the support set.
     *
     * @param query query embeddings {@code [m, dim]}
     * @param support support embeddings {@code [n, dim]}
     * @param temperature multiplies the cosine similarities before the softmax, such as 10
     * @return {@code [m, n]}, rows summing to 1
     */
    public static Var attention(Var query, Var support, float temperature) {
        Var cos = normalize(query).matmul(normalize(support).transpose()).mul(temperature);
        return Activation.SOFTMAX.forward(cos);
    }

    /**
     * The class probabilities of every query.
     *
     * @param query query embeddings {@code [m, dim]}
     * @param support support embeddings {@code [n, dim]}
     * @param labels the class of each support row
     * @param classes the classes in the episode
     * @param temperature the cosine scale
     * @return {@code [m, classes]}
     */
    public static Var probabilities(Var query, Var support, int[] labels, int classes, float temperature) {
        float[] oneHot = new float[labels.length * classes];
        for (int i = 0; i < labels.length; i++) {
            oneHot[i * classes + labels[i]] = 1;
        }
        Tensor y = Tensor.of(Shape.of(labels.length, classes), oneHot, query.value().device());
        return attention(query, support, temperature).matmul(Var.of(y));
    }

    /**
     * The episode loss: the negative log-likelihood of the query labels.
     *
     * @param support support embeddings {@code [n, dim]}
     * @param supportLabels support classes
     * @param query query embeddings {@code [m, dim]}
     * @param queryLabels query classes
     * @param classes the classes in the episode
     * @param temperature the cosine scale
     * @return the loss
     */
    public static Var loss(Var support, int[] supportLabels, Var query, int[] queryLabels, int classes,
            float temperature) {
        float[] ids = new float[queryLabels.length];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = queryLabels[i];
        }
        // log p is a valid logit: its softmax is p again because the rows sum to 1
        Var logp = probabilities(query, support, supportLabels, classes, temperature).add(1e-6f).log();
        return Functional.crossEntropy(logp, Tensor.of(Shape.of(ids.length), ids, query.value().device()));
    }

    /**
     * Classifies queries by their most probable class.
     *
     * @param query query embeddings {@code [m, dim]}
     * @param support support embeddings {@code [n, dim]}
     * @param labels support classes
     * @param classes the classes in the episode
     * @param temperature the cosine scale
     * @return the predicted class of each query
     */
    public static int[] classify(Var query, Var support, int[] labels, int classes, float temperature) {
        float[] p = probabilities(query, support, labels, classes, temperature).value().toArray();
        int[] out = new int[query.shape().dim(0)];
        for (int i = 0; i < out.length; i++) {
            int best = 0;
            for (int c = 1; c < classes; c++) {
                if (p[i * classes + c] > p[i * classes + best]) {
                    best = c;
                }
            }
            out[i] = best;
        }
        return out;
    }

    private static Var normalize(Var x) {
        return x.div(x.square().sum(-1, true).add(1e-8f).sqrt());
    }
}
