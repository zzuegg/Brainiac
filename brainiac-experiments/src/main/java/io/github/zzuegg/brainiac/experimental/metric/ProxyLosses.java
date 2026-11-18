package io.github.zzuegg.brainiac.experimental.metric;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Proxy-based metric learning: one learnable proxy per class, and samples are compared with
 * proxies instead of with each other, so a batch costs {@code O(n C)} instead of
 * {@code O(n^2)}, no mining is needed and training converges in far fewer steps.
 *
 * <ul>
 *   <li>{@link Kind#PROXY_NCA} (Movshovitz-Attias et al., ICCV 2017): NCA between a sample and
 *       the proxies on normalised vectors:
 *       {@code -log(exp(-d(x, p_y)) / sum_{z != y} exp(-d(x, p_z)))} with squared distances.</li>
 *   <li>{@link Kind#PROXY_NCA_PLUS_PLUS} (Teh et al., ECCV 2020): a softmax over all proxies,
 *       including the positive, at a low temperature ({@code 1/9}).</li>
 *   <li>{@link Kind#PROXY_ANCHOR} (Kim et al., CVPR 2020): each proxy is an anchor that pulls all
 *       its positives and pushes all negatives in the batch through log-sum-exps,
 *       {@code 1/|P+| sum_p log(1 + sum_{x in X+_p} e^{-alpha (s(x,p) - delta)}) +
 *       1/|P| sum_p log(1 + sum_{x in X-_p} e^{alpha (s(x,p) + delta)})} with cosines
 *       {@code s}, {@code alpha = 32}, {@code delta = 0.1}: hard samples get larger gradients,
 *       so data-to-data relations come back without pair mining.</li>
 * </ul>
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1703.07464">arXiv:1703.07464</a>,
 * <a href="https://arxiv.org/abs/2003.13911">arXiv:2003.13911</a>,
 * <a href="https://arxiv.org/abs/2004.01113">arXiv:2004.01113</a>.
 */
public final class ProxyLosses implements Loss {

    /** The proxy loss. */
    public enum Kind {
        /** Proxy-NCA. */
        PROXY_NCA,
        /** Proxy-NCA++ (all proxies in the softmax, low temperature). */
        PROXY_NCA_PLUS_PLUS,
        /** Proxy-Anchor. */
        PROXY_ANCHOR
    }

    private final Kind kind;
    private final int classes;
    private final Var proxies;

    /**
     * Creates the loss with random proxies.
     *
     * @param kind the loss
     * @param dim the embedding width
     * @param classes the training classes
     * @param seed the seed of the proxies
     */
    public ProxyLosses(Kind kind, int dim, int classes, long seed) {
        this.kind = kind;
        this.classes = classes;
        this.proxies = Var.param(Tensor.normal(Shape.of(classes, dim), 0, 1, new SplittableRandom(seed)));
    }

    /** @return the trainable proxies {@code [classes, dim]} (papers train them with a larger learning rate) */
    public List<Var> parameters() {
        return List.of(proxies);
    }

    @Override
    public Var compute(Var prediction, Tensor target) {
        int n = prediction.shape().dim(0);
        int[] y = MetricMath.labels(target);
        float[] hot = new float[n * classes];
        for (int i = 0; i < n; i++) {
            hot[i * classes + y[i]] = 1;
        }
        var dev = prediction.value().device();
        Tensor oneHot = Tensor.of(Shape.of(n, classes), hot, dev);
        Var cos = MetricMath.cosine(prediction, proxies);
        return switch (kind) {
            case PROXY_NCA -> {
                // squared distance of unit vectors: 2 - 2 cos
                Var negDist = cos.mul(2f).sub(2f);
                Var pos = negDist.mul(oneHot).sum(1, false);
                Var others = negDist.add(oneHot.mul(-1e4f));
                yield others.logsumexp(1, false).sub(pos).mean();
            }
            case PROXY_NCA_PLUS_PLUS -> Functional.crossEntropy(cos.mul(2f).sub(2f).mul(9f), target);
            case PROXY_ANCHOR -> {
                float alpha = 32;
                float delta = 0.1f;
                float[] present = new float[classes];
                for (int c : y) {
                    present[c] = 1;
                }
                int positives = 0;
                for (float p : present) {
                    positives += p > 0 ? 1 : 0;
                }
                Tensor negMask = oneHot.mul(-1f).add(1f);
                Var posSum = cos.sub(delta).mul(-alpha).exp().mul(oneHot).sum(0, false).add(1f).log()
                        .mul(Tensor.of(Shape.of(classes), present, dev)).sum().div(Math.max(1, positives));
                Var negSum = cos.add(delta).mul(alpha).exp().mul(negMask).sum(0, false).add(1f).log().sum()
                        .div(classes);
                yield posSum.add(negSum);
            }
        };
    }
}
