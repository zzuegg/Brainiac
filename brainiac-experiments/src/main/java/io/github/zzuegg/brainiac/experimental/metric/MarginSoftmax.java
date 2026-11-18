package io.github.zzuegg.brainiac.experimental.metric;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Angular-margin softmax losses, the standard training objective for face and speaker
 * verification. Both the embedding and each class weight are L2-normalised, so the logits are
 * cosines {@code cos(theta_j)} scaled by {@code s}; a margin is then taken off the true class
 * only, which forces same-class embeddings into tight angular clusters:
 *
 * <ul>
 *   <li>{@link Kind#NORMALIZED}: no margin (NormFace), the ablation;</li>
 *   <li>{@link Kind#COSFACE}: {@code s (cos theta_y - m)} (Wang et al., CVPR 2018);</li>
 *   <li>{@link Kind#ARCFACE}: {@code s cos(theta_y + m)} (Deng et al., CVPR 2019), an additive
 *       angular margin. Past {@code theta = pi - m} the function stops being monotone, so the
 *       reference code falls back to {@code cos theta - m sin(m)} there; the "easy margin"
 *       variant applies the margin only where {@code cos theta > 0}.</li>
 * </ul>
 *
 * <p>Sub-center ArcFace (Deng et al., ECCV 2020) gives each class {@code K} centres and uses
 * the closest one ({@code max} over the class's cosines), so noisy or multi-modal classes do
 * not pull one centre apart.
 *
 * <p>As a core {@link Loss} the prediction is the embedding batch {@code [n, d]} and the target
 * the class ids; the class weights are trainable {@link #parameters()}.
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1801.09414">arXiv:1801.09414</a>,
 * <a href="https://arxiv.org/abs/1801.07698">arXiv:1801.07698</a>.
 */
public final class MarginSoftmax implements Loss {

    /** The margin type. */
    public enum Kind {
        /** Normalised softmax, no margin. */
        NORMALIZED,
        /** Additive cosine margin. */
        COSFACE,
        /** Additive angular margin. */
        ARCFACE
    }

    private final Kind kind;
    private final int classes;
    private final int subCenters;
    private final float scale;
    private final float margin;
    private final boolean easyMargin;
    private final Var weights;

    /**
     * Creates the loss with random class centres.
     *
     * @param kind the margin type
     * @param dim the embedding width
     * @param classes the number of training classes
     * @param scale the logit scale {@code s} (64 for faces; 16 suits a few classes)
     * @param margin the margin {@code m} (0.35 CosFace, 0.5 ArcFace in the papers)
     * @param subCenters centres per class, 1 for plain ArcFace
     * @param easyMargin ArcFace's easy-margin variant
     * @param seed the seed of the centres
     */
    public MarginSoftmax(Kind kind, int dim, int classes, float scale, float margin, int subCenters,
            boolean easyMargin, long seed) {
        if (subCenters < 1) {
            throw new IllegalArgumentException("need at least one centre per class: " + subCenters);
        }
        this.kind = kind;
        this.classes = classes;
        this.subCenters = subCenters;
        this.scale = scale;
        this.margin = margin;
        this.easyMargin = easyMargin;
        this.weights = Var.param(Tensor.normal(Shape.of(dim, classes * subCenters), 0, 1, new SplittableRandom(seed)));
    }

    /**
     * ArcFace with {@code s = 16}, {@code m = 0.3}, one centre.
     *
     * @param dim the embedding width
     * @param classes the classes
     * @param seed the seed
     * @return the loss
     */
    public static MarginSoftmax arcFace(int dim, int classes, long seed) {
        return new MarginSoftmax(Kind.ARCFACE, dim, classes, 16, 0.3f, 1, false, seed);
    }

    /**
     * CosFace with {@code s = 16}, {@code m = 0.25}.
     *
     * @param dim the embedding width
     * @param classes the classes
     * @param seed the seed
     * @return the loss
     */
    public static MarginSoftmax cosFace(int dim, int classes, long seed) {
        return new MarginSoftmax(Kind.COSFACE, dim, classes, 16, 0.25f, 1, false, seed);
    }

    /**
     * Normalised softmax with {@code s = 16} and no margin.
     *
     * @param dim the embedding width
     * @param classes the classes
     * @param seed the seed
     * @return the loss
     */
    public static MarginSoftmax normalized(int dim, int classes, long seed) {
        return new MarginSoftmax(Kind.NORMALIZED, dim, classes, 16, 0, 1, false, seed);
    }

    /** @return the trainable class centres {@code [dim, classes * subCenters]} */
    public List<Var> parameters() {
        return List.of(weights);
    }

    /**
     * The cosine of each embedding with each class (its closest sub-centre).
     *
     * @param embeddings {@code [n, dim]}
     * @return {@code [n, classes]}
     */
    public Var cosines(Var embeddings) {
        int n = embeddings.shape().dim(0);
        Var e = embeddings.div(embeddings.square().sum(-1, true).add(1e-8f).sqrt());
        Var w = weights.div(weights.square().sum(0, true).add(1e-8f).sqrt());
        Var cos = e.matmul(w);
        if (subCenters > 1) {
            cos = cos.reshape(n, classes, subCenters).max(2, false);
        }
        return cos.clamp(-1 + 1e-6f, 1 - 1e-6f);
    }

    /**
     * The scaled logits with the margin applied to each row's true class.
     *
     * @param embeddings {@code [n, dim]}
     * @param labels class ids {@code [n]}
     * @return {@code [n, classes]}
     */
    public Var logits(Var embeddings, Tensor labels) {
        Var cos = cosines(embeddings);
        int n = cos.shape().dim(0);
        int[] y = MetricMath.labels(labels);
        float[] hot = new float[n * classes];
        for (int i = 0; i < n; i++) {
            hot[i * classes + y[i]] = 1;
        }
        Tensor oneHot = Tensor.of(Shape.of(n, classes), hot, cos.value().device());
        Var target = switch (kind) {
            case NORMALIZED -> cos;
            case COSFACE -> cos.sub(margin);
            case ARCFACE -> {
                Var sin = cos.square().neg().add(1f).maximum(1e-12f).sqrt();
                Var phi = cos.mul((float) Math.cos(margin)).sub(sin.mul((float) Math.sin(margin)));
                if (easyMargin) {
                    yield Var.where(cos.value().greater(Tensor.zeros(cos.shape())), phi, cos);
                }
                float th = (float) Math.cos(Math.PI - margin);
                float mm = (float) (Math.sin(Math.PI - margin) * margin);
                yield Var.where(cos.value().greater(Tensor.full(cos.shape(), th)), phi, cos.sub(mm));
            }
        };
        return cos.add(target.sub(cos).mul(oneHot)).mul(scale);
    }

    @Override
    public Var compute(Var prediction, Tensor target) {
        return Functional.crossEntropy(logits(prediction, target), target);
    }
}
