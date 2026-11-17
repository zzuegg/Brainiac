package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Distance-aware output heads: single deterministic networks whose uncertainty grows with
 * the distance of an input's features from the training data, so one forward pass can flag
 * out-of-distribution inputs. Both need a body that roughly preserves distances (spectral
 * normalization, as in {@link SpectralNormDense}, or a gradient penalty).
 *
 * <ul>
 *   <li>{@link Kind#SNGP} (Liu et al. 2020, <a href="https://arxiv.org/abs/2006.10108">arXiv:2006.10108</a>):
 *       a Gaussian-process output layer approximated by random Fourier features
 *       {@code phi = sqrt(2/D) cos(h W + b)} and a linear read-out; after training, a
 *       Laplace approximation of the read-out's posterior ({@link #fitCovariance}) gives a
 *       predictive variance {@code phi^T Sigma phi}, applied by the mean-field adjustment
 *       {@code logits / sqrt(1 + pi/8 var)}.</li>
 *   <li>{@link Kind#DUQ} (van Amersfoort et al. 2020, <a href="https://arxiv.org/abs/2003.02037">arXiv:2003.02037</a>):
 *       deterministic uncertainty quantification: one centroid per class in a learned
 *       embedding, an RBF kernel {@code K_c = exp(-|W_c h - e_c|^2 / (2 m sigma^2))} as the
 *       output, trained by binary cross-entropy per class; the centroids follow the class
 *       means by exponential moving average ({@link #updateCentroids}). Certainty is
 *       {@code max_c K_c}. {@link #gradientPenalty} is the paper's two-sided penalty.</li>
 * </ul>
 */
public final class DistanceAwareHead implements Layer {

    /** The head type. */
    public enum Kind {
        /** Spectral-normalized neural Gaussian process head. */
        SNGP,
        /** Deterministic uncertainty quantification (RBF centroids). */
        DUQ
    }

    private final Kind kind;
    private final int classes;
    private final int features;
    private final Tensor rffWeight;
    private final Tensor rffBias;
    private final Var readout;
    private final Var embed;
    private final float sigma;
    private final float gamma;
    private Tensor centroidSums;
    private Tensor centroidCounts;
    private Tensor covariance;

    private DistanceAwareHead(Kind kind, int inputs, int classes, int features, float sigma, RandomGenerator random) {
        this.kind = kind;
        this.classes = classes;
        this.features = features;
        this.sigma = sigma;
        this.gamma = 0.99f;
        if (kind == Kind.SNGP) {
            this.rffWeight = Tensor.normal(Shape.of(inputs, features), 0f, 1f / sigma, random);
            this.rffBias = Tensor.uniform(Shape.of(features), 0f, (float) (2 * Math.PI), random);
            this.readout = Var.param(Tensor.normal(Shape.of(features, classes), 0f, 0.01f, random));
            this.embed = null;
        } else {
            this.rffWeight = null;
            this.rffBias = null;
            this.readout = null;
            this.embed = Var.param(Tensor.normal(Shape.of(inputs, classes * features), 0f,
                    (float) Math.sqrt(1.0 / inputs), random));
            this.centroidCounts = Tensor.full(Shape.of(classes, 1), 12f);
            this.centroidSums = Tensor.normal(Shape.of(classes, features), 0f, 1f, random).mul(12f);
        }
    }

    /**
     * An SNGP head.
     *
     * @param inputs input features (the body's output)
     * @param classes the number of classes
     * @param randomFeatures the number of random Fourier features
     * @param lengthScale the RBF kernel's length scale in feature space (the random
     *        frequencies are {@code N(0, 1 / lengthScale^2)})
     * @param random the initializer
     * @return the head; its output is the logits
     */
    public static DistanceAwareHead sngp(int inputs, int classes, int randomFeatures, float lengthScale,
            RandomGenerator random) {
        return new DistanceAwareHead(Kind.SNGP, inputs, classes, randomFeatures, lengthScale, random);
    }

    /**
     * A DUQ head.
     *
     * @param inputs input features (the body's output)
     * @param classes the number of classes
     * @param embedding the centroid embedding size {@code m}
     * @param lengthScale the kernel length scale {@code sigma}, such as 0.1
     * @param random the initializer
     * @return the head; its output is the kernel values {@code K_c} in {@code (0, 1]}
     */
    public static DistanceAwareHead duq(int inputs, int classes, int embedding, float lengthScale,
            RandomGenerator random) {
        return new DistanceAwareHead(Kind.DUQ, inputs, classes, embedding, lengthScale, random);
    }

    /** @return the head type */
    public Kind kind() {
        return kind;
    }

    private Var phi(Var h) {
        return h.matmul(Var.of(rffWeight)).add(rffBias).cos().mul((float) Math.sqrt(2.0 / features));
    }

    private Var embedded(Var h) {
        return h.matmul(embed).reshape(h.shape().dim(0), classes, features);
    }

    /**
     * The logits (SNGP) or kernel values (DUQ).
     *
     * @param input the body's features {@code [batch, inputs]}
     * @return {@code [batch, classes]}
     */
    @Override
    public Var forward(Var input) {
        if (kind == Kind.SNGP) {
            return phi(input).matmul(readout);
        }
        Var z = embedded(input);
        Tensor e = centroidSums.div(centroidCounts).reshape(1, classes, features);
        Var d = z.sub(e).square().mean(2, false);
        return d.mul(-1f / (2 * sigma * sigma)).exp();
    }

    /**
     * The training loss: cross-entropy (SNGP) or per-class binary cross-entropy (DUQ).
     *
     * @param output this head's output
     * @param ids the class ids
     * @return the loss
     */
    public Var loss(Var output, Tensor ids) {
        if (kind == Kind.SNGP) {
            return Functional.crossEntropy(output, ids).add(readout.square().sum().mul(1e-4f));
        }
        int n = ids.shape().dim(0);
        Tensor y = ids.reshape(n, 1).equalTo(Tensor.arange(classes).reshape(1, classes));
        Var k = output.clamp(1e-6f, 1 - 1e-6f);
        return k.log().mul(y).add(k.neg().add(1f).log().mul(y.neg().add(1f))).mean().mul(-classes);
    }

    /**
     * DUQ's centroid update after an optimizer step: an exponential moving average of each
     * class's embedded features.
     *
     * @param input the batch's body features
     * @param ids the batch's class ids
     */
    public void updateCentroids(Tensor input, Tensor ids) {
        if (kind != Kind.DUQ) {
            return;
        }
        int n = ids.shape().dim(0);
        Tensor y = ids.reshape(n, 1).equalTo(Tensor.arange(classes).reshape(1, classes));
        Tensor z = embedded(Var.of(input)).value();
        Tensor counts = y.sum(0, false).reshape(classes, 1);
        Tensor sums = y.transpose().reshape(classes, n, 1).mul(z.permute(1, 0, 2)).sum(1, false);
        centroidCounts = centroidCounts.mul(gamma).add(counts.mul(1 - gamma));
        centroidSums = centroidSums.mul(gamma).add(sums.mul(1 - gamma));
    }

    /**
     * DUQ's two-sided gradient penalty {@code lambda (|d sum_c K_c / dx|^2 - 1)^2}, which
     * keeps the features sensitive to the input; {@code input} must be a parameter-like
     * {@link Var} the output was computed from.
     *
     * @param input the model input, created with {@code Var.param}
     * @param output this head's output for it
     * @param lambda the penalty weight
     * @return the penalty
     */
    public static Var gradientPenalty(Var input, Var output, float lambda) {
        Var g = Var.gradients(output.sum(), List.of(input), true).get(0);
        return g.square().sum(1, false).sub(1f).square().mean().mul(lambda);
    }

    /**
     * SNGP's Laplace step after training: the posterior covariance of the read-out,
     * {@code (ridge I + sum_i w_i phi_i phi_i^T)^-1}. With {@code laplaceWeights} the
     * weights are the paper's {@code w_i = p_i (1 - p_i)} ({@code p_i} the top softmax
     * probability); otherwise {@code w_i = 1}, the Gaussian-likelihood form the reference
     * implementation uses by default, which keeps the variance informative when the
     * training points are classified with near certainty (and so contribute almost nothing
     * to the Laplace precision).
     *
     * @param inputs the training set's body features
     * @param ridge the prior precision
     * @param laplaceWeights whether to weight by {@code p (1 - p)}
     */
    public void fitCovariance(Tensor inputs, float ridge, boolean laplaceWeights) {
        if (kind != Kind.SNGP) {
            return;
        }
        Tensor f = phi(Var.of(inputs)).value();
        Tensor weighted = f;
        if (laplaceWeights) {
            Var logits = forward(Var.of(inputs));
            Tensor p = io.github.zzuegg.brainiac.nn.Activation.SOFTMAX.forward(logits).value().max(1, true);
            weighted = f.mul(p.mul(p.neg().add(1f)));
        }
        double[] precision = weighted.transpose().matmul(f).toDoubleArray();
        for (int i = 0; i < features; i++) {
            precision[i * features + i] += ridge;
        }
        covariance = Linalg.inverse(Tensor.of(Shape.of(features, features), precision, DType.FLOAT64, Device.cpu()))
                .to(DType.FLOAT32);
    }

    /**
     * The class probabilities: SNGP's variance-adjusted softmax (after {@link #fitCovariance}),
     * or DUQ's kernel values normalized to sum to 1.
     *
     * @param input the body's features
     * @return {@code [batch, classes]}
     */
    public Tensor probabilities(Tensor input) {
        Var out = forward(Var.of(input));
        if (kind == Kind.SNGP) {
            Tensor logits = out.value();
            if (covariance != null) {
                Tensor f = phi(Var.of(input)).value();
                Tensor var = f.matmul(covariance).mul(f).sum(1, true);
                logits = logits.div(var.mul((float) (Math.PI / 8)).add(1f).sqrt());
            }
            return io.github.zzuegg.brainiac.nn.Activation.SOFTMAX.forward(Var.of(logits)).value();
        }
        Tensor k = out.value();
        return k.div(k.sum(1, true).add(1e-12f));
    }

    /**
     * An out-of-distribution score, higher for less familiar inputs: SNGP's predictive
     * variance {@code phi^T Sigma phi}, DUQ's {@code 1 - max_c K_c}.
     *
     * @param input the body's features
     * @return {@code [batch]}
     */
    public Tensor uncertainty(Tensor input) {
        if (kind == Kind.SNGP) {
            if (covariance == null) {
                throw new IllegalStateException("call fitCovariance first");
            }
            Tensor f = phi(Var.of(input)).value();
            return f.matmul(covariance).mul(f).sum(1, false);
        }
        return forward(Var.of(input)).value().max(1, false).neg().add(1f);
    }

    @Override
    public List<Var> parameters() {
        return kind == Kind.SNGP ? List.of(readout) : List.of(embed);
    }

    @Override
    public int outputSize(int inputSize) {
        return classes;
    }

    @Override
    public String toString() {
        return kind + "Head(" + classes + ")";
    }
}
