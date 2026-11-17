package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Evidential deep learning for classification: the head outputs non-negative
 * <em>evidence</em> per class, read as the parameters {@code alpha = evidence + 1} of a
 * Dirichlet distribution over the class probabilities. One forward pass gives both a
 * prediction, {@code p = alpha / S} with {@code S = sum alpha}, and an uncertainty,
 * {@code u = K / S}, which is 1 when the network has no evidence at all (as it should on
 * inputs unlike its training data).
 *
 * <p>The loss ({@link #loss}) is the expected squared error under the Dirichlet,
 * {@code sum_j (y_j - p_j)^2 + p_j (1 - p_j) / (S + 1)}, plus an annealed
 * {@code KL(Dir(alpha~) || Dir(1))} on the evidence left after removing the true class's
 * ({@code alpha~ = y + (1 - y) alpha}), which pushes misleading evidence to zero. The
 * log-gamma and digamma it needs are computed from their asymptotic series after a
 * two-step recurrence shift, as autograd operations.
 *
 * <p>Paper: Sensoy, Kaplan, Kandemir, "Evidential Deep Learning to Quantify Classification
 * Uncertainty" (2018), <a href="https://arxiv.org/abs/1806.01768">arXiv:1806.01768</a>.
 */
public final class EvidentialHead implements Layer {

    private final int classes;
    private final Var weight;
    private final Var bias;

    /**
     * Creates the head.
     *
     * @param inputs input features
     * @param classes the number of classes
     * @param random the initializer
     */
    public EvidentialHead(int inputs, int classes, RandomGenerator random) {
        this.classes = classes;
        this.weight = Var.param(Tensor.normal(Shape.of(inputs, classes), 0f, (float) Math.sqrt(1.0 / inputs), random));
        this.bias = Var.param(Tensor.zeros(Shape.of(classes)));
    }

    /**
     * Describes the head for {@code Model.sequential().add(...)}.
     *
     * @param classes the number of classes
     * @return the spec
     */
    public static LayerSpec spec(int classes) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(classes);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new EvidentialHead(input.dim(-1), classes, random);
            }
        };
    }

    /**
     * The Dirichlet parameters.
     *
     * @param input {@code [batch, inputs]}
     * @return {@code alpha}, {@code [batch, classes]}, each at least 1
     */
    @Override
    public Var forward(Var input) {
        return input.matmul(weight).add(bias).softplus().add(1f);
    }

    /**
     * The evidential loss.
     *
     * @param alpha the head's output {@code [batch, classes]}
     * @param oneHot the labels {@code [batch, classes]}
     * @param klWeight the weight of the misleading-evidence KL, usually annealed from 0 to 1
     * @return the mean loss
     */
    public static Var loss(Var alpha, Tensor oneHot, float klWeight) {
        int k = alpha.shape().dim(1);
        Var s = alpha.sum(1, true);
        Var p = alpha.div(s);
        Var err = p.sub(oneHot).square().add(p.mul(p.neg().add(1f)).div(s.add(1f))).sum(1, false);
        Tensor notY = oneHot.neg().add(1f);
        Var tilde = alpha.mul(notY).add(oneHot);
        Var st = tilde.sum(1, true);
        Var kl = lgamma(st).sum(1, false).sub((float) lgammaDouble(k))
                .sub(lgamma(tilde).sum(1, false))
                .add(tilde.sub(1f).mul(digamma(tilde).sub(digamma(st))).sum(1, false));
        return err.add(kl.mul(klWeight)).mean();
    }

    /**
     * {@code log Gamma(x)} for {@code x >= 1}, as an autograd value.
     *
     * @param x the arguments
     * @return the log-gamma values
     */
    public static Var lgamma(Var x) {
        Var z = x.add(2f);
        Var series = z.sub(0.5f).mul(z.log()).sub(z).add(0.5f * (float) Math.log(2 * Math.PI))
                .add(z.pow(-1f).mul(1f / 12)).sub(z.pow(-3f).mul(1f / 360));
        return series.sub(x.log()).sub(x.add(1f).log());
    }

    /**
     * The digamma function {@code d/dx log Gamma(x)} for {@code x >= 1}, as an autograd value.
     *
     * @param x the arguments
     * @return the digamma values
     */
    public static Var digamma(Var x) {
        Var z = x.add(2f);
        Var series = z.log().sub(z.pow(-1f).mul(0.5f)).sub(z.pow(-2f).mul(1f / 12)).add(z.pow(-4f).mul(1f / 120));
        return series.sub(x.pow(-1f)).sub(x.add(1f).pow(-1f));
    }

    private static double lgammaDouble(int n) {
        double s = 0;
        for (int i = 2; i < n; i++) {
            s += Math.log(i);
        }
        return s;
    }

    /**
     * The expected class probabilities {@code alpha / S}.
     *
     * @param alpha the head's output
     * @return {@code [batch, classes]}
     */
    public static Tensor probabilities(Tensor alpha) {
        return alpha.div(alpha.sum(1, true));
    }

    /**
     * The vacuity {@code K / S} of every row: 1 without evidence, toward 0 with a lot.
     *
     * @param alpha the head's output
     * @return {@code [batch]}
     */
    public static Tensor uncertainty(Tensor alpha) {
        return alpha.sum(1, false).pow(-1f).mul(alpha.shape().dim(1));
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return classes;
    }

    @Override
    public String toString() {
        return "EvidentialHead(" + classes + ")";
    }
}
