package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Concrete dropout: dropout on the input of a wrapped layer whose drop rate {@code p} is
 * <em>learned</em>. The Bernoulli keep mask is replaced by its concrete (relaxed Bernoulli)
 * version {@code z = sigmoid((logit p + log u - log(1 - u)) / t)}, so the rate gets a
 * gradient. The layer's {@link #auxiliaryLoss()} is the variational regulariser of the paper,
 * {@code weightReg * |W|^2 / (1 - p) + dropoutReg * K * (p log p + (1 - p) log(1 - p))}
 * ({@code K} = input features), which trades off fitting against the entropy of the mask, so
 * no grid search over rates is needed. {@code Model.auxiliaryLoss()} picks it up.
 *
 * <p>Paper: Gal, Hron and Kendall, "Concrete Dropout" (NeurIPS 2017),
 * <a href="https://arxiv.org/abs/1705.07832">arXiv:1705.07832</a>.
 */
public final class ConcreteDropout implements Layer {

    private final Layer inner;
    private final int features;
    private final Var logit;
    private final float temperature;
    private final float weightRegularizer;
    private final float dropoutRegularizer;
    private final RandomGenerator random;
    private Var auxiliary;

    /**
     * Wraps a layer.
     *
     * @param inner the layer whose input is dropped
     * @param features the input features per example (for the entropy term)
     * @param initialRate the starting drop rate, in {@code (0, 1)}
     * @param weightRegularizer the weight term, about {@code lengthScale^2 / N}
     * @param dropoutRegularizer the entropy term, about {@code 2 / N}
     * @param random the source of mask noise
     */
    public ConcreteDropout(Layer inner, int features, float initialRate, float weightRegularizer,
            float dropoutRegularizer, RandomGenerator random) {
        if (!(initialRate > 0 && initialRate < 1)) {
            throw new IllegalArgumentException("initial rate must be in (0, 1), got " + initialRate);
        }
        this.inner = inner;
        this.features = features;
        this.logit = Var.param(Tensor.full(Shape.of(1), (float) Math.log(initialRate / (1 - initialRate))));
        this.temperature = 0.1f;
        this.weightRegularizer = weightRegularizer;
        this.dropoutRegularizer = dropoutRegularizer;
        this.random = random;
    }

    /**
     * Describes a concrete-dropout wrapper around another layer for model builders.
     *
     * @param inner the wrapped layer
     * @param initialRate the starting drop rate
     * @param weightRegularizer the weight term
     * @param dropoutRegularizer the entropy term
     * @return the spec
     */
    public static LayerSpec spec(LayerSpec inner, float initialRate, float weightRegularizer,
            float dropoutRegularizer) {
        return ClassicOps.spec(inner::outputShape, (s, r) -> {
            int features = Math.toIntExact(s.size());
            return new ConcreteDropout(inner.create(s, r), features, initialRate, weightRegularizer,
                    dropoutRegularizer, r);
        });
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (!training) {
            auxiliary = null;
            return inner.forward(input, false);
        }
        float eps = 1e-6f;
        Tensor u = ClassicOps.uniform(input.shape(), eps, 1 - eps, random, input.value().device());
        Tensor noise = u.log().sub(u.neg().add(1f).log());
        Var drop = logit.add(Var.of(noise)).div(temperature).sigmoid();
        Var p = logit.sigmoid();
        Var keep = p.neg().add(1f);
        Var out = inner.forward(input.mul(drop.neg().add(1f)).div(keep), true);
        Var squares = null;
        for (Var w : inner.parameters()) {
            Var s = w.square().sum();
            squares = squares == null ? s : squares.add(s);
        }
        Var entropy = p.mul(p.add(1e-7f).log()).add(keep.mul(keep.add(1e-7f).log())).mul(dropoutRegularizer * features);
        Var aux = entropy.sum();
        if (squares != null) {
            aux = aux.add(squares.div(keep).mul(weightRegularizer).sum());
        }
        Var innerAux = inner.auxiliaryLoss();
        auxiliary = innerAux == null ? aux : aux.add(innerAux);
        return out;
    }

    @Override
    public Var auxiliaryLoss() {
        return auxiliary;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(inner.parameters());
        p.add(logit);
        return p;
    }

    @Override
    public List<Var> buffers() {
        return inner.buffers();
    }

    @Override
    public Shape outputShape(Shape inputShape) {
        return inner.outputShape(inputShape);
    }

    @Override
    public int outputSize(int inputSize) {
        return inner.outputSize(inputSize);
    }

    /** @return the learned drop rate */
    public float rate() {
        return logit.value().sigmoid().item();
    }

    @Override
    public String toString() {
        return "ConcreteDropout(" + inner + ", p=" + rate() + ")";
    }
}
