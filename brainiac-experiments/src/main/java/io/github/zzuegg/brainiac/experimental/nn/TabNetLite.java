package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A compact TabNet (Arik and Pfister, 2019, <a href="https://arxiv.org/abs/1908.07442">arXiv:1908.07442</a>):
 * deep learning for tables by <b>sequential attentive feature selection</b>. In each of
 * several decision steps an attentive transformer produces a {@link Sparsemax} mask over
 * the input features (scaled by a prior that discourages reusing features, with
 * relaxation {@code gamma}); a gated (GLU) feature transformer processes the masked
 * features into a decision part, summed over steps after ReLU, and an attention part that
 * drives the next mask. The masks show which features each prediction used.
 *
 * <p>This lite version keeps the mechanism but drops ghost batch norm and shares one
 * feature transformer across steps. Input {@code [batch, features]}, output
 * {@code [batch, outputs]}.
 */
public final class TabNetLite implements Layer {

    private final int features;
    private final int outputs;
    private final int steps;
    private final int width;
    private final float gamma;
    private final Var shared1;
    private final Var sharedBias1;
    private final List<Var> stepWeights = new ArrayList<>();
    private final List<Var> attention = new ArrayList<>();
    private final Var head;
    private final Var headBias;
    private Tensor lastMasks;

    /**
     * Creates the model.
     *
     * @param features input features
     * @param outputs output values
     * @param steps decision steps
     * @param width the decision and attention width (each)
     * @param gamma the prior relaxation: 1 forces different features per step, larger allows reuse
     * @param random the initialization source
     */
    public TabNetLite(int features, int outputs, int steps, int width, float gamma, SplittableRandom random) {
        this.features = features;
        this.outputs = outputs;
        this.steps = steps;
        this.width = width;
        this.gamma = gamma;
        int glu = 2 * width;
        shared1 = Var.param(Tensor.normal(Shape.of(features, 2 * glu), 0, (float) Math.sqrt(1.0 / features), random));
        sharedBias1 = Var.param(Tensor.zeros(Shape.of(1, 2 * glu)));
        for (int s = 0; s <= steps; s++) {
            stepWeights.add(Var.param(Tensor.normal(Shape.of(glu, 2 * glu), 0, (float) Math.sqrt(1.0 / glu), random)));
            stepWeights.add(Var.param(Tensor.zeros(Shape.of(1, 2 * glu))));
        }
        for (int s = 0; s < steps; s++) {
            attention.add(Var.param(Tensor.normal(Shape.of(width, features), 0, (float) Math.sqrt(1.0 / width), random)));
        }
        head = Var.param(Tensor.normal(Shape.of(width, outputs), 0, (float) Math.sqrt(1.0 / width), random));
        headBias = Var.param(Tensor.zeros(Shape.of(1, outputs)));
    }

    /**
     * @param outputs output values
     * @param steps decision steps
     * @param width decision width
     * @return a spec for {@code Model.sequential().input(features).add(...)}
     */
    public static LayerSpec spec(int outputs, int steps, int width) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new TabNetLite(input.dim(-1), outputs, steps, width, 1.3f, random);
            }
        };
    }

    private static Var glu(Var x, Var w, Var b) {
        Var h = x.matmul(w).add(b);
        int n = h.shape().dim(1) / 2;
        return h.slice(1, 0, n).mul(h.slice(1, n, 2 * n).sigmoid());
    }

    /** The shared GLU layer followed by a step-specific GLU layer with a scaled residual. */
    private Var transform(Var x, int step) {
        Var h = glu(x, shared1, sharedBias1);
        Var h2 = glu(h, stepWeights.get(2 * step), stepWeights.get(2 * step + 1));
        return h.add(h2).mul((float) Math.sqrt(0.5));
    }

    /**
     * @return the feature masks of the last forward pass, {@code [steps, batch, features]},
     *     or {@code null} before the first one
     */
    public Tensor lastMasks() {
        return lastMasks;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().dim(-1) != features) {
            throw new IllegalArgumentException("TabNetLite expects " + features + " features, got " + input.shape());
        }
        int b = input.shape().dim(0);
        Var prior = Var.of(Tensor.full(Shape.of(b, features), 1f));
        Var a = transform(input, 0).slice(1, width, 2 * width);
        Var decision = null;
        List<Tensor> masks = new ArrayList<>();
        for (int s = 0; s < steps; s++) {
            Var mask = Sparsemax.apply(a.matmul(attention.get(s)).mul(prior));
            masks.add(mask.value().reshape(1, b, features));
            prior = prior.mul(mask.neg().add(gamma));
            Var h = transform(input.mul(mask), s + 1);
            Var d = h.slice(1, 0, width).relu();
            decision = decision == null ? d : decision.add(d);
            a = h.slice(1, width, 2 * width);
        }
        lastMasks = Tensor.concat(0, masks.toArray(Tensor[]::new));
        return decision.matmul(head).add(headBias);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(shared1, sharedBias1, head, headBias));
        p.addAll(stepWeights);
        p.addAll(attention);
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    @Override
    public String toString() {
        return "TabNetLite(" + features + " -> " + outputs + ", " + steps + " steps)";
    }
}
