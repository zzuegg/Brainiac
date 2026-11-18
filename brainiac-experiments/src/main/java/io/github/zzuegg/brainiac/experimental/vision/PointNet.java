package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * PointNet (Qi et al., 2017): a point-cloud classifier that applies one shared MLP to every
 * point and max-pools over the points, so the result does not depend on their order. An
 * optional T-Net (itself a small PointNet) first predicts a 3x3 input transform, initialised
 * at the identity, that aligns the cloud before the shared MLP sees it.
 *
 * <p>Input {@code [batch, points, 3]}, output {@code [batch, classes]} logits.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1612.00593">arXiv:1612.00593</a>.
 */
public final class PointNet implements Layer {

    private final int classes;
    private final boolean transform;
    private final List<Dense> tnet = new ArrayList<>();
    private final List<Dense> shared = new ArrayList<>();
    private final List<Dense> head = new ArrayList<>();

    /**
     * Creates the network.
     *
     * @param classes the number of classes
     * @param width the width of the pooled feature
     * @param transform whether to use the T-Net
     * @param random the source of initial weights
     */
    public PointNet(int classes, int width, boolean transform, RandomGenerator random) {
        this.classes = classes;
        this.transform = transform;
        if (transform) {
            tnet.add(new Dense(3, 32, Activation.RELU, Init.HE, random));
            tnet.add(new Dense(32, 64, Activation.RELU, Init.HE, random));
            tnet.add(new Dense(64, 32, Activation.RELU, Init.HE, random));
            float[] eye = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            tnet.add(new Dense(32, 9, Activation.IDENTITY, Tensor.zeros(Shape.of(32, 9)), Tensor.of(Shape.of(9), eye)));
        }
        shared.add(new Dense(3, 32, Activation.RELU, Init.HE, random));
        shared.add(new Dense(32, 64, Activation.RELU, Init.HE, random));
        shared.add(new Dense(64, width, Activation.RELU, Init.HE, random));
        head.add(new Dense(width, 64, Activation.RELU, Init.HE, random));
        head.add(new Dense(64, classes, Activation.IDENTITY, Init.XAVIER, random));
    }

    /**
     * The input transform the T-Net predicts.
     *
     * @param points {@code [batch, points, 3]}
     * @return {@code [batch, 3, 3]}
     */
    public Var alignment(Var points) {
        int b = points.shape().dim(0);
        Var h = tnet.get(1).forward(tnet.get(0).forward(points)).max(1, false);
        return tnet.get(3).forward(tnet.get(2).forward(h)).reshape(b, 3, 3);
    }

    /**
     * The pooled global feature.
     *
     * @param points {@code [batch, points, 3]}
     * @return {@code [batch, width]}
     */
    public Var globalFeature(Var points) {
        Var x = transform ? points.matmul(alignment(points)) : points;
        for (Dense d : shared) {
            x = d.forward(x);
        }
        return x.max(1, false);
    }

    @Override
    public Var forward(Var input) {
        Var h = globalFeature(input);
        for (Dense d : head) {
            h = d.forward(h);
        }
        return h;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), classes);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (List<Dense> group : List.of(tnet, shared, head)) {
            for (Dense d : group) {
                p.addAll(d.parameters());
            }
        }
        return p;
    }

    @Override
    public String toString() {
        return "PointNet(" + classes + " classes" + (transform ? ", T-Net" : "") + ")";
    }
}
