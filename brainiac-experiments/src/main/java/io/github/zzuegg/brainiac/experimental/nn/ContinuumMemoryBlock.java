package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A continuum memory system block (the CMS of HOPE, Nested Learning, Behrouz et al. 2025,
 * <a href="https://arxiv.org/abs/2512.24695">arXiv:2512.24695</a>): a chain of residual MLPs
 * {@code x + MLP_i(x)}, one per memory level, where level {@code i} is meant to be updated
 * every {@code periods[i]} steps. A transformer's single feed-forward layer is the special
 * case of one level updated every step; several levels spread knowledge over time scales,
 * so fast levels adapt to the current context while slow ones hold on to what was learned
 * long ago.
 *
 * <p>Pair it with {@code experimental.train.ContinuumOptimizer}, giving each level's
 * {@link #levelParameters(int)} its {@link #periods() period}. Input {@code [..., features]},
 * output the same shape.
 */
public final class ContinuumMemoryBlock implements Layer {

    private final int features;
    private final int[] periods;
    private final List<Var[]> levels = new ArrayList<>();

    /**
     * Creates the block.
     *
     * @param features the features
     * @param hidden the hidden units of each level's MLP
     * @param periods each level's update period, fastest first by convention
     * @param random the random source for the weights
     */
    public ContinuumMemoryBlock(int features, int hidden, int[] periods, SplittableRandom random) {
        if (periods.length == 0) {
            throw new IllegalArgumentException("need at least one level");
        }
        this.features = features;
        this.periods = periods.clone();
        for (int i = 0; i < periods.length; i++) {
            levels.add(new Var[] {
                Var.param(Init.HE.create(Shape.of(features, hidden), features, hidden, random)),
                Var.param(Tensor.zeros(Shape.of(hidden))),
                Var.param(Tensor.normal(Shape.of(hidden, features), 0, 0.01f, random)),
                Var.param(Tensor.zeros(Shape.of(features)))
            });
        }
    }

    /**
     * Describes the block for model builders.
     *
     * @param hidden the hidden units per level
     * @param periods each level's update period
     * @return the spec
     */
    public static LayerSpec spec(int hidden, int... periods) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new ContinuumMemoryBlock(input.dim(-1), hidden, periods, random);
            }
        };
    }

    /** @return each level's update period */
    public int[] periods() {
        return periods.clone();
    }

    /**
     * @param level the level
     * @return its parameters
     */
    public List<Var> levelParameters(int level) {
        return List.of(levels.get(level));
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().dim(-1) != features) {
            throw new IllegalArgumentException(this + " expects " + features + " features, got " + input.shape());
        }
        Var x = input;
        for (Var[] l : levels) {
            x = x.add(Activation.GELU.forward(x.matmul(l[0]).add(l[1])).matmul(l[2]).add(l[3]));
        }
        return x;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Var[] l : levels) {
            p.addAll(List.of(l));
        }
        return p;
    }

    @Override
    public String toString() {
        return "ContinuumMemoryBlock(periods " + java.util.Arrays.toString(periods) + ")";
    }
}
