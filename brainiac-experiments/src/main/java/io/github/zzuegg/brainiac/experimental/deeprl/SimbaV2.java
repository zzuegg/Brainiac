package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.HlGaussValue;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * SimbaV2 (Lee et al., ICML 2025, <a href="https://arxiv.org/abs/2502.15280">arXiv:2502.15280</a>):
 * hyperspherical normalisation for scalable RL. Features live on the unit hypersphere and
 * weights are projected back onto it after every update, which removes the need for weight
 * decay and keeps the effective learning rate from drifting as parameter norms grow.
 *
 * <ul>
 *   <li>{@link Embedding}: a constant coordinate is appended to the input before
 *       ℓ2-normalisation (so its magnitude is not lost), then a bias-free linear map, a
 *       learnable per-channel scaler and another ℓ2-normalisation.</li>
 *   <li>{@link Block}: a bias-free inverted-bottleneck MLP with scalers, normalised, then mixed
 *       into the stream by a learnable linear interpolation (LERP) and re-normalised.</li>
 *   <li>{@link #project}: the weight projection run after each optimizer step.</li>
 *   <li>{@link #learner}: the networks on {@link ModernSac} with a categorical (HL-Gauss)
 *       critic, reward scaling by the running return's standard deviation and a linearly
 *       decaying learning rate.</li>
 * </ul>
 */
public final class SimbaV2 {

    private SimbaV2() {
    }

    private static Var l2(Var x) {
        int axis = x.shape().rank() - 1;
        return x.div(x.square().sum(axis, true).add(1e-6f).sqrt());
    }

    /** A bias-free linear map whose weight columns are kept at unit norm by {@link #project()}. */
    private static final class SphereLinear {
        final Var weight;
        final Var scaler;

        SphereLinear(int in, int out, SplittableRandom random) {
            weight = Var.param(Tensor.normal(Shape.of(in, out), 0, 1, random));
            scaler = Var.param(Tensor.ones(Shape.of(out)));
            project();
        }

        Var forward(Var x) {
            return x.matmul(weight).mul(scaler);
        }

        void project() {
            Tensor w = weight.value();
            weight.assign(w.div(w.square().sum(0, true).add(1e-8f).sqrt()));
        }
    }

    /** Projectable layers. */
    private interface Projected {
        void project();
    }

    /** The input embedding: append a constant, ℓ2-normalise, linear, scaler, ℓ2-normalise. */
    public static final class Embedding implements Layer, Projected {
        private final float shift;
        private final int width;
        private final SphereLinear linear;

        /**
         * Creates the embedding.
         *
         * @param inputs input values
         * @param width the hypersphere width
         * @param shift the appended constant (3 in the paper)
         * @param random the initialisation randomness
         */
        public Embedding(int inputs, int width, float shift, SplittableRandom random) {
            this.shift = shift;
            this.width = width;
            this.linear = new SphereLinear(inputs + 1, width, random);
        }

        /**
         * @param width the hypersphere width
         * @return a spec for {@code Model.sequential().add(...)}
         */
        public static LayerSpec spec(int width) {
            return new LayerSpec() {
                @Override
                public Shape outputShape(Shape input) {
                    int[] d = input.toArray();
                    d[d.length - 1] = width;
                    return Shape.of(d);
                }

                @Override
                public Layer create(Shape input, SplittableRandom random) {
                    return new Embedding(input.dim(-1), width, 3f, random);
                }
            };
        }

        @Override
        public Var forward(Var input) {
            Tensor c = Tensor.full(Shape.of(input.shape().dim(0), 1), shift).to(input.value().device());
            return l2(linear.forward(l2(Var.concat(1, input, Var.of(c)))));
        }

        @Override
        public int outputSize(int inputSize) {
            return width;
        }

        @Override
        public void project() {
            linear.project();
        }

        @Override
        public List<Var> parameters() {
            return List.of(linear.weight, linear.scaler);
        }

        @Override
        public String toString() {
            return "SimbaV2Embedding(" + width + ")";
        }
    }

    /** The residual block: normalised MLP, LERP into the stream, re-normalise. */
    public static final class Block implements Layer, Projected {
        private final SphereLinear up;
        private final SphereLinear down;
        private final Var alpha;

        /**
         * Creates a block.
         *
         * @param width the width
         * @param random the initialisation randomness
         */
        public Block(int width, SplittableRandom random) {
            up = new SphereLinear(width, 4 * width, random);
            down = new SphereLinear(4 * width, width, random);
            alpha = Var.param(Tensor.full(Shape.of(width), 0.25f));
        }

        /** @return a spec for {@code Model.sequential().add(...)} */
        public static LayerSpec spec() {
            return new LayerSpec() {
                @Override
                public Shape outputShape(Shape input) {
                    return input;
                }

                @Override
                public Layer create(Shape input, SplittableRandom random) {
                    return new Block(input.dim(-1), random);
                }
            };
        }

        @Override
        public Var forward(Var input) {
            Var h = l2(down.forward(up.forward(input).relu()));
            return l2(input.add(h.sub(input).mul(alpha)));
        }

        @Override
        public void project() {
            up.project();
            down.project();
        }

        @Override
        public List<Var> parameters() {
            return List.of(up.weight, up.scaler, down.weight, down.scaler, alpha);
        }

        @Override
        public String toString() {
            return "SimbaV2Block";
        }
    }

    /**
     * SimbaV2 networks: embedding, residual blocks, a linear head.
     *
     * @param width the hypersphere width
     * @param blocks residual blocks
     * @return the factory
     */
    public static RlNetworkFactory network(int width, int blocks) {
        return (inputs, outputs, seed) -> {
            Model.Builder b = Model.sequential().input(inputs).add(Embedding.spec(width));
            for (int i = 0; i < blocks; i++) {
                b.add(Block.spec());
            }
            return b.dense(outputs).seed(seed).build();
        };
    }

    /**
     * Projects every hyperspherical weight of a model back onto the unit sphere.
     *
     * @param model the model
     */
    public static void project(Model model) {
        for (Layer l : model.layers()) {
            if (l instanceof Projected p) {
                p.project();
            }
        }
    }

    /**
     * The mean ℓ2 norm of a model's parameters, for the norm-growth diagnostic.
     *
     * @param model the model
     * @return the mean parameter norm
     */
    public static double meanParameterNorm(Model model) {
        List<Double> norms = new ArrayList<>();
        for (Var p : model.parameters()) {
            norms.add(Math.sqrt(p.value().square().sum().item()));
        }
        return norms.stream().mapToDouble(Double::doubleValue).average().orElse(0);
    }

    /**
     * SimbaV2 on {@link ModernSac}.
     *
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param width the critic width
     * @param replayRatio gradient updates per environment step
     * @return the learner
     */
    public static ModernSac learner(int observationDims, int actionDims, int width, int replayRatio) {
        return ModernSac.of(observationDims, actionDims, network(Math.min(width, 64), 1), network(width, 1))
                .categorical(HlGaussValue.of(-20f, 20f, 41)).returnScaling(true).learningRateDecay(true)
                .afterStep(SimbaV2::project).updateToData(replayRatio);
    }
}
