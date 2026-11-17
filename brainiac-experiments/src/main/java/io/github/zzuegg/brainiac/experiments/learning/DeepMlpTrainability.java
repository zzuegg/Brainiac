package io.github.zzuegg.brainiac.experiments.learning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.HighwayDense;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 6 item 2.4 (second half): can a very deep (30 hidden layers), narrow MLP be trained on
 * {@link ToyTasks#spiral spirals} with plain SGD with momentum? Compared: a plain stack of
 * dense ReLU layers (which barely trains at this depth), a residual stack
 * ({@code x + 0.1 relu(x W + b)}, the modern answer) and a {@link HighwayDense} stack (the gated
 * ancestor of residual connections).
 */
public final class DeepMlpTrainability {

    /** The compared stacks. */
    public enum Stack {
        /** plain dense ReLU layers */
        PLAIN,
        /** residual dense layers */
        RESIDUAL,
        /** highway layers with carry bias -2 */
        HIGHWAY
    }

    private DeepMlpTrainability() {
    }

    /** {@code x + 0.1 relu(x W + b)}: a residual layer with a down-scaled branch. */
    private static LayerSpec residual() {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                int d = input.dim(input.rank() - 1);
                Dense inner = new Dense(d, d, Activation.RELU, Init.HE, random);
                return new Layer() {
                    @Override
                    public Var forward(Var x) {
                        return x.add(inner.forward(x).mul(0.1f));
                    }

                    @Override
                    public List<Var> parameters() {
                        return inner.parameters();
                    }
                };
            }
        };
    }

    /**
     * Builds the stack.
     *
     * @param stack the kind of stack
     * @param depth the hidden layers
     * @param width their width
     * @param seed the seed
     * @return the model
     */
    public static Model build(Stack stack, int depth, int width, long seed) {
        Model.Builder b = Model.sequential().input(2).seed(seed).dense(width, Activation.RELU);
        for (int i = 0; i < depth; i++) {
            switch (stack) {
                case PLAIN -> b.dense(width, Activation.RELU);
                case RESIDUAL -> b.add(residual());
                case HIGHWAY -> b.add(HighwayDense.spec(Activation.RELU, -2f));
            }
        }
        return b.dense(3, Activation.IDENTITY).build();
    }

    /**
     * Trains a stack on three-arm spirals.
     *
     * @param stack the kind of stack
     * @param depth the hidden layers
     * @param steps full-batch steps
     * @param seed the seed
     * @return {@code {final training loss, training accuracy}}
     */
    public static float[] run(Stack stack, int depth, int steps, long seed) {
        ToyTasks.Data data = ToyTasks.spiral(100, 3, 0.1, seed);
        Model model = build(stack, depth, 24, seed);
        Optimizer opt = Optimizer.momentum(0.05f, 0.9f);
        Tensor x = data.inputs();
        Tensor y = data.ids();
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            Var loss = Functional.crossEntropy(model.forward(Var.of(x), true), y);
            last = loss.value().item();
            loss.backward();
            opt.step(model.parameters());
        }
        float acc = model.forward(Var.of(x)).value().argmax(1).equalTo(y).mean().item();
        return new float[] {Float.isFinite(last) ? last : Float.MAX_VALUE, acc};
    }
}
