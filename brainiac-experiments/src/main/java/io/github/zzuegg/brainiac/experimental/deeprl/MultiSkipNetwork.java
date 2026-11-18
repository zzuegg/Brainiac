package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Multi-skip residual trunks (Castanyer et al., "Stable Gradients for Stable Learning at Scale
 * in Deep Reinforcement Learning", 2025, <a href="https://arxiv.org/abs/2506.15544">arXiv:2506.15544</a>):
 * deep RL agents degrade as networks deepen because gradients become unstable under
 * non-stationary targets; broadcasting the encoder's features directly to every subsequent
 * layer gives each layer a short gradient path to the input. Paired with a Kronecker-factored
 * preconditioner ({@link io.github.zzuegg.brainiac.experimental.train.PsgdKron}) the paper finds
 * that agents keep improving with depth.
 *
 * <p>The trunk embeds the input, {@code e = relu(W_e x)}, then applies {@code depth} layers
 * {@code h <- relu(W_l (h + e))}: the paper does not fix how the broadcast features are merged, so
 * this implementation adds them to each layer's input. With skips off it is a plain deep MLP of
 * the same size, the baseline that degrades with depth. The trunk is one {@link Layer} (with a
 * {@link #spec}); {@link #factory} closes it with a linear head as an {@link RlNetworkFactory}.
 */
public final class MultiSkipNetwork implements Layer {

    private final Dense embed;
    private final List<Dense> layers = new ArrayList<>();
    private final boolean skips;
    private final int width;

    /**
     * Creates a trunk.
     *
     * @param inputs input values
     * @param width the width
     * @param depth hidden layers after the embedding
     * @param skips whether the embedding is broadcast to every layer
     * @param random the initialisation randomness
     */
    public MultiSkipNetwork(int inputs, int width, int depth, boolean skips, SplittableRandom random) {
        this.width = width;
        this.skips = skips;
        this.embed = new Dense(inputs, width, Activation.RELU, Init.HE, random);
        for (int i = 0; i < depth; i++) {
            layers.add(new Dense(width, width, Activation.RELU, Init.HE, random));
        }
    }

    /**
     * @param width the width
     * @param depth hidden layers after the embedding
     * @param skips whether the embedding is broadcast to every layer
     * @return a spec for {@code Model.sequential().add(...)}
     */
    public static LayerSpec spec(int width, int depth, boolean skips) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = width;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new MultiSkipNetwork(input.dim(-1), width, depth, skips, random);
            }
        };
    }

    /**
     * Trunk plus a linear head.
     *
     * @param width the width
     * @param depth hidden layers after the embedding
     * @param skips whether the embedding is broadcast to every layer
     * @return the factory
     */
    public static RlNetworkFactory factory(int width, int depth, boolean skips) {
        return (inputs, outputs, seed) -> Model.sequential().input(inputs).add(spec(width, depth, skips))
                .dense(outputs).seed(seed).build();
    }

    @Override
    public Var forward(Var input) {
        Var e = embed.forward(input);
        Var h = e;
        for (Dense d : layers) {
            h = d.forward(skips ? h.add(e) : h);
        }
        return h;
    }

    @Override
    public int outputSize(int inputSize) {
        return width;
    }

    /**
     * The gradient norm of each layer's weight after a backward pass (embedding first), for the
     * gradient-spread diagnostic.
     *
     * @return the norms (0 where no gradient)
     */
    public double[] gradientNorms() {
        double[] out = new double[layers.size() + 1];
        out[0] = norm(embed.weight());
        for (int i = 0; i < layers.size(); i++) {
            out[i + 1] = norm(layers.get(i).weight());
        }
        return out;
    }

    private static double norm(Var w) {
        return w.grad() == null ? 0 : Math.sqrt(w.grad().square().sum().item());
    }

    /**
     * The gradient spread of a model's multi-skip trunk: the ratio of the largest to the smallest
     * per-layer gradient norm of {@code sum(output)} on a batch, a measure of how unevenly
     * gradients reach the layers.
     *
     * @param model a model containing a multi-skip trunk
     * @param batch the inputs
     * @return the spread (1 is perfectly even), NaN if the model has no trunk
     */
    public static double gradientSpread(Model model, io.github.zzuegg.brainiac.tensor.Tensor batch) {
        for (Layer l : model.layers()) {
            if (l instanceof MultiSkipNetwork m) {
                for (Var p : model.parameters()) {
                    p.zeroGrad();
                }
                model.forward(Var.of(batch), true).sum().backward();
                double[] n = m.gradientNorms();
                double max = 0;
                double min = Double.POSITIVE_INFINITY;
                for (double v : n) {
                    max = Math.max(max, v);
                    min = Math.min(min, v);
                }
                for (Var p : model.parameters()) {
                    p.zeroGrad();
                }
                return max / Math.max(min, 1e-12);
            }
        }
        return Double.NaN;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(embed.parameters());
        for (Dense d : layers) {
            p.addAll(d.parameters());
        }
        return p;
    }

    @Override
    public String toString() {
        return "MultiSkipNetwork(" + width + " x " + layers.size() + (skips ? ", skips" : "") + ")";
    }
}
