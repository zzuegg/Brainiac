package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * The Neural GPU: a weight-tied stack of convolutional GRUs applied to a state as wide as
 * the input, once per input position. Each step runs every CGRU
 * {@code u = sigmoid(U * s + b_u)}, {@code r = sigmoid(R * s + b_r)},
 * {@code s = u o s + (1 - u) o tanh(W * (r o s) + b)}, where {@code *} is a width-3
 * convolution along the input positions. Because the same cell runs everywhere and for as
 * many steps as the input is long, information can travel across the whole input, and a
 * rule learned on short inputs is a cellular automaton that runs unchanged on long ones,
 * which is how it learns binary addition and multiplication that generalise in length.
 *
 * <p>Input {@code [batch, width, features]} is embedded to the state size, the steps run,
 * and the final state {@code [batch, width, state]} is returned for a per-position head.
 * The paper's state also has a small height dimension (4); here it is folded into the
 * channels. Its training tricks (curriculum on length, gradient noise, a relaxation of
 * parameter sharing) are left to the caller; {@link #dropout} gives state dropout.
 *
 * <p>Paper: Kaiser and Sutskever, "Neural GPUs Learn Algorithms" (ICLR 2016),
 * <a href="https://arxiv.org/abs/1511.08228">arXiv:1511.08228</a>.
 */
public final class NeuralGpu implements Layer {

    private final int state;
    private final Var embed;
    private final List<Var[]> cells = new ArrayList<>();   // {U, bu, R, br, W, b}
    private final SplittableRandom noise;
    private float dropout;
    private int steps = -1;

    /**
     * Creates the layer.
     *
     * @param features input features per position
     * @param state state channels per position
     * @param layers CGRUs per step, such as 2
     * @param random the random source
     */
    public NeuralGpu(int features, int state, int layers, RandomGenerator random) {
        this.state = state;
        this.embed = MixerOps.weight(random, features, state);
        for (int l = 0; l < layers; l++) {
            cells.add(new Var[] {MixerOps.weight(random, 3 * state, state), MixerOps.filled(0.7f, state),
                MixerOps.weight(random, 3 * state, state), MixerOps.filled(0.5f, state),
                MixerOps.weight(random, 3 * state, state), MixerOps.filled(0f, state)});
        }
        this.noise = new SplittableRandom(random.nextLong());
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param state state channels per position
     * @param layers CGRUs per step
     * @return the spec
     */
    public static LayerSpec of(int state, int layers) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = state;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new NeuralGpu(input.dim(-1), state, layers, random);
            }
        };
    }

    /**
     * Sets the number of steps; by default it equals the input width.
     *
     * @param n the steps, or -1 for the input width
     * @return this layer
     */
    public NeuralGpu steps(int n) {
        this.steps = n;
        return this;
    }

    /**
     * Sets the state dropout used while training (the paper uses a small one).
     *
     * @param p the drop probability
     * @return this layer
     */
    public NeuralGpu dropout(float p) {
        this.dropout = p;
        return this;
    }

    /** A width-3 convolution along positions: [s_{i-1}, s_i, s_{i+1}] W, zero-padded. */
    private static Var conv(Var s, Var w) {
        int b = s.shape().dim(0);
        int width = s.shape().dim(1);
        int c = s.shape().dim(2);
        Var zero = Var.of(Tensor.zeros(Shape.of(b, 1, c)));
        Var left = width == 1 ? zero : Var.concat(1, zero, s.slice(1, 0, width - 1));
        Var right = width == 1 ? zero : Var.concat(1, s.slice(1, 1, width), zero);
        return Var.concat(2, left, s, right).matmul(w);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Var s = input.matmul(embed);
        int n = steps > 0 ? steps : input.shape().dim(1);
        for (int t = 0; t < n; t++) {
            for (Var[] c : cells) {
                Var u = conv(s, c[0]).add(c[1]).sigmoid();
                Var r = conv(s, c[2]).add(c[3]).sigmoid();
                Var cand = conv(r.mul(s), c[4]).add(c[5]).tanh();
                if (training && dropout > 0) {
                    cand = cand.mul(dropMask(cand.shape()));
                }
                s = u.mul(s).add(u.neg().add(1f).mul(cand));
            }
        }
        return s;
    }

    private Tensor dropMask(Shape shape) {
        float[] m = new float[(int) shape.size()];
        for (int i = 0; i < m.length; i++) {
            m[i] = noise.nextDouble() < dropout ? 0 : 1f / (1 - dropout);
        }
        return Tensor.of(shape, m);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed));
        for (Var[] c : cells) {
            p.addAll(List.of(c));
        }
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return state;
    }

    @Override
    public String toString() {
        return "NeuralGpu(state " + state + ", " + cells.size() + " CGRUs)";
    }
}
