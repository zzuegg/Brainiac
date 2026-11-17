package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A layer of a differentiable logic gate network (Petersen et al.,
 * <a href="https://arxiv.org/abs/2210.08277">2210.08277</a> and
 * <a href="https://arxiv.org/abs/2411.04732">2411.04732</a>). Every neuron reads two fixed,
 * randomly wired inputs {@code a, b} and is one of the 16 two-input boolean functions;
 * which one is learned. During training each gate is a softmax mixture of the 16
 * real-valued relaxations ({@code AND = ab}, {@code OR = a + b - ab}, {@code XOR = a + b - 2ab}, ...),
 * so gradients flow. Every relaxation is a combination of {@code 1, a, b, ab}, so the
 * mixture is too: the layer just computes four coefficients per neuron.
 *
 * <p>After training, {@link #gates()} takes the most likely gate per neuron and
 * {@link LogicCircuit} runs the network as pure bit operations, 64 samples per {@code long}.
 * Inputs and outputs are in [0, 1] (exactly {0, 1} after discretization).
 */
public final class LogicGateLayer implements Layer {
    /** Gate names in the order of their ids 0..15. */
    public static final List<String> NAMES = List.of("FALSE", "AND", "A_AND_NOT_B", "A", "NOT_A_AND_B", "B", "XOR",
            "OR", "NOR", "XNOR", "NOT_B", "A_OR_NOT_B", "NOT_A", "NOT_A_OR_B", "NAND", "TRUE");
    /** Coefficients of {@code (1, a, b, ab)} for each gate's relaxation. */
    static final float[][] COEFFICIENTS = {
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, -1}, {0, 1, 0, 0},
        {0, 0, 1, -1}, {0, 0, 1, 0}, {0, 1, 1, -2}, {0, 1, 1, -1},
        {1, -1, -1, 1}, {1, -1, -1, 2}, {1, 0, -1, 0}, {1, 0, -1, 1},
        {1, -1, 0, 0}, {1, -1, 0, 1}, {1, 0, 0, -1}, {1, 0, 0, 0}};
    private final int inputs;
    private final int units;
    private final Tensor left;
    private final Tensor right;
    private final int[] leftIndex;
    private final int[] rightIndex;
    private final Var logits;   // [units, 16]

    private LogicGateLayer(int inputs, int units, SplittableRandom random) {
        this.inputs = inputs;
        this.units = units;
        leftIndex = new int[units];
        rightIndex = new int[units];
        float[] l = new float[units];
        float[] r = new float[units];
        for (int i = 0; i < units; i++) {
            // spread wiring evenly so every input is used, as in the paper
            leftIndex[i] = i % inputs;
            int other = random.nextInt(inputs - 1);
            rightIndex[i] = other >= leftIndex[i] ? other + 1 : other;
            l[i] = leftIndex[i];
            r[i] = rightIndex[i];
        }
        shuffle(leftIndex, rightIndex, l, r, random);
        left = Tensor.of(Shape.of(units), l);
        right = Tensor.of(Shape.of(units), r);
        float[] init = new float[units * 16];
        for (int i = 0; i < init.length; i++) {
            init[i] = (float) random.nextGaussian();
        }
        logits = Var.param(Tensor.of(Shape.of(units, 16), init));
    }

    private static void shuffle(int[] a, int[] b, float[] fa, float[] fb, SplittableRandom random) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
            t = b[i];
            b[i] = b[j];
            b[j] = t;
            fa[i] = a[i];
            fa[j] = a[j];
            fb[i] = b[i];
            fb[j] = b[j];
        }
    }

    /**
     * Creates a layer.
     *
     * @param inputs the input width, at least 2
     * @param units the number of gates
     * @param seed the seed for wiring and logits
     * @return the layer
     */
    public static LogicGateLayer of(int inputs, int units, long seed) {
        if (inputs < 2 || units < 1) {
            throw new IllegalArgumentException("a logic layer needs at least 2 inputs and 1 gate");
        }
        return new LogicGateLayer(inputs, units, new SplittableRandom(seed));
    }

    /**
     * Describes the layer for model builders.
     *
     * @param units the number of gates
     * @return the spec
     */
    public static LayerSpec spec(int units) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(units);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new LogicGateLayer(input.dim(-1), units, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().rank() != 2 || input.shape().dim(1) != inputs) {
            throw new IllegalArgumentException("LogicGateLayer expects [batch, " + inputs + "], got " + input.shape());
        }
        Var a = input.gather(1, left.to(input.value().device()));
        Var b = input.gather(1, right.to(input.value().device()));
        Var c = Activation.SOFTMAX.forward(logits).matmul(Var.of(coefficients().to(input.value().device())));
        Var c0 = c.slice(1, 0, 1).reshape(units);
        Var c1 = c.slice(1, 1, 2).reshape(units);
        Var c2 = c.slice(1, 2, 3).reshape(units);
        Var c3 = c.slice(1, 3, 4).reshape(units);
        return a.mul(c1).add(b.mul(c2)).add(a.mul(b).mul(c3)).add(c0);
    }

    static Tensor coefficients() {
        float[] f = new float[64];
        for (int g = 0; g < 16; g++) {
            System.arraycopy(COEFFICIENTS[g], 0, f, g * 4, 4);
        }
        return Tensor.of(Shape.of(16, 4), f);
    }

    @Override
    public List<Var> parameters() {
        return List.of(logits);
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    /** @return the most likely gate id per neuron */
    public int[] gates() {
        int[] g = new int[units];
        float[] l = logits.value().toArray();
        for (int i = 0; i < units; i++) {
            int best = 0;
            for (int k = 1; k < 16; k++) {
                if (l[i * 16 + k] > l[i * 16 + best]) {
                    best = k;
                }
            }
            g[i] = best;
        }
        return g;
    }

    /** @return the left input index of each gate */
    public int[] leftInputs() {
        return leftIndex.clone();
    }

    /** @return the right input index of each gate */
    public int[] rightInputs() {
        return rightIndex.clone();
    }

    /** @return the input width */
    public int inputs() {
        return inputs;
    }

    /**
     * Replaces the soft gates by their hard choices, so the layer computes exactly the
     * boolean functions on {0, 1} inputs (for checking the compiled circuit).
     */
    public void harden() {
        int[] g = gates();
        float[] l = new float[units * 16];
        java.util.Arrays.fill(l, -1e4f);
        for (int i = 0; i < units; i++) {
            l[i * 16 + g[i]] = 1e4f;
        }
        logits.assign(Tensor.of(Shape.of(units, 16), l).to(logits.value().device()));
    }
}
