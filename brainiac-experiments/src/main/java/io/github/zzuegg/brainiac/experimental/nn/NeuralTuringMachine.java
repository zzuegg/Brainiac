package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A Neural Turing Machine (Graves, Wayne and Danihelka 2014,
 * <a href="https://arxiv.org/abs/1410.5401">arXiv:1410.5401</a>): a controller network with
 * an external memory matrix {@code M [slots, width]} that it reads and writes through
 * differentiable heads. Each head addresses memory in four steps:
 * <ol>
 * <li>content: {@code w_c = softmax(beta * cosine(key, M_i))};</li>
 * <li>interpolation with the previous weighting: {@code w_g = g w_c + (1 - g) w_prev};</li>
 * <li>circular shift by -1, 0 or +1 with a learned distribution {@code s};</li>
 * <li>sharpening: {@code w = w^gamma / sum(w^gamma)}, {@code gamma >= 1}.</li>
 * </ol>
 * The write head erases and adds: {@code M = M . (1 - w e^T) + w a^T}; the read head returns
 * {@code r = w M}. The controller is a feed-forward tanh layer on {@code [x_t, r_(t-1)]} (the
 * feed-forward NTM of the paper), with one read and one write head. Everything is composed of
 * autograd operations, so the machine trains end to end on algorithmic tasks such as copying.
 * Input {@code [batch, steps, inputs]}, output {@code [batch, steps, outputs]}; memory starts
 * blank for every sequence.
 */
public final class NeuralTuringMachine implements Layer {

    private final int inputs;
    private final int outputs;
    private final int slots;
    private final int width;
    private final int hidden;
    private final Var wc;
    private final Var bc;
    private final Var wHeads;
    private final Var bHeads;
    private final Var wy;
    private final Var by;
    private Tensor lastMemory;

    /**
     * Creates the machine.
     *
     * @param inputs the input features per step
     * @param outputs the output features per step
     * @param slots the memory slots {@code N}
     * @param width the memory width {@code W}
     * @param hidden the controller units
     * @param random the random source for the weights
     */
    public NeuralTuringMachine(int inputs, int outputs, int slots, int width, int hidden, SplittableRandom random) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.slots = slots;
        this.width = width;
        this.hidden = hidden;
        int heads = headSize() * 2 + 2 * width;
        wc = Var.param(Init.XAVIER.create(Shape.of(inputs + width, hidden), inputs + width, hidden, random));
        bc = Var.param(Tensor.zeros(Shape.of(hidden)));
        wHeads = Var.param(Init.XAVIER.create(Shape.of(hidden, heads), hidden, heads, random));
        bHeads = Var.param(Tensor.zeros(Shape.of(heads)));
        wy = Var.param(Init.XAVIER.create(Shape.of(hidden + width, outputs), hidden + width, outputs, random));
        by = Var.param(Tensor.zeros(Shape.of(outputs)));
    }

    /**
     * Describes the machine for model builders, on inputs {@code [steps, features]}.
     *
     * @param outputs the output features per step
     * @param slots the memory slots
     * @param width the memory width
     * @param hidden the controller units
     * @return the spec
     */
    public static LayerSpec spec(int outputs, int slots, int width, int hidden) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new NeuralTuringMachine(input.dim(-1), outputs, slots, width, hidden, random);
            }
        };
    }

    /** key, strength, gate, three shifts, sharpening. */
    private int headSize() {
        return width + 6;
    }

    /** @return the memory after the last forward pass, {@code [batch, slots, width]} */
    public Tensor lastMemory() {
        return lastMemory;
    }

    /**
     * NTM addressing for a batch of heads.
     *
     * @param memory {@code [batch, N, W]}
     * @param key {@code [batch, 1, W]}
     * @param strength {@code [batch, 1, 1]}, positive
     * @param gate {@code [batch, 1, 1]}, in [0, 1]
     * @param shift {@code [batch, 1, 3]}, a distribution over shifts -1, 0, +1
     * @param sharpen {@code [batch, 1, 1]}, at least 1
     * @param previous {@code [batch, 1, N]}
     * @return the new weighting {@code [batch, 1, N]}
     */
    static Var address(Var memory, Var key, Var strength, Var gate, Var shift, Var sharpen, Var previous) {
        Var content = contentWeights(memory, key, strength);
        Var wg = content.mul(gate).add(previous.mul(gate.neg().add(1f)));
        int n = memory.shape().dim(1);
        Var next = Var.concat(-1, wg.slice(-1, 1, n), wg.slice(-1, 0, 1));          // w(i + 1)
        Var prev = Var.concat(-1, wg.slice(-1, n - 1, n), wg.slice(-1, 0, n - 1));  // w(i - 1)
        Var shifted = next.mul(shift.slice(-1, 0, 1)).add(wg.mul(shift.slice(-1, 1, 2)))
                .add(prev.mul(shift.slice(-1, 2, 3)));
        Var powered = shifted.add(1e-6f).log().mul(sharpen).exp();
        return powered.div(powered.sum(-1, true));
    }

    /**
     * Content addressing: a softmax over slots of the scaled cosine similarity to a key.
     *
     * @param memory {@code [batch, N, W]}
     * @param key {@code [batch, k, W]}
     * @param strength {@code [batch, k, 1]}
     * @return {@code [batch, k, N]}
     */
    static Var contentWeights(Var memory, Var key, Var strength) {
        int b = memory.shape().dim(0);
        int n = memory.shape().dim(1);
        Var dot = key.matmul(memory.transpose());
        Var kn = key.square().sum(-1, true).add(1e-6f).sqrt();
        Var mn = memory.square().sum(-1, false).add(1e-6f).sqrt().reshape(b, 1, n);
        Var cos = dot.div(kn.mul(mn));
        return Activation.SOFTMAX.forward(cos.mul(strength));
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        int steps = input.shape().dim(1);
        if (input.shape().rank() != 3 || input.shape().dim(2) != inputs) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + inputs + "], got " + input.shape());
        }
        Var memory = Var.of(Tensor.full(Shape.of(b, slots, width), 1e-6f));
        float[] first = new float[b * slots];
        for (int i = 0; i < b; i++) {
            first[i * slots] = 1;
        }
        Var wRead = Var.of(Tensor.of(Shape.of(b, 1, slots), first));
        Var wWrite = wRead;
        Var read = Var.of(Tensor.zeros(Shape.of(b, 1, width)));
        int hs = headSize();
        Var[] outs = new Var[steps];
        for (int t = 0; t < steps; t++) {
            Var x = input.slice(1, t, t + 1);
            Var h = Var.concat(-1, x, read).matmul(wc).add(bc).tanh();
            Var p = h.matmul(wHeads).add(bHeads);
            wRead = head(memory, p, 0, wRead);
            wWrite = head(memory, p, hs, wWrite);
            Var erase = p.slice(-1, 2 * hs, 2 * hs + width).sigmoid();
            Var add = p.slice(-1, 2 * hs + width, 2 * hs + 2 * width).tanh();
            Var wT = wWrite.transpose();
            memory = memory.mul(wT.matmul(erase).neg().add(1f)).add(wT.matmul(add));
            read = wRead.matmul(memory);
            outs[t] = Var.concat(-1, h, read).matmul(wy).add(by);
        }
        lastMemory = memory.value();
        return Var.concat(1, outs);
    }

    private Var head(Var memory, Var p, int o, Var previous) {
        Var key = p.slice(-1, o, o + width).tanh();
        Var strength = p.slice(-1, o + width, o + width + 1).softplus();
        Var gate = p.slice(-1, o + width + 1, o + width + 2).sigmoid();
        Var shift = Activation.SOFTMAX.forward(p.slice(-1, o + width + 2, o + width + 5));
        Var sharpen = p.slice(-1, o + width + 5, o + width + 6).softplus().add(1f);
        return address(memory, key, strength, gate, shift, sharpen, previous);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wc, bc, wHeads, bHeads, wy, by);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), outputs);
    }

    @Override
    public String toString() {
        return "NeuralTuringMachine(" + slots + "x" + width + ", " + hidden + " controller units)";
    }
}
