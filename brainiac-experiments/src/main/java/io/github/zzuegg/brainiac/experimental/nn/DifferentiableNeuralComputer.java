package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A Differentiable Neural Computer (Graves et al. 2016, Nature 538,
 * <a href="https://www.nature.com/articles/nature20101">doi:10.1038/nature20101</a>): a
 * Neural Turing Machine whose memory adds
 * <ul>
 * <li><b>dynamic allocation</b>: a usage vector {@code u} (raised by writes, lowered by
 *     free gates after reads) ranks slots, and the allocation weighting
 *     {@code a[phi_j] = (1 - u[phi_j]) prod_(i<j) u[phi_i]} over the slots sorted by usage
 *     points writes at free memory, so stored items never overwrite each other;</li>
 * <li><b>temporal links</b>: a link matrix {@code L[i, j]} records that slot {@code i} was
 *     written right after slot {@code j}, so read heads can step forward
 *     ({@code w L^T}) or backward ({@code w L}) through the order things were written,
 *     besides looking them up by content.</li>
 * </ul>
 * The memory is usable on its own: {@link #initialState} and {@link #step} take an explicit
 * {@link Interface}, so any controller can drive it; as a {@link Layer} a feed-forward tanh
 * controller produces the interface from {@code [x_t, reads_(t-1)]}. Everything except the
 * sorting order of the allocation (as in the paper) is differentiable. Input
 * {@code [batch, steps, inputs]}, output {@code [batch, steps, outputs]}.
 */
public final class DifferentiableNeuralComputer implements Layer {

    /**
     * The memory state between steps.
     *
     * @param memory {@code [batch, N, W]}
     * @param usage {@code [batch, 1, N]}
     * @param precedence {@code [batch, 1, N]}: how much each slot was the last one written
     * @param links {@code [batch, N, N]}
     * @param readWeights {@code [batch, R, N]}
     * @param writeWeights {@code [batch, 1, N]}
     * @param reads {@code [batch, R, W]}: what the read heads returned
     */
    public record State(Var memory, Var usage, Var precedence, Var links, Var readWeights, Var writeWeights,
            Var reads) {
    }

    /**
     * What the controller emits each step, already squashed into range.
     *
     * @param readKeys {@code [batch, R, W]}
     * @param readStrengths {@code [batch, R, 1]}, at least 1
     * @param writeKey {@code [batch, 1, W]}
     * @param writeStrength {@code [batch, 1, 1]}, at least 1
     * @param erase {@code [batch, 1, W]}, in [0, 1]
     * @param write {@code [batch, 1, W]}
     * @param freeGates {@code [batch, R, 1]}, in [0, 1]
     * @param allocationGate {@code [batch, 1, 1]}, in [0, 1]: allocation versus content writing
     * @param writeGate {@code [batch, 1, 1]}, in [0, 1]
     * @param readModes {@code [batch, R, 3]}: backward, content and forward shares, summing to 1
     */
    public record Interface(Var readKeys, Var readStrengths, Var writeKey, Var writeStrength, Var erase, Var write,
            Var freeGates, Var allocationGate, Var writeGate, Var readModes) {
    }

    private final int inputs;
    private final int outputs;
    private final int slots;
    private final int width;
    private final int readHeads;
    private final Var wc;
    private final Var bc;
    private final Var wi;
    private final Var bi;
    private final Var wy;
    private final Var by;
    private State lastState;

    /**
     * Creates the computer.
     *
     * @param inputs the input features per step
     * @param outputs the output features per step
     * @param slots the memory slots {@code N}
     * @param width the memory width {@code W}
     * @param readHeads the read heads {@code R}
     * @param hidden the controller units
     * @param random the random source for the weights
     */
    public DifferentiableNeuralComputer(int inputs, int outputs, int slots, int width, int readHeads, int hidden,
            SplittableRandom random) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.slots = slots;
        this.width = width;
        this.readHeads = readHeads;
        int in = inputs + readHeads * width;
        int iface = readHeads * (width + 5) + 3 * width + 3;
        wc = Var.param(Init.XAVIER.create(Shape.of(in, hidden), in, hidden, random));
        bc = Var.param(Tensor.zeros(Shape.of(hidden)));
        wi = Var.param(Init.XAVIER.create(Shape.of(hidden, iface), hidden, iface, random));
        bi = Var.param(Tensor.zeros(Shape.of(iface)));
        int out = hidden + readHeads * width;
        wy = Var.param(Init.XAVIER.create(Shape.of(out, outputs), out, outputs, random));
        by = Var.param(Tensor.zeros(Shape.of(outputs)));
    }

    /**
     * Describes the computer for model builders, on inputs {@code [steps, features]}.
     *
     * @param outputs the output features per step
     * @param slots the memory slots
     * @param width the memory width
     * @param readHeads the read heads
     * @param hidden the controller units
     * @return the spec
     */
    public static LayerSpec spec(int outputs, int slots, int width, int readHeads, int hidden) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new DifferentiableNeuralComputer(input.dim(-1), outputs, slots, width, readHeads, hidden,
                        random);
            }
        };
    }

    /**
     * A blank memory.
     *
     * @param batch the sequences side by side
     * @param slots the memory slots
     * @param width the memory width
     * @param readHeads the read heads
     * @return the state
     */
    public static State initialState(int batch, int slots, int width, int readHeads) {
        return new State(Var.of(Tensor.zeros(Shape.of(batch, slots, width))), Var.of(Tensor.zeros(Shape.of(batch, 1,
                slots))), Var.of(Tensor.zeros(Shape.of(batch, 1, slots))), Var.of(Tensor.zeros(Shape.of(batch, slots,
                slots))), Var.of(Tensor.zeros(Shape.of(batch, readHeads, slots))), Var.of(Tensor.zeros(Shape.of(batch,
                1, slots))), Var.of(Tensor.zeros(Shape.of(batch, readHeads, width))));
    }

    /**
     * The allocation weighting: free slots, least used first.
     *
     * @param usage {@code [batch, 1, N]} in [0, 1]
     * @return {@code [batch, 1, N]}
     */
    static Var allocation(Var usage) {
        int b = usage.shape().dim(0);
        int n = usage.shape().dim(2);
        float[] u = usage.value().toArray();
        float[] perm = new float[b * n * n];
        for (int i = 0; i < b; i++) {
            final int row = i;
            Integer[] order = new Integer[n];
            for (int j = 0; j < n; j++) {
                order[j] = j;
            }
            Arrays.sort(order, (x, y) -> Float.compare(u[row * n + x], u[row * n + y]));
            for (int j = 0; j < n; j++) {
                perm[(row * n + j) * n + order[j]] = 1;   // P[j, phi_j] = 1
            }
        }
        Tensor p = Tensor.of(Shape.of(b, n, n), perm);
        float[] upper = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                upper[i * n + j] = 1;
            }
        }
        Var sorted = usage.matmul(Var.of(p.transpose()));
        Var exclusive = sorted.add(1e-6f).log().matmul(Var.of(Tensor.of(Shape.of(n, n), upper))).exp();
        Var allocSorted = sorted.neg().add(1f).mul(exclusive);
        return allocSorted.matmul(Var.of(p));
    }

    /**
     * One memory step: frees, allocates, writes, updates temporal links and reads.
     *
     * @param s the state before
     * @param in the controller's interface
     * @return the state after
     */
    public static State step(State s, Interface in) {
        int n = s.memory().shape().dim(1);
        int heads = s.readWeights().shape().dim(1);
        // retention: prod over read heads of (1 - f_i w_i)
        Var keep = in.freeGates().mul(s.readWeights()).neg().add(1f);
        Var retention = keep.slice(1, 0, 1);
        for (int i = 1; i < heads; i++) {
            retention = retention.mul(keep.slice(1, i, i + 1));
        }
        Var ww0 = s.writeWeights();
        Var usage = s.usage().add(ww0).sub(s.usage().mul(ww0)).mul(retention);
        Var alloc = allocation(usage);
        Var content = NeuralTuringMachine.contentWeights(s.memory(), in.writeKey(), in.writeStrength());
        Var ga = in.allocationGate();
        Var ww = alloc.mul(ga).add(content.mul(ga.neg().add(1f))).mul(in.writeGate());
        Var wwT = ww.transpose();
        Var memory = s.memory().mul(wwT.matmul(in.erase()).neg().add(1f)).add(wwT.matmul(in.write()));
        float[] off = new float[n * n];
        Arrays.fill(off, 1);
        for (int i = 0; i < n; i++) {
            off[i * n + i] = 0;
        }
        Var links = s.links().mul(wwT.neg().sub(ww).add(1f)).add(wwT.matmul(s.precedence()))
                .mul(Tensor.of(Shape.of(n, n), off));
        Var precedence = s.precedence().mul(ww.sum(-1, true).neg().add(1f)).add(ww);
        Var forward = s.readWeights().matmul(links.transpose());
        Var backward = s.readWeights().matmul(links);
        Var lookup = NeuralTuringMachine.contentWeights(memory, in.readKeys(), in.readStrengths());
        Var modes = in.readModes();
        Var wr = backward.mul(modes.slice(-1, 0, 1)).add(lookup.mul(modes.slice(-1, 1, 2)))
                .add(forward.mul(modes.slice(-1, 2, 3)));
        Var reads = wr.matmul(memory);
        return new State(memory, usage, precedence, links, wr, ww, reads);
    }

    private Interface parse(Var v, int b) {
        int r = readHeads;
        int w = width;
        int o = 0;
        Var readKeys = v.slice(-1, o, o + r * w).reshape(b, r, w);
        o += r * w;
        Var readStrengths = oneplus(v.slice(-1, o, o + r)).reshape(b, r, 1);
        o += r;
        Var writeKey = v.slice(-1, o, o + w);
        o += w;
        Var writeStrength = oneplus(v.slice(-1, o, o + 1));
        o += 1;
        Var erase = v.slice(-1, o, o + w).sigmoid();
        o += w;
        Var write = v.slice(-1, o, o + w);
        o += w;
        Var free = v.slice(-1, o, o + r).sigmoid().reshape(b, r, 1);
        o += r;
        Var allocGate = v.slice(-1, o, o + 1).sigmoid();
        o += 1;
        Var writeGate = v.slice(-1, o, o + 1).sigmoid();
        o += 1;
        Var modes = Activation.SOFTMAX.forward(v.slice(-1, o, o + 3 * r).reshape(b, r, 3));
        return new Interface(readKeys, readStrengths, writeKey, writeStrength, erase, write, free, allocGate, writeGate,
                modes);
    }

    private static Var oneplus(Var x) {
        return x.softplus().add(1f);
    }

    /** @return the memory state after the last forward pass */
    public State lastState() {
        return lastState;
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        int steps = input.shape().dim(1);
        if (input.shape().rank() != 3 || input.shape().dim(2) != inputs) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + inputs + "], got " + input.shape());
        }
        State s = initialState(b, slots, width, readHeads);
        Var[] outs = new Var[steps];
        for (int t = 0; t < steps; t++) {
            Var x = input.slice(1, t, t + 1);
            Var flat = s.reads().reshape(b, 1, readHeads * width);
            Var h = Var.concat(-1, x, flat).matmul(wc).add(bc).tanh();
            s = step(s, parse(h.matmul(wi).add(bi), b));
            outs[t] = Var.concat(-1, h, s.reads().reshape(b, 1, readHeads * width)).matmul(wy).add(by);
        }
        lastState = s;
        return Var.concat(1, outs);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wc, bc, wi, bi, wy, by);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), outputs);
    }

    @Override
    public String toString() {
        return "DifferentiableNeuralComputer(" + slots + "x" + width + ", " + readHeads + " read heads)";
    }
}
