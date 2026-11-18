package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A GRU over a sequence {@code [batch, steps, inputs] -> [batch, steps, units]} with
 * variational (time-tied) dropout: one dropout mask per sequence is drawn for the inputs and
 * one for the recurrent state, and the same masks are reused at every step. That is the
 * Bayesian reading of dropout for RNNs, and it lets dropout regularise the recurrent
 * connections without destroying the memory the way fresh per-step masks do. Masks can also
 * be fresh at each step (the naive baseline) or switched off.
 *
 * <p>Paper: Gal and Ghahramani, "A Theoretically Grounded Application of Dropout in Recurrent
 * Neural Networks" (NeurIPS 2016), <a href="https://arxiv.org/abs/1512.05287">arXiv:1512.05287</a>.
 */
public final class VariationalGru implements Layer {

    /** How dropout masks are drawn over time. */
    public enum Masks {
        /** one mask per sequence for inputs and state (variational) */
        TIED,
        /** a fresh mask at every step for inputs and state (naive) */
        PER_STEP,
        /** no dropout */
        NONE
    }

    private final GruCell cell;
    private final Masks masks;
    private final float rate;
    private final RandomGenerator random;

    /**
     * Creates the layer.
     *
     * @param inputs the input features
     * @param units the hidden units
     * @param masks how masks are drawn
     * @param rate the drop rate of inputs and state
     * @param random the source of weights and masks
     */
    public VariationalGru(int inputs, int units, Masks masks, float rate, RandomGenerator random) {
        if (!(rate >= 0 && rate < 1)) {
            throw new IllegalArgumentException("rate must be in [0, 1), got " + rate);
        }
        this.cell = new GruCell(inputs, units, random);
        this.masks = masks;
        this.rate = rate;
        this.random = random;
    }

    /**
     * Describes the layer for model builders on {@code [steps, inputs]}.
     *
     * @param units the hidden units
     * @param masks how masks are drawn
     * @param rate the drop rate
     * @return the spec
     */
    public static LayerSpec spec(int units, Masks masks, float rate) {
        return ClassicOps.spec(s -> ClassicOps.withLast(s, units),
                (s, r) -> new VariationalGru(s.dim(s.rank() - 1), units, masks, rate, r));
    }

    private Tensor mask(int batch, int width, Tensor device) {
        return ClassicOps.bernoulli(Shape.of(batch, width), 1 - rate, random, device.device()).div(1 - rate);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(2) != cell.inputs()) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + cell.inputs() + "], got " + s);
        }
        int b = s.dim(0);
        int t = s.dim(1);
        boolean drop = training && rate > 0 && masks != Masks.NONE;
        Tensor inMask = drop ? mask(b, cell.inputs(), input.value()) : null;
        Tensor hMask = drop ? mask(b, cell.units(), input.value()) : null;
        Var h = cell.zeroState(b);
        List<Var> outs = new ArrayList<>(t);
        for (int i = 0; i < t; i++) {
            Var x = input.slice(1, i, i + 1).reshape(b, cell.inputs());
            Var hIn = h;
            if (drop) {
                if (masks == Masks.PER_STEP && i > 0) {
                    inMask = mask(b, cell.inputs(), input.value());
                    hMask = mask(b, cell.units(), input.value());
                }
                x = x.mul(inMask);
                hIn = h.mul(hMask);
            }
            // the dropped state feeds the gates; the carried state stays clean
            h = cell.stepProjected(cell.project(x), hIn, h);
            outs.add(h.reshape(b, 1, cell.units()));
        }
        return Var.concat(1, outs.toArray(Var[]::new));
    }

    @Override
    public List<Var> parameters() {
        return cell.parameters();
    }

    @Override
    public int outputSize(int inputSize) {
        return cell.units();
    }

    /** @return how masks are drawn */
    public Masks masks() {
        return masks;
    }

    @Override
    public String toString() {
        return "VariationalGru(" + cell.inputs() + " -> " + cell.units() + ", " + masks + ", rate " + rate + ")";
    }
}
