package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * One step of a gated recurrent unit, for blocks that drive the recurrence themselves (seq2seq
 * decoders, pointer networks, slot attention, variational RNN dropout):
 * {@code r = sigmoid(x Wr + h Ur + br)}, {@code z = sigmoid(x Wz + h Uz + bz)},
 * {@code n = tanh(x Wn + bn + r * (h Un))}, {@code h' = (1 - z) n + z h}. Core
 * {@code Recurrent.gru} runs a whole sequence at once; this cell exposes the single step.
 *
 * <p>Paper: Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014),
 * <a href="https://arxiv.org/abs/1406.1078">arXiv:1406.1078</a>.
 */
public final class GruCell {

    private final int inputs;
    private final int units;
    private final Var inputWeight;
    private final Var hiddenWeight;
    private final Var bias;

    /**
     * Creates a cell.
     *
     * @param inputs the input features
     * @param units the hidden units
     * @param random the source of initial weights
     */
    public GruCell(int inputs, int units, RandomGenerator random) {
        this.inputs = inputs;
        this.units = units;
        this.inputWeight = ClassicOps.matrix(random, inputs, 3 * units);
        float a = (float) Math.sqrt(3.0 / units);
        this.hiddenWeight = Var.param(Tensor.uniform(Shape.of(units, 3 * units), -a, a, random));
        this.bias = ClassicOps.zeros(3 * units);
    }

    /**
     * Advances the state by one step.
     *
     * @param x the input, {@code [batch, inputs]}
     * @param h the state, {@code [batch, units]}
     * @return the new state
     */
    public Var step(Var x, Var h) {
        return stepProjected(x.matmul(inputWeight).add(bias), h);
    }

    /**
     * Advances the state from inputs already projected by {@link #project}, so a whole
     * sequence's input projection can be done in one matrix product.
     *
     * @param projected {@code [batch, 3 * units]}, including the bias
     * @param h the state
     * @return the new state
     */
    public Var stepProjected(Var projected, Var h) {
        return stepProjected(projected, h, h);
    }

    /**
     * Advances the state with a separate state for the gate inputs, such as a state with a
     * dropout mask applied: the gates and candidate read {@code gateState}, while the carried
     * part {@code z * h} uses the clean {@code h}.
     *
     * @param projected {@code [batch, 3 * units]}, including the bias
     * @param gateState the state the recurrent weights see
     * @param h the state that is carried over
     * @return the new state
     */
    public Var stepProjected(Var projected, Var gateState, Var h) {
        Var gh = gateState.matmul(hiddenWeight);
        int u = units;
        Var r = projected.slice(1, 0, u).add(gh.slice(1, 0, u)).sigmoid();
        Var z = projected.slice(1, u, 2 * u).add(gh.slice(1, u, 2 * u)).sigmoid();
        Var n = projected.slice(1, 2 * u, 3 * u).add(r.mul(gh.slice(1, 2 * u, 3 * u))).tanh();
        return n.sub(n.mul(z)).add(z.mul(h));
    }

    /**
     * Projects inputs of any leading shape: {@code x Wx + b}.
     *
     * @param x {@code [..., inputs]}
     * @return {@code [..., 3 * units]}
     */
    public Var project(Var x) {
        return x.matmul(inputWeight).add(bias);
    }

    /**
     * A zero state.
     *
     * @param batch the batch size
     * @return {@code [batch, units]} zeros
     */
    public Var zeroState(int batch) {
        return Var.of(Tensor.zeros(Shape.of(batch, units)));
    }

    /** @return the input features */
    public int inputs() {
        return inputs;
    }

    /** @return the hidden units */
    public int units() {
        return units;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        return List.of(inputWeight, hiddenWeight, bias);
    }

    @Override
    public String toString() {
        return "GruCell(" + inputs + " -> " + units + ")";
    }
}
