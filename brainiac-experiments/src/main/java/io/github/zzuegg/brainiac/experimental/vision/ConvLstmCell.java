package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * One step of a convolutional LSTM (Shi et al., 2015): the LSTM's matrix products become
 * convolutions over the concatenated input and hidden maps, so the state is a feature map
 * and the cell sees local spatial context:
 * {@code [i, f, o, g] = conv([x, h])}, {@code c' = sigmoid(f) c + sigmoid(i) tanh(g)},
 * {@code h' = sigmoid(o) tanh(c')}. Driving it frame by frame (and feeding its own
 * predictions back) gives a recurrent video predictor.
 *
 * <p>Paper: Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for
 * Precipitation Nowcasting" (2015), <a href="https://arxiv.org/abs/1506.04214">arXiv:1506.04214</a>.
 */
public final class ConvLstmCell {

    private final int inputs;
    private final int hidden;
    private final Var weight;
    private final Var bias;

    /**
     * Creates a cell.
     *
     * @param inputs the input channels
     * @param hidden the state channels
     * @param kernel the square kernel side, odd
     * @param random the source of initial weights
     */
    public ConvLstmCell(int inputs, int hidden, int kernel, RandomGenerator random) {
        this.inputs = inputs;
        this.hidden = hidden;
        this.weight = Var.param(Tensor.normal(Shape.of(4 * hidden, inputs + hidden, kernel, kernel), 0,
                (float) Math.sqrt(1.0 / ((inputs + hidden) * kernel * kernel)), random));
        float[] b = new float[4 * hidden];
        for (int i = hidden; i < 2 * hidden; i++) {
            b[i] = 1f;
        }
        this.bias = Var.param(Tensor.of(Shape.of(4 * hidden), b));
    }

    /**
     * A zero state.
     *
     * @param batch the batch size
     * @param height the map height
     * @param width the map width
     * @return {@code {h, c}}, each {@code [batch, hidden, height, width]}
     */
    public Var[] zeroState(int batch, int height, int width) {
        Tensor z = Tensor.zeros(Shape.of(batch, hidden, height, width));
        return new Var[] {Var.of(z), Var.of(z)};
    }

    /**
     * Advances the state by one frame.
     *
     * @param x the input {@code [batch, inputs, h, w]}
     * @param state {@code {h, c}}
     * @return the new {@code {h, c}}
     */
    public Var[] step(Var x, Var[] state) {
        Var gates = VisionOps.conv(Var.concat(1, x, state[0]), weight, bias);
        int k = hidden;
        Var i = gates.slice(1, 0, k).sigmoid();
        Var f = gates.slice(1, k, 2 * k).sigmoid();
        Var o = gates.slice(1, 2 * k, 3 * k).sigmoid();
        Var g = gates.slice(1, 3 * k, 4 * k).tanh();
        Var c = f.mul(state[1]).add(i.mul(g));
        return new Var[] {o.mul(c.tanh()), c};
    }

    /** @return the input channels */
    public int inputs() {
        return inputs;
    }

    /** @return the state channels */
    public int hidden() {
        return hidden;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public String toString() {
        return "ConvLstmCell(" + inputs + " -> " + hidden + ")";
    }
}
