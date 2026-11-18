package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * One step of a gated recurrent unit (Cho et al., 2014,
 * <a href="https://arxiv.org/abs/1406.1078">arXiv:1406.1078</a>) with the state passed in and
 * out explicitly, for recurrent RL agents that store, replay and reset their state
 * themselves: R2D2's recurrent replay ({@link RecurrentReplayDqn}), RL² and VariBAD
 * ({@link MetaRlAgent}) and the deterministic path of PlaNet's state-space model
 * ({@link RecurrentStateSpaceModel}). The core {@code Recurrent} layer runs whole sequences
 * and keeps its state hidden, which these agents cannot use.
 *
 * <pre>
 * z = sigmoid(x Wz + h Uz + bz), r = sigmoid(x Wr + h Ur + br)
 * n = tanh(x Wn + bn + r * (h Un)), h' = (1 - z) * n + z * h
 * </pre>
 */
public final class RlGruCell {

    private final int inputs;
    private final int units;
    private final Var wx;
    private final Var wh;
    private final Var bias;

    /**
     * Creates a cell with Glorot-uniform weights and zero biases (update gate biased to 1,
     * so the state is kept by default).
     *
     * @param inputs values per input row
     * @param units the state size
     * @param seed the initialization seed
     */
    public RlGruCell(int inputs, int units, long seed) {
        this.inputs = inputs;
        this.units = units;
        SplittableRandom random = new SplittableRandom(seed);
        this.wx = Var.param(uniform(inputs, 3 * units, random));
        this.wh = Var.param(uniform(units, 3 * units, random));
        float[] b = new float[3 * units];
        for (int i = 0; i < units; i++) {
            b[i] = 1f;
        }
        this.bias = Var.param(Tensor.of(Shape.of(1, 3 * units), b));
    }

    private static Tensor uniform(int rows, int cols, SplittableRandom random) {
        double limit = Math.sqrt(6.0 / (rows + cols / 3.0));
        float[] w = new float[rows * cols];
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) random.nextDouble(-limit, limit);
        }
        return Tensor.of(Shape.of(rows, cols), w);
    }

    /**
     * Advances the state by one input.
     *
     * @param x the inputs, {@code [batch, inputs]}
     * @param h the state, {@code [batch, units]}
     * @return the next state
     */
    public Var step(Var x, Var h) {
        Var gx = x.matmul(wx).add(bias);
        Var gh = h.matmul(wh);
        Var z = gx.slice(1, 0, units).add(gh.slice(1, 0, units)).sigmoid();
        Var r = gx.slice(1, units, 2 * units).add(gh.slice(1, units, 2 * units)).sigmoid();
        Var n = gx.slice(1, 2 * units, 3 * units).add(r.mul(gh.slice(1, 2 * units, 3 * units))).tanh();
        return z.neg().add(1f).mul(n).add(z.mul(h));
    }

    /**
     * @param batch the number of rows
     * @return a zero state
     */
    public Var zeroState(int batch) {
        return Var.of(Tensor.zeros(Shape.of(batch, units)));
    }

    /** @return the input size */
    public int inputs() {
        return inputs;
    }

    /** @return the state size */
    public int units() {
        return units;
    }

    /** @return the weights and biases */
    public List<Var> parameters() {
        return List.of(wx, wh, bias);
    }
}
