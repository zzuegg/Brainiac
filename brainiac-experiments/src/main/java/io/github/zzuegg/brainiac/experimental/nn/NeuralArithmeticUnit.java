package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Neural arithmetic units, layers biased towards exact arithmetic so they extrapolate far
 * outside their training range:
 *
 * <ul>
 *   <li><b>NAC</b> (neural accumulator): {@code a = x W} with {@code W = tanh(W') * sigmoid(M')},
 *       which saturates to weights in {-1, 0, 1}, so the layer learns exact sums and
 *       differences of its inputs with no scaling drift;</li>
 *   <li><b>NALU</b> (neural arithmetic logic unit): a NAC in linear space and one in log space,
 *       {@code m = exp(log(|x| + eps) W)} (products, quotients, powers), mixed by a learned gate
 *       {@code g = sigmoid(x G)}: {@code y = g a + (1 - g) m}.</li>
 * </ul>
 *
 * <p>An MLP fit on {@code [0, 10]} returns roughly constant or linear garbage on
 * {@code [10, 1000]}; a NALU that found {@code a + b} or {@code a * b} keeps it exactly. A
 * drop-in for core {@code Dense}.
 *
 * <p>Paper: Trask et al., "Neural Arithmetic Logic Units" (NeurIPS 2018),
 * <a href="https://arxiv.org/abs/1808.00508">arXiv:1808.00508</a>.
 */
public final class NeuralArithmeticUnit implements Layer {

    /** The unit. */
    public enum Kind {
        /** additive accumulator only */
        NAC,
        /** gated additive and multiplicative paths */
        NALU
    }

    private final int inputs;
    private final int outputs;
    private final Kind kind;
    private final Var hatW;
    private final Var hatM;
    private final Var gate;

    /**
     * Creates the layer.
     *
     * @param inputs the input features
     * @param outputs the output features
     * @param kind the unit
     * @param random the source of initial weights
     */
    public NeuralArithmeticUnit(int inputs, int outputs, Kind kind, RandomGenerator random) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.kind = kind;
        this.hatW = ClassicOps.matrix(random, inputs, outputs);
        this.hatM = ClassicOps.matrix(random, inputs, outputs);
        this.gate = kind == Kind.NALU ? ClassicOps.matrix(random, inputs, outputs) : null;
    }

    /**
     * Describes the layer for model builders.
     *
     * @param outputs the output features
     * @param kind the unit
     * @return the spec
     */
    public static LayerSpec spec(int outputs, Kind kind) {
        return ClassicOps.spec(s -> ClassicOps.withLast(s, outputs),
                (s, r) -> new NeuralArithmeticUnit(s.dim(s.rank() - 1), outputs, kind, r));
    }

    /** @return the effective weight {@code tanh(W') * sigmoid(M')} */
    public Var weight() {
        return hatW.tanh().mul(hatM.sigmoid());
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().dim(input.shape().rank() - 1) != inputs) {
            throw new IllegalArgumentException(this + " expects [*, " + inputs + "], got " + input.shape());
        }
        Var w = weight();
        Var a = input.matmul(w);
        if (kind == Kind.NAC) {
            return a;
        }
        Var m = input.abs().add(1e-7f).log().matmul(w).exp();
        Var g = input.matmul(gate).sigmoid();
        return m.add(g.mul(a.sub(m)));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(hatW, hatM));
        if (gate != null) {
            p.add(gate);
        }
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    @Override
    public String toString() {
        return "NeuralArithmeticUnit(" + kind + ", " + inputs + " -> " + outputs + ")";
    }
}
