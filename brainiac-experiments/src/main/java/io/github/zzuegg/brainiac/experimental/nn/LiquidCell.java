package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Liquid continuous-time recurrent cells, run over a sequence with (optionally irregular)
 * time gaps between steps. Two kinds:
 *
 * <ul>
 *   <li>{@link Kind#LTC}, a liquid time-constant network: each neuron follows
 *       {@code dx/dt = -(1/tau + f) x + f A} with a gate {@code f = sigmoid([u, x] W + b)}
 *       that depends on input and state, so the effective time constant changes with the
 *       input ("liquid"). Integrated by the paper's fused semi-implicit Euler step
 *       {@code x <- (x + dt f A) / (1 + dt (1/tau + f))}, unfolded several times per gap,
 *       which keeps the state bounded and stable.</li>
 *   <li>{@link Kind#CFC}, closed-form continuous-time: the LTC solution approximated in
 *       closed form, {@code x(t) = sigma(-f t) . g + (1 - sigma(-f t)) . h}, with
 *       {@code f, g, h} heads on a shared backbone of {@code [u, x]}. No ODE solver, so it is
 *       one to two orders of magnitude faster, and the elapsed time {@code t} enters the
 *       gate directly.</li>
 * </ul>
 *
 * <p>Input {@code [batch, steps, inputs]}, output the states {@code [batch, steps, hidden]}.
 * Tiny networks of such neurons make interpretable controllers for robotics and RL.
 *
 * <p>Papers: Hasani et al., "Liquid Time-constant Networks",
 * <a href="https://arxiv.org/abs/2006.04439">arXiv:2006.04439</a>; Hasani et al., "Closed-form
 * Continuous-time Neural Models", <a href="https://arxiv.org/abs/2106.13898">arXiv:2106.13898</a>.
 */
public final class LiquidCell implements Layer {

    /** The cell. */
    public enum Kind {
        /** Liquid time-constant ODE with a fused solver. */
        LTC,
        /** Closed-form continuous-time cell. */
        CFC
    }

    private static final int UNFOLDS = 6;

    private final Kind kind;
    private final int inputs;
    private final int hidden;
    private final List<Var> params;

    /**
     * Creates the cell.
     *
     * @param kind LTC or CfC
     * @param inputs the input width
     * @param hidden the number of neurons
     * @param random the random source
     */
    public LiquidCell(Kind kind, int inputs, int hidden, RandomGenerator random) {
        this.kind = kind;
        this.inputs = inputs;
        this.hidden = hidden;
        int in = inputs + hidden;
        if (kind == Kind.LTC) {
            params = List.of(MixerOps.weight(random, in, hidden), MixerOps.filled(0f, hidden),
                    MixerOps.filled(0f, hidden),                                   // log tau (softplus)
                    MixerOps.uniform(random, 1f, hidden));                         // A, the reversal potentials
        } else {
            int backbone = 2 * hidden;
            params = List.of(MixerOps.weight(random, in, backbone), MixerOps.filled(0f, backbone),
                    MixerOps.weight(random, backbone, hidden), MixerOps.filled(0f, hidden),
                    MixerOps.weight(random, backbone, hidden), MixerOps.filled(0f, hidden),
                    MixerOps.weight(random, backbone, hidden), MixerOps.filled(0f, hidden));
        }
    }

    /**
     * Describes the cell for {@code Model.sequential().add(...)}.
     *
     * @param kind LTC or CfC
     * @param hidden the number of neurons
     * @return the spec
     */
    public static LayerSpec of(Kind kind, int hidden) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), hidden);
            }

            @Override
            public Layer create(Shape input, java.util.SplittableRandom random) {
                return new LiquidCell(kind, input.dim(-1), hidden, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return forward(input, null);
    }

    /**
     * Runs the cell with irregular time gaps.
     *
     * @param input {@code [batch, steps, inputs]}
     * @param gaps the time elapsed before each step, {@code [batch][steps]}, or {@code null} for 1
     * @return the states {@code [batch, steps, hidden]}
     */
    public Var forward(Var input, float[][] gaps) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(2) != inputs) {
            throw new IllegalArgumentException("LiquidCell expects [batch, steps, " + inputs + "], got " + s);
        }
        int b = s.dim(0);
        int t = s.dim(1);
        Var x = Var.of(Tensor.zeros(Shape.of(b, hidden)));
        List<Var> out = new ArrayList<>();
        for (int step = 0; step < t; step++) {
            Var u = input.slice(1, step, step + 1).reshape(b, inputs);
            float[] g = new float[b];
            for (int i = 0; i < b; i++) {
                g[i] = gaps == null ? 1f : gaps[i][step];
            }
            Tensor dt = Tensor.of(Shape.of(b, 1), g);
            x = kind == Kind.LTC ? ltc(u, x, dt) : cfc(u, x, dt);
            out.add(x.reshape(b, 1, hidden));
        }
        return Var.concat(1, out.toArray(Var[]::new));
    }

    private Var ltc(Var u, Var x, Tensor dt) {
        Var tauInv = params.get(2).softplus().add(0.1f).pow(-1f);
        Tensor h = dt.div(UNFOLDS);
        for (int i = 0; i < UNFOLDS; i++) {
            Var f = Var.concat(1, u, x).matmul(params.get(0)).add(params.get(1)).sigmoid();
            x = x.add(f.mul(params.get(3)).mul(h)).div(f.add(tauInv).mul(h).add(1f));
        }
        return x;
    }

    private Var cfc(Var u, Var x, Tensor dt) {
        Var bb = Var.concat(1, u, x).matmul(params.get(0)).add(params.get(1)).tanh();
        Var f = bb.matmul(params.get(2)).add(params.get(3));
        Var g = bb.matmul(params.get(4)).add(params.get(5)).tanh();
        Var hh = bb.matmul(params.get(6)).add(params.get(7)).tanh();
        Var gate = f.neg().mul(dt).sigmoid();
        return gate.mul(g).add(gate.neg().add(1f).mul(hh));
    }

    @Override
    public List<Var> parameters() {
        return params;
    }

    @Override
    public int outputSize(int inputSize) {
        return hidden;
    }

    @Override
    public String toString() {
        return "LiquidCell(" + kind + ", " + inputs + " -> " + hidden + ")";
    }
}
