package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Tiny Recursive Model (Jolicoeur-Martineau, <a href="https://arxiv.org/abs/2510.04871">2510.04871</a>):
 * one small network, applied over and over, refines a latent reasoning state {@code z} and
 * a current answer {@code y} for an embedded question {@code x}:
 * <pre>
 * repeat cycles times:
 *     repeat latentSteps times:  z = net(x, y, z)     // think
 *     y = net(0, y, z)                                 // revise the answer
 * </pre>
 * Only the last cycle is differentiated (the earlier ones only move the state), which keeps
 * memory constant in the recursion depth. Training uses <em>deep supervision</em>: the
 * whole recursion runs {@code supervisionSteps} times, the state carried over (detached)
 * from one to the next, with a loss after each; {@link #supervised} returns those
 * intermediate answers. Depth comes from iteration, not parameters: a 2-layer net reasoning
 * for dozens of effective layers.
 *
 * <p>As a {@link Layer} it maps {@code [batch, inputs]} to {@code [batch, outputs]} (the
 * logits after the last supervision step), so it drops into {@code Model.sequential()} via
 * {@link #spec}.
 */
public final class TinyRecursiveModel implements Layer {

    private final int inputs;
    private final int dim;
    private final int outputs;
    private final int latentSteps;
    private final int cycles;
    private final int supervisionSteps;
    private final Lin embed;
    private final Lin hidden;
    private final Lin back;
    private final Lin head;
    private final Var y0;
    private final Var z0;

    private record Lin(Var w, Var b) {
        static Lin of(int in, int out, SplittableRandom r) {
            float scale = (float) Math.sqrt(1.0 / in);
            float[] w = new float[in * out];
            for (int i = 0; i < w.length; i++) {
                w[i] = (float) (r.nextGaussian() * scale);
            }
            return new Lin(Var.param(Tensor.of(Shape.of(in, out), w)), Var.param(Tensor.zeros(Shape.of(1, out))));
        }

        Var apply(Var x) {
            return x.matmul(w).add(b);
        }
    }

    private TinyRecursiveModel(int inputs, int dim, int outputs, int latentSteps, int cycles, int supervisionSteps,
            SplittableRandom r) {
        this.inputs = inputs;
        this.dim = dim;
        this.outputs = outputs;
        this.latentSteps = latentSteps;
        this.cycles = cycles;
        this.supervisionSteps = supervisionSteps;
        this.embed = Lin.of(inputs, dim, r);
        this.hidden = Lin.of(3 * dim, 2 * dim, r);
        this.back = Lin.of(2 * dim, dim, r);
        this.head = Lin.of(dim, outputs, r);
        this.y0 = Var.of(Tensor.zeros(Shape.of(1, dim)));
        this.z0 = Var.of(Tensor.zeros(Shape.of(1, dim)));
    }

    /**
     * @param inputs input features
     * @param dim the width of {@code x}, {@code y} and {@code z}
     * @param outputs output logits
     * @param latentSteps latent updates per cycle ({@code n})
     * @param cycles cycles per supervision step ({@code T}); only the last gets gradients
     * @param supervisionSteps deep-supervision steps
     * @param seed the initialization seed
     * @return the layer
     */
    public static TinyRecursiveModel of(int inputs, int dim, int outputs, int latentSteps, int cycles,
            int supervisionSteps, long seed) {
        return new TinyRecursiveModel(inputs, dim, outputs, latentSteps, cycles, supervisionSteps,
                new SplittableRandom(seed));
    }

    /**
     * @param dim the state width
     * @param outputs output logits
     * @param latentSteps latent updates per cycle
     * @param cycles cycles per supervision step
     * @param supervisionSteps deep-supervision steps
     * @return a spec for model builders
     */
    public static LayerSpec spec(int dim, int outputs, int latentSteps, int cycles, int supervisionSteps) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new TinyRecursiveModel(input.dim(input.rank() - 1), dim, outputs, latentSteps, cycles,
                        supervisionSteps, random);
            }
        };
    }

    /** The carried reasoning state of a batch. */
    private record State(Var y, Var z) {
    }

    private static Var rmsNorm(Var v) {
        return v.div(v.square().mean(1, true).add(1e-5f).sqrt());
    }

    private Var net(Var x, Var y, Var z) {
        Var h = hidden.apply(Var.concat(1, x, y, z));
        h = h.mul(h.sigmoid());
        return rmsNorm(z.add(back.apply(h)));
    }

    private State cycle(Var x, Var zeros, State s) {
        Var z = s.z;
        for (int i = 0; i < latentSteps; i++) {
            z = net(x, s.y, z);
        }
        Var y = rmsNorm(s.y.add(back.apply(silu(hidden.apply(Var.concat(1, zeros, s.y, z))))));
        return new State(y, z);
    }

    private static Var silu(Var h) {
        return h.mul(h.sigmoid());
    }

    /**
     * Runs deep supervision: the recursion {@code supervisionSteps} times, carrying the
     * (detached) state between steps.
     *
     * @param input {@code [batch, inputs]}
     * @return the logits after each supervision step; put a loss on each
     */
    public List<Var> supervised(Var input) {
        int batch = input.shape().dim(0);
        Var x = embed.apply(input);
        Var zeros = Var.of(Tensor.zeros(Shape.of(batch, dim)));
        Var ones = Var.of(Tensor.ones(Shape.of(batch, 1)));
        State s = new State(ones.matmul(y0), ones.matmul(z0));
        List<Var> outs = new ArrayList<>();
        for (int step = 0; step < supervisionSteps; step++) {
            for (int c = 0; c < cycles - 1; c++) {
                State next = cycle(x.detach(), zeros, s);
                s = new State(next.y.detach(), next.z.detach());
            }
            s = cycle(x, zeros, s);
            outs.add(head.apply(s.y));
            s = new State(s.y.detach(), s.z.detach());
        }
        return outs;
    }

    @Override
    public Var forward(Var input) {
        List<Var> outs = supervised(input);
        return outs.get(outs.size() - 1);
    }

    @Override
    public List<Var> parameters() {
        return List.of(embed.w, embed.b, hidden.w, hidden.b, back.w, back.b, head.w, head.b);
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    /** @return the input width */
    public int inputs() {
        return inputs;
    }

    /** @return the effective depth: network applications per supervision step */
    public int effectiveDepth() {
        return cycles * (latentSteps + 1);
    }
}
