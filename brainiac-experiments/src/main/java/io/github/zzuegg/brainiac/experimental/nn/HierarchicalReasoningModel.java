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
 * Hierarchical Reasoning Model (Wang et al., <a href="https://arxiv.org/abs/2506.21734">2506.21734</a>):
 * two coupled recurrent modules at different time scales. The fast low-level module
 * {@code L} takes {@code lowSteps} steps for every single step of the slow high-level
 * module {@code H}, which then updates from the low-level result and resets its context:
 * <pre>
 * repeat highSteps times:
 *     repeat lowSteps times:  zL = fL(zL, zH, x)
 *     zH = fH(zH, zL)
 * </pre>
 * Training uses the paper's one-step gradient approximation (only the last {@code L} and
 * {@code H} updates are differentiated, the rest just move the state, so memory does not grow
 * with depth) and deep supervision over {@code supervisionSteps} segments, as
 * {@link #supervised} returns.
 *
 * <p>As a {@link Layer} it maps {@code [batch, inputs]} to {@code [batch, outputs]};
 * {@link #spec} plugs it into {@code Model.sequential()}.
 */
public final class HierarchicalReasoningModel implements Layer {

    private final int dim;
    private final int outputs;
    private final int lowSteps;
    private final int highSteps;
    private final int supervisionSteps;
    private final Lin embed;
    private final Lin low;
    private final Lin high;
    private final Lin head;
    private final Var l0;
    private final Var h0;

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

    private HierarchicalReasoningModel(int inputs, int dim, int outputs, int lowSteps, int highSteps,
            int supervisionSteps, SplittableRandom r) {
        this.dim = dim;
        this.outputs = outputs;
        this.lowSteps = lowSteps;
        this.highSteps = highSteps;
        this.supervisionSteps = supervisionSteps;
        this.embed = Lin.of(inputs, dim, r);
        this.low = Lin.of(3 * dim, dim, r);
        this.high = Lin.of(2 * dim, dim, r);
        this.head = Lin.of(dim, outputs, r);
        this.l0 = Var.of(Tensor.zeros(Shape.of(1, dim)));
        this.h0 = Var.of(Tensor.zeros(Shape.of(1, dim)));
    }

    /**
     * @param inputs input features
     * @param dim the width of both states
     * @param outputs output logits
     * @param lowSteps low-level steps per high-level step ({@code T})
     * @param highSteps high-level steps per segment ({@code N})
     * @param supervisionSteps deep-supervision segments
     * @param seed the initialization seed
     * @return the layer
     */
    public static HierarchicalReasoningModel of(int inputs, int dim, int outputs, int lowSteps, int highSteps,
            int supervisionSteps, long seed) {
        return new HierarchicalReasoningModel(inputs, dim, outputs, lowSteps, highSteps, supervisionSteps,
                new SplittableRandom(seed));
    }

    /**
     * @param dim the width of both states
     * @param outputs output logits
     * @param lowSteps low-level steps per high-level step
     * @param highSteps high-level steps per segment
     * @param supervisionSteps deep-supervision segments
     * @return a spec for model builders
     */
    public static LayerSpec spec(int dim, int outputs, int lowSteps, int highSteps, int supervisionSteps) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new HierarchicalReasoningModel(input.dim(input.rank() - 1), dim, outputs, lowSteps,
                        highSteps, supervisionSteps, random);
            }
        };
    }

    private static Var norm(Var v) {
        return v.div(v.square().mean(1, true).add(1e-5f).sqrt());
    }

    private Var lowStep(Var zl, Var zh, Var x) {
        return norm(zl.add(low.apply(Var.concat(1, zl, zh, x)).tanh()));
    }

    private Var highStep(Var zh, Var zl) {
        return norm(zh.add(high.apply(Var.concat(1, zh, zl)).tanh()));
    }

    /**
     * Runs deep supervision.
     *
     * @param input {@code [batch, inputs]}
     * @return the logits after each segment
     */
    public List<Var> supervised(Var input) {
        int batch = input.shape().dim(0);
        Var x = embed.apply(input);
        Var ones = Var.of(Tensor.ones(Shape.of(batch, 1)));
        Var zl = ones.matmul(l0);
        Var zh = ones.matmul(h0);
        List<Var> outs = new ArrayList<>();
        int total = lowSteps * highSteps;
        for (int seg = 0; seg < supervisionSteps; seg++) {
            Var xd = x.detach();
            int step = 0;
            for (int h = 0; h < highSteps; h++) {
                for (int l = 0; l < lowSteps; l++) {
                    step++;
                    boolean last = step == total;
                    zl = last ? lowStep(zl.detach(), zh.detach(), x) : lowStep(zl, zh, xd).detach();
                }
                zh = h == highSteps - 1 ? highStep(zh.detach(), zl) : highStep(zh, zl).detach();
            }
            outs.add(head.apply(zh));
            zl = zl.detach();
            zh = zh.detach();
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
        return List.of(embed.w, embed.b, low.w, low.b, high.w, high.b, head.w, head.b);
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    /** @return the state width */
    public int dim() {
        return dim;
    }
}
