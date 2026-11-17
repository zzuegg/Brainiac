package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Active dendrites with k-winners-take-all (Iyer et al., "Avoiding catastrophe: active
 * dendrites enable multi-task learning in dynamic environments",
 * <a href="https://arxiv.org/abs/2201.00042">arXiv:2201.00042</a>). Each unit has a
 * feed-forward weight vector, as in a dense layer, and {@code segments} dendritic segments,
 * each a weight vector over a <em>context</em> (for example a task embedding). The segment that
 * responds most strongly (largest magnitude) to the context gates the unit:
 *
 * <pre>
 * f   = W x + b
 * d   = u_s* . c,   s* = argmax_s |u_s . c|
 * y   = kWTA(f * sigmoid(d))      (only the k largest units of each sample stay active)
 * </pre>
 *
 * <p>Different contexts recruit different segments and so different sparse subnetworks, and
 * the k-WTA keeps the subnetworks small; gradients only flow through the winners, so learning a
 * new task leaves most of the weights other tasks rely on untouched.
 *
 * <p>The input is the feature vector followed by the context, {@code [batch, inputs + context]};
 * the output is {@code [batch, units]} followed by the unchanged context, so layers stack and the
 * context reaches every layer (the last layer of a network usually drops it with a slice or uses
 * {@link #dropContext}).
 */
public final class ActiveDendriteLayer implements Layer {
    /**
     * The scale of the initial dendritic weights, times {@code 1/sqrt(context)}. Large on
     * purpose: {@code sigmoid(d)} starts near 0 or 1, so each context opens a random subset of
     * units, a near-binary task mask that training then refines; with a small scale every gate
     * starts near one half and the tasks share their units.
     */
    public static final double DENDRITE_INIT = 30;
    private final int inputs;
    private final int context;
    private final int units;
    private final int segments;
    private final int winners;
    private final boolean dropContext;
    private final Var weight;
    private final Var bias;
    private final Var dendrites;

    private ActiveDendriteLayer(int inputs, int context, int units, int segments, float density, boolean dropContext,
            RandomGenerator random) {
        if (segments < 1 || !(density > 0 && density <= 1)) {
            throw new IllegalArgumentException("need segments >= 1 and 0 < density <= 1");
        }
        this.inputs = inputs;
        this.context = context;
        this.units = units;
        this.segments = segments;
        this.winners = Math.max(1, Math.round(units * density));
        this.dropContext = dropContext;
        this.weight = Var.param(Init.HE.create(Shape.of(inputs, units), inputs, units, random));
        this.bias = Var.param(Tensor.zeros(Shape.of(units)));
        float[] dw = new float[context * units * segments];
        for (int i = 0; i < dw.length; i++) {
            dw[i] = (float) (random.nextGaussian() * DENDRITE_INIT / Math.sqrt(context));
        }
        this.dendrites = Var.param(Tensor.of(Shape.of(context, units * segments), dw));
    }

    /**
     * Creates a layer.
     *
     * @param inputs feed-forward inputs
     * @param context context width
     * @param units units
     * @param segments dendritic segments per unit (at least the number of tasks, in the paper)
     * @param density the share of units the k-WTA keeps active
     * @param dropContext whether the output omits the context (for the last dendritic layer)
     * @param seed the seed
     * @return the layer
     */
    public static ActiveDendriteLayer of(int inputs, int context, int units, int segments, float density,
            boolean dropContext, long seed) {
        return new ActiveDendriteLayer(inputs, context, units, segments, density, dropContext,
                new SplittableRandom(seed));
    }

    /**
     * Describes the layer for model builders; the input width is {@code features + context}.
     *
     * @param context the context width (the trailing part of the input)
     * @param units units
     * @param segments segments per unit
     * @param density the active share
     * @param dropContext whether the output omits the context
     * @return the spec
     */
    public static LayerSpec spec(int context, int units, int segments, float density, boolean dropContext) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(units + (dropContext ? 0 : context));
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new ActiveDendriteLayer(input.dim(-1) - context, context, units, segments, density,
                        dropContext, random);
            }
        };
    }

    /** @return whether the output omits the context */
    public boolean dropContext() {
        return dropContext;
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 2 || s.dim(1) != inputs + context) {
            throw new IllegalArgumentException("ActiveDendriteLayer expects [batch, " + (inputs + context) + "], got "
                    + s);
        }
        int b = s.dim(0);
        Var x = input.slice(1, 0, inputs);
        Var c = input.slice(1, inputs, inputs + context);
        Var f = x.matmul(weight).add(bias);
        Var a = c.matmul(dendrites).reshape(b, units, segments);
        Var d = a.mask(strongestSegment(a.value())).sum(2, false);
        Var gated = f.mul(d.sigmoid());
        Var out = gated.mask(winnerMask(gated.value(), b));
        return dropContext ? out : Var.concat(1, out, c);
    }

    /** One-hot mask of the segment with the largest absolute response, per unit. */
    private Tensor strongestSegment(Tensor responses) {
        float[] r = responses.toArray();
        float[] m = new float[r.length];
        for (int u = 0; u < r.length / segments; u++) {
            int best = 0;
            for (int k = 1; k < segments; k++) {
                if (Math.abs(r[u * segments + k]) > Math.abs(r[u * segments + best])) {
                    best = k;
                }
            }
            m[u * segments + best] = 1;
        }
        return Tensor.of(responses.shape(), m);
    }

    /** The k-WTA mask: ones at the {@code winners} largest units of every row. */
    private Tensor winnerMask(Tensor v, int b) {
        float[] a = v.toArray();
        float[] m = new float[a.length];
        float[] row = new float[units];
        for (int i = 0; i < b; i++) {
            System.arraycopy(a, i * units, row, 0, units);
            float[] sorted = row.clone();
            java.util.Arrays.sort(sorted);
            float threshold = sorted[units - winners];
            int kept = 0;
            for (int j = 0; j < units && kept < winners; j++) {
                if (row[j] >= threshold) {
                    m[i * units + j] = 1;
                    kept++;
                }
            }
        }
        return Tensor.of(Shape.of(b, units), m);
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias, dendrites);
    }

    @Override
    public int outputSize(int inputSize) {
        return units + (dropContext ? 0 : context);
    }
}
