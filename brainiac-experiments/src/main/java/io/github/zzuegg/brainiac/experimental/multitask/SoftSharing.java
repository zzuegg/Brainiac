package io.github.zzuegg.brainiac.experimental.multitask;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Multi-task architectures, from hard to soft parameter sharing; every task has a linear head and
 * {@link #forward} concatenates the heads' outputs. The loss side is left to the caller, so any
 * {@link LossWeighting} or {@link GradientSurgery} scheme applies ({@link #sharedParameters} are
 * the parameters more than one task's loss reaches).
 * <ul>
 *   <li>{@link Kind#HARD}: one shared ReLU trunk.</li>
 *   <li>{@link Kind#SEPARATE}: one column per task and no sharing (single-task networks inside one
 *       module).</li>
 *   <li>{@link Kind#CROSS_STITCH} (Misra et al., CVPR 2016,
 *       <a href="https://arxiv.org/abs/1604.03539">arXiv:1604.03539</a>): one column per task; after
 *       each layer the columns' activations are mixed, {@code h~_t = sum_s alpha_ts h_s}, with a
 *       learned {@code T x T} unit per layer, initialised at 0.9 on the diagonal.</li>
 *   <li>{@link Kind#SLUICE} (Ruder et al., AAAI 2019,
 *       <a href="https://arxiv.org/abs/1705.08142">arXiv:1705.08142</a>): each column's layer is split
 *       into two subspaces and a {@code 2T x 2T} unit mixes all subspaces of all tasks; each task's head
 *       reads a softmax-weighted mixture of all layers (layer skips), and
 *       {@link #auxiliaryLoss()} gives the subspace orthogonality penalty
 *       {@code sum |G_1^T G_2|_F^2 / batch^2} of the last forward pass.</li>
 *   <li>{@link Kind#TASK_ATTENTION} (MTAN, Liu, Johns and Davison, CVPR 2019,
 *       <a href="https://arxiv.org/abs/1803.10704">arXiv:1803.10704</a>): a shared trunk and, per
 *       task, a soft attention mask at every layer, {@code a_t^l = sigmoid([s^l, f_t^{l-1}] A_t^l)},
 *       with task features {@code f_t^l = a_t^l * s^l}.</li>
 * </ul>
 */
public final class SoftSharing implements Layer {

    /** The architecture. */
    public enum Kind {
        /** A shared trunk. */
        HARD,
        /** Separate task columns. */
        SEPARATE,
        /** Task columns mixed by cross-stitch units. */
        CROSS_STITCH,
        /** Sluice network: subspace mixing and layer skips. */
        SLUICE,
        /** A shared trunk with per-task attention masks (MTAN). */
        TASK_ATTENTION
    }

    private final Kind kind;
    private final int inputs;
    private final int tasks;
    private final int hidden;
    private final int layers;
    private final int[] outputs;
    /** [column][layer] weight and bias; one column for HARD and TASK_ATTENTION. */
    private final List<List<Var[]>> columns = new ArrayList<>();
    private final List<Var> stitches = new ArrayList<>();
    private final List<Var> skips = new ArrayList<>();
    /** [task][layer] attention weight and bias. */
    private final List<List<Var[]>> attention = new ArrayList<>();
    private final List<Var[]> heads = new ArrayList<>();
    private Var orthogonality;

    /**
     * Creates the module.
     *
     * @param kind the architecture
     * @param inputs input features
     * @param outputs output width of each task's head
     * @param hidden the width of each layer (of each column for the column architectures; even for SLUICE)
     * @param layers hidden layers
     * @param random the initialiser
     */
    public SoftSharing(Kind kind, int inputs, int[] outputs, int hidden, int layers, SplittableRandom random) {
        if (kind == Kind.SLUICE && hidden % 2 != 0) {
            throw new IllegalArgumentException("SLUICE needs an even width");
        }
        this.kind = kind;
        this.inputs = inputs;
        this.tasks = outputs.length;
        this.hidden = hidden;
        this.layers = layers;
        this.outputs = outputs.clone();
        int cols = kind == Kind.HARD || kind == Kind.TASK_ATTENTION ? 1 : tasks;
        for (int c = 0; c < cols; c++) {
            List<Var[]> column = new ArrayList<>();
            for (int l = 0; l < layers; l++) {
                column.add(dense(l == 0 ? inputs : hidden, hidden, random));
            }
            columns.add(column);
        }
        if (kind == Kind.CROSS_STITCH || kind == Kind.SLUICE) {
            int units = kind == Kind.SLUICE ? 2 * tasks : tasks;
            for (int l = 0; l < layers; l++) {
                float[] a = new float[units * units];
                for (int i = 0; i < units; i++) {
                    for (int j = 0; j < units; j++) {
                        a[i * units + j] = i == j ? 0.9f : 0.1f / (units - 1);
                    }
                }
                stitches.add(Var.param(Tensor.of(Shape.of(units, units), a)));
            }
        }
        if (kind == Kind.SLUICE) {
            for (int t = 0; t < tasks; t++) {
                float[] b = new float[layers];
                b[layers - 1] = 1;
                skips.add(Var.param(Tensor.of(Shape.of(1, layers), b)));
            }
        }
        if (kind == Kind.TASK_ATTENTION) {
            for (int t = 0; t < tasks; t++) {
                List<Var[]> masks = new ArrayList<>();
                for (int l = 0; l < layers; l++) {
                    masks.add(dense(l == 0 ? hidden : 2 * hidden, hidden, random));
                }
                attention.add(masks);
            }
        }
        for (int t = 0; t < tasks; t++) {
            heads.add(dense(hidden, outputs[t], random));
        }
    }

    private static Var[] dense(int in, int out, SplittableRandom r) {
        float[] w = new float[in * out];
        double s = Math.sqrt(2.0 / in);
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) (s * r.nextGaussian());
        }
        return new Var[] {Var.param(Tensor.of(Shape.of(in, out), w)), Var.param(Tensor.zeros(Shape.of(out)))};
    }

    private static Var apply(Var[] layer, Var x) {
        return x.matmul(layer[0]).add(layer[1]);
    }

    /** Mixes {@code parts} (each {@code [batch, width]}) by a {@code [k, k]} unit. */
    private static List<Var> mix(Var unit, List<Var> parts) {
        int b = parts.get(0).shape().dim(0);
        int w = parts.get(0).shape().dim(1);
        Var[] rows = new Var[parts.size()];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = parts.get(i).reshape(1, b * w);
        }
        Var mixed = unit.matmul(Var.concat(0, rows));
        List<Var> out = new ArrayList<>();
        for (int i = 0; i < rows.length; i++) {
            out.add(mixed.slice(0, i, i + 1).reshape(b, w));
        }
        return out;
    }

    /**
     * The task representations fed to the heads.
     *
     * @param input {@code [batch, inputs]}
     * @return one {@code [batch, hidden]} representation per task
     */
    public List<Var> taskRepresentation(Var input) {
        List<Var> out = new ArrayList<>();
        orthogonality = null;
        switch (kind) {
            case HARD -> {
                Var h = input;
                for (Var[] layer : columns.get(0)) {
                    h = apply(layer, h).relu();
                }
                for (int t = 0; t < tasks; t++) {
                    out.add(h);
                }
            }
            case SEPARATE, CROSS_STITCH -> {
                List<Var> h = new ArrayList<>();
                for (int t = 0; t < tasks; t++) {
                    h.add(input);
                }
                for (int l = 0; l < layers; l++) {
                    List<Var> next = new ArrayList<>();
                    for (int t = 0; t < tasks; t++) {
                        next.add(apply(columns.get(t).get(l), h.get(t)).relu());
                    }
                    h = kind == Kind.CROSS_STITCH ? mix(stitches.get(l), next) : next;
                }
                out = h;
            }
            case SLUICE -> out = sluice(input);
            case TASK_ATTENTION -> {
                Var s = input;
                List<Var> f = new ArrayList<>();
                for (int l = 0; l < layers; l++) {
                    s = apply(columns.get(0).get(l), s).relu();
                    List<Var> next = new ArrayList<>();
                    for (int t = 0; t < tasks; t++) {
                        Var in = l == 0 ? s : Var.concat(1, s, f.get(t));
                        Var mask = apply(attention.get(t).get(l), in).sigmoid();
                        next.add(mask.mul(s));
                    }
                    f = next;
                }
                out = f;
            }
            default -> throw new IllegalStateException(kind.toString());
        }
        return out;
    }

    private List<Var> sluice(Var input) {
        int half = hidden / 2;
        List<Var> h = new ArrayList<>();
        for (int t = 0; t < tasks; t++) {
            h.add(input);
        }
        List<List<Var>> perLayer = new ArrayList<>();
        Var ortho = null;
        for (int l = 0; l < layers; l++) {
            List<Var> parts = new ArrayList<>();
            for (int t = 0; t < tasks; t++) {
                Var a = apply(columns.get(t).get(l), h.get(t)).relu();
                Var g1 = a.slice(1, 0, half);
                Var g2 = a.slice(1, half, hidden);
                parts.add(g1);
                parts.add(g2);
                int b = a.shape().dim(0);
                Var o = g1.transpose().matmul(g2).square().sum().mul(1f / ((float) b * b));
                ortho = ortho == null ? o : ortho.add(o);
            }
            List<Var> mixed = mix(stitches.get(l), parts);
            List<Var> next = new ArrayList<>();
            for (int t = 0; t < tasks; t++) {
                next.add(Var.concat(1, mixed.get(2 * t), mixed.get(2 * t + 1)));
            }
            h = next;
            perLayer.add(next);
        }
        orthogonality = ortho;
        List<Var> out = new ArrayList<>();
        for (int t = 0; t < tasks; t++) {
            List<Var> reps = new ArrayList<>();
            for (int l = 0; l < layers; l++) {
                reps.add(perLayer.get(l).get(t));
            }
            Var beta = skips.get(t).softmax(1);
            int b = reps.get(0).shape().dim(0);
            Var[] rows = new Var[layers];
            for (int l = 0; l < layers; l++) {
                rows[l] = reps.get(l).reshape(1, b * hidden);
            }
            out.add(beta.matmul(Var.concat(0, rows)).reshape(b, hidden));
        }
        return out;
    }

    /**
     * Applies one task's head.
     *
     * @param task the task
     * @param representation its representation
     * @return {@code [batch, outputs[task]]}
     */
    public Var head(int task, Var representation) {
        return apply(heads.get(task), representation);
    }

    /**
     * All tasks' outputs.
     *
     * @param input {@code [batch, inputs]}
     * @return one output per task
     */
    public List<Var> taskOutputs(Var input) {
        List<Var> reps = taskRepresentation(input);
        List<Var> out = new ArrayList<>();
        for (int t = 0; t < tasks; t++) {
            out.add(head(t, reps.get(t)));
        }
        return out;
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().dim(-1) != inputs) {
            throw new IllegalArgumentException("expected " + inputs + " features, got " + input.shape());
        }
        return Var.concat(1, taskOutputs(input).toArray(Var[]::new));
    }

    /** @return the sluice orthogonality penalty of the last forward pass, or null for the other kinds */
    @Override
    public Var auxiliaryLoss() {
        return orthogonality;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(sharedParameters());
        for (Var[] h : heads) {
            p.add(h[0]);
            p.add(h[1]);
        }
        return p;
    }

    /**
     * The parameters other than the task heads: the trunk or columns, the stitch units, the layer
     * skips and the attention masks (for SEPARATE, each column only receives its own task's loss).
     *
     * @return the parameters
     */
    public List<Var> sharedParameters() {
        List<Var> p = new ArrayList<>();
        for (List<Var[]> column : columns) {
            for (Var[] layer : column) {
                p.add(layer[0]);
                p.add(layer[1]);
            }
        }
        p.addAll(stitches);
        p.addAll(skips);
        for (List<Var[]> masks : attention) {
            for (Var[] m : masks) {
                p.add(m[0]);
                p.add(m[1]);
            }
        }
        return p;
    }

    /**
     * The learned mixing units.
     *
     * @return {@code [layer][k][k]} ({@code k = T}, or {@code 2T} for SLUICE); empty for the other kinds
     */
    public double[][][] stitchWeights() {
        double[][][] out = new double[stitches.size()][][];
        for (int l = 0; l < out.length; l++) {
            float[] v = stitches.get(l).value().toArray();
            int k = (int) Math.round(Math.sqrt(v.length));
            out[l] = new double[k][k];
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    out[l][i][j] = v[i * k + j];
                }
            }
        }
        return out;
    }

    /**
     * The share of each mixing unit's mass on the own task (the diagonal blocks): 1 means separate
     * columns, {@code 1 / T} means uniform sharing.
     *
     * @return per layer
     */
    public double[] selfShare() {
        double[][][] w = stitchWeights();
        double[] out = new double[w.length];
        int block = kind == Kind.SLUICE ? 2 : 1;
        for (int l = 0; l < w.length; l++) {
            double own = 0;
            double all = 0;
            for (int i = 0; i < w[l].length; i++) {
                for (int j = 0; j < w[l].length; j++) {
                    double a = Math.abs(w[l][i][j]);
                    all += a;
                    own += i / block == j / block ? a : 0;
                }
            }
            out[l] = own / Math.max(1e-12, all);
        }
        return out;
    }

    /** @return the number of scalar parameters */
    public int parameterCount() {
        int s = 0;
        for (Var v : parameters()) {
            s += (int) v.value().shape().size();
        }
        return s;
    }

    /** @return the architecture */
    public Kind kind() {
        return kind;
    }

    @Override
    public int outputSize(int inputSize) {
        int s = 0;
        for (int o : outputs) {
            s += o;
        }
        return s;
    }

    @Override
    public String toString() {
        return "SoftSharing(" + kind + ", " + tasks + " tasks, " + layers + "x" + hidden + ")";
    }
}
