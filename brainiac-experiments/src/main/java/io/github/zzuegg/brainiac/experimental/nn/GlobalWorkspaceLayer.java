package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A shared global workspace for specialist modules (Goyal et al., "Coordination among neural
 * modules through a shared global workspace", <a href="https://arxiv.org/abs/2103.01197">2103.01197</a>;
 * after Baars' global workspace theory). Specialists (tokens, slots, modules) do not talk
 * pairwise; they <em>compete</em> to write into a few workspace slots, and the workspace
 * is then <em>broadcast</em> back to all of them:
 *
 * <ol>
 * <li>write: learned workspace queries attend over the specialists; only the {@code topK}
 *     specialists with the strongest attention may write (the others' attention weights are
 *     zeroed, the winners keep their softmax share), which makes the workspace a
 *     limited-capacity bottleneck;</li>
 * <li>broadcast: every specialist attends over the updated slots and adds the result to its
 *     state (residual).</li>
 * </ol>
 *
 * <p>Input and output {@code [batch, specialists, dim]}. {@link #lastWriters()} reports
 * which specialists won the competition in the last pass.
 */
public final class GlobalWorkspaceLayer implements Layer {
    private final int dim;
    private final int slots;
    private final int topK;
    private final Var memory;
    private final Var writeQuery;
    private final Var writeKey;
    private final Var writeValue;
    private final Var readQuery;
    private final Var readKey;
    private final Var readValue;
    private boolean[][] lastWriters;

    private GlobalWorkspaceLayer(int dim, int slots, int topK, SplittableRandom random) {
        this.dim = dim;
        this.slots = slots;
        this.topK = topK;
        this.memory = Var.param(Tensor.normal(Shape.of(slots, dim), 0, 1, random));
        this.writeQuery = w(random);
        this.writeKey = w(random);
        this.writeValue = w(random);
        this.readQuery = w(random);
        this.readKey = w(random);
        this.readValue = w(random);
    }

    private Var w(SplittableRandom random) {
        return Var.param(Init.XAVIER.create(Shape.of(dim, dim), dim, dim, random));
    }

    /**
     * Creates a layer.
     *
     * @param dim the specialist and slot width
     * @param slots the workspace slots
     * @param topK how many specialists may write per pass
     * @param seed the seed
     * @return the layer
     */
    public static GlobalWorkspaceLayer of(int dim, int slots, int topK, long seed) {
        if (slots < 1 || topK < 1) {
            throw new IllegalArgumentException("need at least one slot and one writer");
        }
        return new GlobalWorkspaceLayer(dim, slots, topK, new SplittableRandom(seed));
    }

    /**
     * Describes the layer for model builders; input {@code [specialists, dim]}.
     *
     * @param slots the workspace slots
     * @param topK the writers per pass
     * @return the spec
     */
    public static LayerSpec spec(int slots, int topK) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new GlobalWorkspaceLayer(input.dim(-1), slots, topK, random);
            }
        };
    }

    private static Var softmax(Var x, int axis) {
        Var e = x.sub(Var.of(x.value().max(axis, true))).exp();
        return e.div(e.sum(axis, true));
    }

    @Override
    public Var forward(Var h) {
        Shape s = h.shape();
        if (s.rank() != 3 || s.dim(2) != dim) {
            throw new IllegalArgumentException("GlobalWorkspaceLayer expects [batch, specialists, " + dim + "], got " + s);
        }
        int b = s.dim(0);
        int n = s.dim(1);
        float scale = (float) (1 / Math.sqrt(dim));
        // write: slots query the specialists, only the top-k strongest specialists may write
        Var q = memory.matmul(writeQuery);                                    // [slots, dim]
        Var k = h.matmul(writeKey);                                           // [b, n, dim]
        Var scores = k.matmul(q.transpose()).mul(scale).permute(0, 2, 1);     // [b, slots, n]
        float[] strength = scores.value().max(1, false).toArray();            // [b, n]
        float[] mask = new float[b * n];
        lastWriters = new boolean[b][n];
        int keep = Math.min(topK, n);
        for (int i = 0; i < b; i++) {
            Integer[] order = new Integer[n];
            for (int j = 0; j < n; j++) {
                order[j] = j;
            }
            int row = i;
            java.util.Arrays.sort(order, (x, y) -> Float.compare(strength[row * n + y], strength[row * n + x]));
            for (int t = 0; t < keep; t++) {
                mask[i * n + order[t]] = 1;
                lastWriters[i][order[t]] = true;
            }
        }
        // attention over all specialists, then only the winners' shares are written (not renormalized,
        // so the competition itself stays differentiable: a winner writes more the more it dominates)
        Var attend = softmax(scores, 2).mul(Tensor.of(Shape.of(b, 1, n), mask, h.value().device()));
        Var workspace = attend.matmul(h.matmul(writeValue)).add(memory);    // [b, slots, dim]
        // broadcast: every specialist reads the workspace
        Var rq = h.matmul(readQuery);
        Var rk = workspace.matmul(readKey);
        Var read = softmax(rq.matmul(rk.permute(0, 2, 1)).mul(scale), 2);   // [b, n, slots]
        return h.add(read.matmul(workspace.matmul(readValue)));
    }

    /** @return which specialists wrote in the last pass, {@code [batch][specialists]} */
    public boolean[][] lastWriters() {
        return lastWriters;
    }

    /** @return the number of slots */
    public int slots() {
        return slots;
    }

    @Override
    public List<Var> parameters() {
        return List.of(memory, writeQuery, writeKey, writeValue, readQuery, readKey, readValue);
    }
}
