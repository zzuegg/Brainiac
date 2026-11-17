package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Mixture-of-Recursions: a shared block applied recursively, where a router gives every
 * token its own recursion depth. This uses the paper's expert-choice routing: at recursion
 * {@code r}, a router of its own scores the tokens still active, and only the top
 * {@code ceil(keep^r * steps)} of them continue (a subset of the previous recursion's
 * tokens), so the number of active tokens shrinks geometrically. Selected tokens are
 * gathered in order, run through the block as a shorter sequence (attention sees only the
 * tokens active at that depth, the paper's recursion-wise KV caching) and updated as
 * {@code h + sigmoid(score) (block(h) - h)}, which gives the router a gradient.
 *
 * <p>{@link #lastDepths()} reports how many recursions each token got. Like
 * {@link MixtureOfDepths}, top-k over a sequence looks ahead; that is fine for training and
 * encoders, and causal sampling needs a predictor or token-choice routing.
 *
 * <p>Paper: Bae et al., "Mixture-of-Recursions: Learning Dynamic Recursive Depths for
 * Adaptive Token-Level Computation", <a href="https://arxiv.org/abs/2507.10524">arXiv:2507.10524</a>.
 */
public final class MixtureOfRecursions implements Layer {

    private final Layer block;
    private final int features;
    private final int recursions;
    private final float keep;
    private final List<Var> routers = new ArrayList<>();
    private int[][] lastDepths;

    /**
     * Creates the layer.
     *
     * @param block the shared block, mapping a state {@code [b, c, features]} to the next state
     * @param features the width
     * @param recursions the most recursions
     * @param keep the share of active tokens that continue at each recursion, in {@code (0, 1]}
     * @param random the random source
     */
    public MixtureOfRecursions(Layer block, int features, int recursions, float keep, RandomGenerator random) {
        if (!(keep > 0 && keep <= 1) || recursions < 1) {
            throw new IllegalArgumentException("need recursions >= 1 and keep in (0, 1]");
        }
        this.block = block;
        this.features = features;
        this.recursions = recursions;
        this.keep = keep;
        for (int i = 0; i < recursions; i++) {
            routers.add(MixerOps.weight(random, features, 1));
        }
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param block the spec of the shared block
     * @param recursions the most recursions
     * @param keep the share of active tokens that continue at each recursion
     * @return the spec
     */
    public static LayerSpec of(LayerSpec block, int recursions, float keep) {
        return MixerOps.sameShape((in, r) -> new MixtureOfRecursions(block.create(in, r), in.dim(-1), recursions, keep, r));
    }

    /** @return the recursion depth of every token in the last forward pass, {@code [batch][steps]} */
    public int[][] lastDepths() {
        return lastDepths;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        MixerOps.requireSequence(input, features, "MixtureOfRecursions");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int[][] active = new int[b][t];
        for (int n = 0; n < b; n++) {
            for (int i = 0; i < t; i++) {
                active[n][i] = i;
            }
        }
        int[][] depth = new int[b][t];
        Var h = input.reshape(b * t, features);
        for (int r = 0; r < recursions; r++) {
            int c = Math.max(1, (int) Math.ceil(Math.pow(keep, r + 1) * t));
            c = Math.min(c, active[0].length);
            // score the active tokens, keep the top c, in order
            int[] flatActive = new int[b * active[0].length];
            for (int n = 0; n < b; n++) {
                for (int j = 0; j < active[n].length; j++) {
                    flatActive[n * active[n].length + j] = n * t + active[n][j];
                }
            }
            float[] scores = h.gather(0, MixerOps.ids(flatActive)).matmul(routers.get(r)).value().toArray();
            int[][] next = new int[b][];
            int[] flatSel = new int[b * c];
            for (int n = 0; n < b; n++) {
                int[] top = MixerOps.topK(scores, n * active[n].length, active[n].length, c);
                int[] pos = new int[c];
                for (int j = 0; j < c; j++) {
                    pos[j] = active[n][top[j]];
                }
                Arrays.sort(pos);
                next[n] = pos;
                for (int j = 0; j < c; j++) {
                    flatSel[n * c + j] = n * t + pos[j];
                    depth[n][pos[j]]++;
                }
            }
            Tensor ids = MixerOps.ids(flatSel);
            Var sel = h.gather(0, ids);
            Var gate = sel.matmul(routers.get(r)).sigmoid();
            Var out = block.forward(sel.reshape(b, c, features), training).reshape(b * c, features);
            Var delta = out.sub(sel).mul(gate);
            h = h.add(MixerOps.scatterAdd(delta, 0, ids, b * t));
            active = next;
        }
        lastDepths = depth;
        return h.reshape(b, t, features);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(routers);
        p.addAll(block.parameters());
        return p;
    }

    @Override
    public String toString() {
        return "MixtureOfRecursions(" + block + ", recursions=" + recursions + ", keep=" + keep + ")";
    }
}
