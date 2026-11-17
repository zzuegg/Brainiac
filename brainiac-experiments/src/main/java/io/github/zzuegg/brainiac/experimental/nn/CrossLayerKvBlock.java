package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A causal transformer stack whose layers share key/value caches, to cut KV memory across
 * depth rather than across heads:
 *
 * <ul>
 *   <li>{@link Sharing#CLA cross-layer attention} (CLA): groups of {@code group} adjacent
 *       layers share one key/value projection, computed by the first layer of the group;
 *       the others bring only queries. The cache shrinks by the factor {@code group}.</li>
 *   <li>{@link Sharing#YOCO you only cache once} (YOCO): the lower half of the stack is a
 *       self-decoder with sliding-window attention (a constant-size cache); its output is
 *       projected once into a single global key/value set that every layer of the upper
 *       half (the cross-decoder) attends to. Only that one set grows with the context.</li>
 *   <li>{@link Sharing#NONE}: the unshared baseline, one cache per layer.</li>
 * </ul>
 *
 * <p>{@link #cachedValuesPerToken()} counts what grows with the context. Every layer is
 * pre-norm with a SwiGLU feed-forward network and RoPE; input and output are
 * {@code [batch, steps, features]}.
 *
 * <p>Papers: Brandon et al., "Reducing Transformer Key-Value Cache Size with Cross-Layer
 * Attention", <a href="https://arxiv.org/abs/2405.12981">arXiv:2405.12981</a>; Sun et al.,
 * "You Only Cache Once: Decoder-Decoder Architectures for Language Models",
 * <a href="https://arxiv.org/abs/2405.05254">arXiv:2405.05254</a>.
 */
public final class CrossLayerKvBlock implements Layer {

    /** How layers share keys and values. */
    public enum Sharing {
        /** Every layer has its own keys and values. */
        NONE,
        /** Groups of adjacent layers share one key/value projection. */
        CLA,
        /** A sliding-window lower half, then one global key/value set for the upper half. */
        YOCO
    }

    private final int features;
    private final int heads;
    private final int depth;
    private final Sharing sharing;
    private final int group;
    private final int window;
    private final float rotaryBase;
    private final List<Var[]> layers = new ArrayList<>();
    private final List<GluFeedForward> ffns = new ArrayList<>();
    private final Var globalKey;
    private final Var globalValue;
    private final Var globalGain;

    /**
     * Creates the stack.
     *
     * @param features the width
     * @param heads the attention heads
     * @param depth the number of layers
     * @param sharing the sharing scheme
     * @param group CLA: layers per shared key/value set; ignored otherwise
     * @param window YOCO: the self-decoder's sliding window; ignored otherwise
     * @param rotaryBase the RoPE base
     * @param random the random source
     */
    public CrossLayerKvBlock(int features, int heads, int depth, Sharing sharing, int group, int window,
            float rotaryBase, RandomGenerator random) {
        if (sharing == Sharing.CLA && group < 1 || sharing == Sharing.YOCO && (depth < 2 || window < 1)) {
            throw new IllegalArgumentException("CLA needs group >= 1, YOCO depth >= 2 and a window");
        }
        this.features = features;
        this.heads = heads;
        this.depth = depth;
        this.sharing = sharing;
        this.group = group;
        this.window = window;
        this.rotaryBase = rotaryBase;
        for (int l = 0; l < depth; l++) {
            boolean ownKv = ownsKv(l);
            layers.add(new Var[] {MixerOps.weight(random, features, features),
                ownKv ? MixerOps.weight(random, features, features) : null,
                ownKv ? MixerOps.weight(random, features, features) : null,
                MixerOps.weight(random, features, features), RefinementOps.ones(features),
                RefinementOps.ones(features)});
            ffns.add(new GluFeedForward(GluFeedForward.Kind.SWIGLU, features, 2 * features, random));
        }
        boolean yoco = sharing == Sharing.YOCO;
        globalKey = yoco ? MixerOps.weight(random, features, features) : null;
        globalValue = yoco ? MixerOps.weight(random, features, features) : null;
        globalGain = yoco ? RefinementOps.ones(features) : null;
    }

    /**
     * Describes the stack for {@code Model.sequential().add(...)}.
     *
     * @param heads the attention heads
     * @param depth the number of layers
     * @param sharing the sharing scheme
     * @param groupOrWindow the CLA group size or the YOCO window
     * @return the spec
     */
    public static LayerSpec of(int heads, int depth, Sharing sharing, int groupOrWindow) {
        return MixerOps.sameShape((in, r) -> new CrossLayerKvBlock(in.dim(-1), heads, depth, sharing,
                groupOrWindow, groupOrWindow, 10000f, r));
    }

    private boolean ownsKv(int layer) {
        return switch (sharing) {
            case NONE -> true;
            case CLA -> layer % group == 0;
            case YOCO -> layer < depth / 2;
        };
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "CrossLayerKvBlock");
        Var x = input;
        Var k = null;
        Var v = null;
        for (int l = 0; l < depth; l++) {
            Var[] w = layers.get(l);
            if (sharing == Sharing.YOCO && l == depth / 2) {
                Var g = RefinementOps.rms(x, globalGain);
                k = MixerOps.split(g.matmul(globalKey), heads);
                v = MixerOps.split(g.matmul(globalValue), heads);
            }
            Var n = RefinementOps.rms(x, w[4]);
            if (w[1] != null) {
                k = MixerOps.split(n.matmul(w[1]), heads);
                v = MixerOps.split(n.matmul(w[2]), heads);
            }
            Var q = MixerOps.split(n.matmul(w[0]), heads);
            int win = sharing == Sharing.YOCO && l < depth / 2 ? window : 0;
            x = x.add(MixerOps.merge(RefinementOps.causalAttention(q, k, v, rotaryBase, win)).matmul(w[3]));
            x = x.add(ffns.get(l).forward(RefinementOps.rms(x, w[5])));
        }
        return x;
    }

    /**
     * Counts the cached numbers per token that grow with the context, over all layers: keys
     * plus values of every distinct key/value set (sliding-window caches have a fixed size and
     * are not counted).
     *
     * @return the count; {@code 2 * features * depth} for the unshared stack
     */
    public int cachedValuesPerToken() {
        int sets = switch (sharing) {
            case NONE -> depth;
            case CLA -> (depth + group - 1) / group;
            case YOCO -> 1;
        };
        return 2 * features * sets;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int l = 0; l < depth; l++) {
            for (Var w : layers.get(l)) {
                if (w != null) {
                    p.add(w);
                }
            }
            p.addAll(ffns.get(l).parameters());
        }
        if (globalKey != null) {
            p.addAll(List.of(globalKey, globalValue, globalGain));
        }
        return p;
    }

    @Override
    public String toString() {
        return "CrossLayerKvBlock(" + features + ", depth=" + depth + ", " + sharing + ")";
    }
}
