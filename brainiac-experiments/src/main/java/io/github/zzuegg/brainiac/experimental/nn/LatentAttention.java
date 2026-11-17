package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Multi-head latent attention (MLA), the attention of DeepSeek-V2 and V3. Keys and values
 * of all heads are reconstructed from one small latent vector per token,
 * {@code c = rms(x W_dkv)} with {@code latent << 2 * features}:
 *
 * <pre>
 * k_h = [c W_uk,h ; rope(x W_kr)]      v_h = c W_uv,h      q_h = [x W_q,h ; rope(x W_qr,h)]
 * </pre>
 *
 * <p>Position goes through a separate, "decoupled" RoPE part: a small key {@code x W_kr}
 * shared by all heads, and per-head query parts. Rotating the compressed keys themselves
 * would stop the up-projection {@code W_uk} from being absorbed into the queries at
 * inference, so RoPE gets its own few dimensions. The KV cache ({@link Cache}) stores only
 * {@code c} and the shared rotary key: {@code latent + ropeDim} numbers per token and layer,
 * against {@code 2 * features} for multi-head attention and {@code 2 * kvHeads * headSize}
 * for grouped-query attention, while every head keeps its own keys and values.
 *
 * <p>Input and output are {@code [batch, steps, features]}, causal.
 *
 * <p>Paper: DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
 * Language Model", <a href="https://arxiv.org/abs/2405.04434">arXiv:2405.04434</a>.
 */
public final class LatentAttention implements Layer {

    /**
     * The compressed KV cache of one layer: the latents and the shared rotary keys of the
     * steps seen so far.
     */
    public static final class Cache {
        private Tensor latents;
        private Tensor ropeKeys;

        /** Creates an empty cache. */
        public Cache() {
        }

        /** @return the number of cached steps */
        public int length() {
            return latents == null ? 0 : latents.shape().dim(1);
        }

        /** @return the cached numbers per step of one sequence: latent plus rotary key */
        public int valuesPerStep() {
            return latents == null ? 0 : latents.shape().dim(2) + ropeKeys.shape().dim(3);
        }

        /** Forgets every cached step. */
        public void clear() {
            latents = null;
            ropeKeys = null;
        }
    }

    private final int features;
    private final int heads;
    private final int headSize;
    private final int latent;
    private final int ropeDim;
    private final float rotaryBase;
    private final Var wq;
    private final Var wqr;
    private final Var wdkv;
    private final Var latentGain;
    private final Var wuk;
    private final Var wuv;
    private final Var wkr;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param features the model width
     * @param heads the number of heads, dividing {@code features}
     * @param latent the latent (cache) width per token
     * @param ropeDim the decoupled rotary width, even
     * @param rotaryBase the RoPE base
     * @param random the random source
     */
    public LatentAttention(int features, int heads, int latent, int ropeDim, float rotaryBase,
            RandomGenerator random) {
        if (features % heads != 0 || ropeDim % 2 != 0 || latent < 1) {
            throw new IllegalArgumentException("need heads | features, an even rope width and a latent");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.latent = latent;
        this.ropeDim = ropeDim;
        this.rotaryBase = rotaryBase;
        wq = MixerOps.weight(random, features, features);
        wqr = MixerOps.weight(random, features, heads * ropeDim);
        wdkv = MixerOps.weight(random, features, latent);
        latentGain = RefinementOps.ones(latent);
        wuk = MixerOps.weight(random, latent, features);
        wuv = MixerOps.weight(random, latent, features);
        wkr = MixerOps.weight(random, features, ropeDim);
        wo = MixerOps.weight(random, features, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}, reading the width from the
     * input {@code [steps, features]}.
     *
     * @param heads the number of heads
     * @param latent the latent width
     * @param ropeDim the decoupled rotary width
     * @return the spec
     */
    public static LayerSpec of(int heads, int latent, int ropeDim) {
        return MixerOps.sameShape((in, r) -> new LatentAttention(in.dim(-1), heads, latent, ropeDim, 10000f, r));
    }

    @Override
    public Var forward(Var input) {
        return attend(input, null, 0);
    }

    /**
     * Runs new steps against a cache and appends their latents to it, for incremental
     * decoding; the result equals the matching rows of a full {@link #forward(Var)}.
     *
     * @param input the new steps {@code [batch, steps, features]}
     * @param cache the cache of the earlier steps, updated in place
     * @return {@code [batch, steps, features]}
     */
    public Var forward(Var input, Cache cache) {
        return attend(input, cache, cache.length());
    }

    private Var attend(Var input, Cache cache, int offset) {
        MixerOps.requireSequence(input, features, "LatentAttention");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var c = RefinementOps.rms(input.matmul(wdkv), latentGain);                         // [b, t, latent]
        Var kr = Functional.rotary(input.matmul(wkr).reshape(b, 1, t, ropeDim), rotaryBase, offset);
        if (cache != null) {
            if (cache.latents != null) {
                c = Var.concat(1, Var.of(cache.latents), c);
                kr = Var.concat(2, Var.of(cache.ropeKeys), kr);
            }
            cache.latents = c.value();
            cache.ropeKeys = kr.value();
        }
        int n = c.shape().dim(1);
        Var qc = MixerOps.split(input.matmul(wq), heads);                                   // [b, h, t, hs]
        Var qr = Functional.rotary(MixerOps.split(input.matmul(wqr), heads), rotaryBase, offset);
        Var kc = MixerOps.split(c.matmul(wuk), heads);                                      // [b, h, n, hs]
        Var v = MixerOps.split(c.matmul(wuv), heads);
        Var q = Var.concat(-1, qc, qr);
        Var k = Var.concat(-1, kc, kr.expand(Shape.of(b, heads, n, ropeDim)));
        Tensor mask = RefinementOps.additive(RefinementOps.keep(t, n, 0, offset, true, 0, input.value().device()));
        Var o = TechniqueOps.attend(q, k, v, null, mask, (float) (1 / Math.sqrt(headSize + ropeDim)));
        return MixerOps.merge(o).matmul(wo);
    }

    /** @return the cached numbers per token: {@code latent + ropeDim} */
    public int cacheValuesPerToken() {
        return latent + ropeDim;
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wqr, wdkv, latentGain, wuk, wuv, wkr, wo);
    }

    @Override
    public String toString() {
        return "LatentAttention(" + features + ", " + heads + " heads, latent=" + latent + ", rope=" + ropeDim + ")";
    }
}
