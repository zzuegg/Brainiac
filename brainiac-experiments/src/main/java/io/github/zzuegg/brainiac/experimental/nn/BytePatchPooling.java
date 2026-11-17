package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * The byte-to-patch bridge of a Byte Latent Transformer: {@link #pool} turns byte states into
 * one vector per dynamic patch (mean of the patch's bytes, projected to the latent width),
 * and {@link #unpool} hands each byte the latent state of the patch <em>before</em> its own,
 * so a byte decoder stays causal. Patch boundaries come from an entropy patcher (see
 * {@code experimental.data.EntropyPatcher}); the latent transformer in between runs once per
 * patch instead of once per byte.
 *
 * <p>The paper pools with cross-attention from patch queries to their bytes; a masked mean
 * followed by a linear map is its simplest special case (uniform attention weights).
 *
 * <p>Paper: Pagnoni et al., "Byte Latent Transformer: Patches Scale Better Than Tokens",
 * <a href="https://arxiv.org/abs/2412.09871">arXiv:2412.09871</a>.
 */
public final class BytePatchPooling {

    private final Var down;
    private final Var up;
    private final Var start;

    /**
     * Creates the bridge.
     *
     * @param byteWidth the byte state width
     * @param latentWidth the patch (latent) width
     * @param random the random source
     */
    public BytePatchPooling(int byteWidth, int latentWidth, RandomGenerator random) {
        this.down = MixerOps.weight(random, byteWidth, latentWidth);
        this.up = MixerOps.weight(random, latentWidth, byteWidth);
        this.start = MixerOps.uniform(random, 0.1f, 1, byteWidth);
    }

    /**
     * Pools bytes into patches.
     *
     * @param bytes byte states {@code [steps, byteWidth]}
     * @param patchIds the patch of each byte, non-decreasing from 0
     * @return patch vectors {@code [patches, latentWidth]}
     */
    public Var pool(Var bytes, int[] patchIds) {
        int patches = patchIds[patchIds.length - 1] + 1;
        float[] counts = new float[patches];
        for (int p : patchIds) {
            counts[p]++;
        }
        for (int i = 0; i < patches; i++) {
            counts[i] = 1 / counts[i];
        }
        Var sums = MixerOps.scatterAdd(bytes, 0, MixerOps.ids(patchIds), patches);
        return sums.mul(Tensor.of(Shape.of(patches, 1), counts)).matmul(down);
    }

    /**
     * Gives every byte the latent output of the previous patch (a learned start vector for
     * the first patch), projected back to the byte width.
     *
     * @param latents patch outputs {@code [patches, latentWidth]}; row {@code p} may depend on
     *     patches up to {@code p}
     * @param patchIds the patch of each byte
     * @return {@code [steps, byteWidth]}
     */
    public Var unpool(Var latents, int[] patchIds) {
        Var shifted = Var.concat(0, start, latents.matmul(up));   // row p + 1 holds patch p
        return shifted.gather(0, MixerOps.ids(patchIds));
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return List.of(down, up, start);
    }
}
