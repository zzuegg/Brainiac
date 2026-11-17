package io.github.zzuegg.brainiac.experimental.rl;

import java.util.SplittableRandom;

/**
 * A policy that outputs an <em>action chunk</em>, the next {@link #horizon()} actions at
 * once, as the robot policies of the last years do (ACT, Diffusion Policy, pi0). Executors in
 * {@link ActionChunking} and {@link RealTimeChunking} turn a chunk policy into a core
 * {@link io.github.zzuegg.brainiac.rl.Agent}.
 */
public interface ChunkPolicy {

    /** @return how many actions a chunk holds */
    int horizon();

    /** @return the size of one action */
    int actionSize();

    /**
     * Predicts a chunk.
     *
     * @param observation the current observation
     * @param random the source of sampling noise
     * @return {@code [horizon][actionSize]}
     */
    float[][] chunk(float[] observation, SplittableRandom random);

    /**
     * Predicts a chunk that continues actions already committed to, for real-time chunking:
     * {@code previous[i]} is what the running chunk does at this chunk's step {@code i}. The
     * first {@code frozen} of them will certainly be executed (they run while this chunk is
     * computed); the rest are a soft target. The default ignores them; policies that can
     * inpaint override it.
     *
     * @param observation the observation when the chunk was requested
     * @param previous the committed actions, aligned to this chunk's steps, at most
     *     {@link #horizon()} long
     * @param frozen how many leading ones are fixed
     * @param random the source of sampling noise
     * @return {@code [horizon][actionSize]}
     */
    default float[][] chunk(float[] observation, float[][] previous, int frozen, SplittableRandom random) {
        return chunk(observation, random);
    }
}
