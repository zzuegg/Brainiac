package io.github.zzuegg.brainiac.experimental.data;

/**
 * Turns robot action chunks {@code [horizon][actionSize]} into discrete tokens and back, so
 * that an autoregressive sequence model can be a policy (RT-2, OpenVLA, pi0-FAST). See
 * {@link FastActionTokenizer} (frequency-space compression) and
 * {@link BinningActionTokenizer} (one token per action value).
 */
public interface ActionTokenizer {

    /**
     * Tokenizes a chunk.
     *
     * @param chunk {@code [horizon][actionSize]}
     * @return the tokens, each in {@code 0..size()-1}
     */
    int[] encodeChunk(float[][] chunk);

    /**
     * Decodes tokens; any sequence decodes, missing values are filled in.
     *
     * @param tokens the tokens
     * @return {@code [horizon][actionSize]}
     */
    float[][] decodeChunk(int[] tokens);

    /** @return the number of distinct tokens */
    int size();

    /** @return the chunk length */
    int horizon();

    /** @return the action size */
    int actionSize();
}
