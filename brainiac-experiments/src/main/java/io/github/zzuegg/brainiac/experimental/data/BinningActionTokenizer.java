package io.github.zzuegg.brainiac.experimental.data;

/**
 * The usual per-step action tokenization of RT-2 and OpenVLA: every action value of every
 * step becomes one token, its bin among {@code bins} uniform bins over {@code [-1, 1]}. A
 * chunk of H steps of D values always costs {@code H * D} tokens; the baseline that
 * {@link FastActionTokenizer} compresses.
 */
public final class BinningActionTokenizer implements ActionTokenizer {
    private final int horizon;
    private final int actionSize;
    private final int bins;

    /**
     * Creates the tokenizer.
     *
     * @param horizon the chunk length
     * @param actionSize the action size
     * @param bins the number of bins per value
     */
    public BinningActionTokenizer(int horizon, int actionSize, int bins) {
        if (horizon < 1 || actionSize < 1 || bins < 2) {
            throw new IllegalArgumentException("bad binning tokenizer size");
        }
        this.horizon = horizon;
        this.actionSize = actionSize;
        this.bins = bins;
    }

    @Override
    public int[] encodeChunk(float[][] chunk) {
        int[] out = new int[horizon * actionSize];
        for (int t = 0; t < horizon; t++) {
            for (int d = 0; d < actionSize; d++) {
                double u = (Math.max(-1, Math.min(1, chunk[t][d])) + 1) / 2;
                out[t * actionSize + d] = (int) Math.min(bins - 1, Math.floor(u * bins));
            }
        }
        return out;
    }

    @Override
    public float[][] decodeChunk(int[] tokens) {
        float[][] out = new float[horizon][actionSize];
        for (int i = 0; i < horizon * actionSize; i++) {
            int b = i < tokens.length ? Math.max(0, Math.min(bins - 1, tokens[i])) : bins / 2;
            out[i / actionSize][i % actionSize] = (float) ((b + 0.5) / bins * 2 - 1);
        }
        return out;
    }

    @Override
    public int size() {
        return bins;
    }

    @Override
    public int horizon() {
        return horizon;
    }

    @Override
    public int actionSize() {
        return actionSize;
    }
}
