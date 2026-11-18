package io.github.zzuegg.brainiac.experimental.timeseries;

/**
 * Chronos's value tokenizer (Ansari et al., 2024, <a href="https://arxiv.org/abs/2403.07815">arXiv:2403.07815</a>):
 * a series is divided by its context's mean absolute value (mean scaling), and the scaled
 * values are quantised into {@code bins} uniform bins on {@code [low, high]}; decoding maps a
 * token to its bin centre times the scale. Forecasting then becomes next-token prediction
 * with cross-entropy, and sampled token paths give probabilistic forecasts. (Core
 * {@code data.Tokenizer} maps text to ids, so this tokenizer has its own small interface.)
 */
public final class ValueTokenizer {

    private final int bins;
    private final double low;
    private final double high;

    /**
     * Creates the tokenizer.
     *
     * @param bins the vocabulary size
     * @param low the smallest scaled value represented
     * @param high the largest scaled value represented
     */
    public ValueTokenizer(int bins, double low, double high) {
        this.bins = bins;
        this.low = low;
        this.high = high;
    }

    /** @return the vocabulary size */
    public int vocabulary() {
        return bins;
    }

    /**
     * The mean-scaling factor of a context.
     *
     * @param context the observed values
     * @return their mean absolute value (1 if it is 0)
     */
    public static double scale(float[] context) {
        double s = 0;
        for (float v : context) {
            s += Math.abs(v);
        }
        s /= Math.max(1, context.length);
        return s > 1e-8 ? s : 1;
    }

    /**
     * Encodes values.
     *
     * @param values the values
     * @param scale the mean-scaling factor
     * @return token ids in {@code [0, bins)}
     */
    public int[] encode(float[] values, double scale) {
        int[] ids = new int[values.length];
        double width = (high - low) / bins;
        for (int i = 0; i < values.length; i++) {
            double v = values[i] / scale;
            ids[i] = (int) Math.max(0, Math.min(bins - 1, Math.floor((v - low) / width)));
        }
        return ids;
    }

    /**
     * Decodes one token.
     *
     * @param id a token id
     * @param scale the mean-scaling factor
     * @return the bin centre on the original scale
     */
    public double decode(int id, double scale) {
        double width = (high - low) / bins;
        return (low + (id + 0.5) * width) * scale;
    }
}
