package io.github.zzuegg.brainiac.experimental.llm;

import java.util.Arrays;

/**
 * Two training-free fixes for "lost in the middle" (Liu et al., TACL 2024,
 * <a href="https://arxiv.org/abs/2307.03172">arXiv:2307.03172</a>): LMs use information at the
 * start and end of a long context better than in the middle, because attention carries a
 * positional bias on top of relevance.
 * <ul>
 * <li>{@link #foundInTheMiddle} (Hsieh et al. 2024,
 *     <a href="https://arxiv.org/abs/2406.16008">arXiv:2406.16008</a>): estimate the bias by
 *     placing an irrelevant (dummy) document at every position and recording the attention it
 *     gets; the calibrated relevance of a document is its attention minus the bias at its
 *     position.</li>
 * <li>{@link #multiScalePositions} (Ms-PoE, Zhang et al. 2024,
 *     <a href="https://arxiv.org/abs/2403.04797">arXiv:2403.04797</a>): rescale the position
 *     indices fed to RoPE by a different ratio per head (from {@code minRatio} to
 *     {@code maxRatio}, heads that are more position-aware getting the smaller ratios), so some
 *     heads see distant middle tokens at a "nearer" scale.</li>
 * </ul>
 * {@link #rotate} applies RoPE to one vector at a (possibly fractional) position with the core
 * {@code Rope}'s inverse frequencies.
 */
public final class PositionBiasCalibration {

    private final double[] ratios;

    private PositionBiasCalibration(double[] ratios) {
        this.ratios = ratios;
    }

    /**
     * Ms-PoE position scaling.
     *
     * @param heads the heads
     * @param minRatio the smallest ratio (1.2 in the paper)
     * @param maxRatio the largest ratio (1.8 in the paper)
     * @param awareness per head a position-awareness score (higher gets a smaller ratio), or
     *     {@code null} to assign ratios in head order
     * @return the scaling
     */
    public static PositionBiasCalibration multiScalePositions(int heads, double minRatio, double maxRatio,
            double[] awareness) {
        if (heads < 1 || minRatio <= 0 || maxRatio < minRatio) {
            throw new IllegalArgumentException("need heads >= 1 and 0 < minRatio <= maxRatio");
        }
        double[] ratios = new double[heads];
        Integer[] order = new Integer[heads];
        for (int h = 0; h < heads; h++) {
            order[h] = h;
        }
        if (awareness != null) {
            Arrays.sort(order, (a, b) -> Double.compare(awareness[b], awareness[a]));
        }
        for (int i = 0; i < heads; i++) {
            ratios[order[i]] = heads == 1 ? minRatio : minRatio + (maxRatio - minRatio) * i / (heads - 1);
        }
        return new PositionBiasCalibration(ratios);
    }

    /**
     * @param head a head
     * @return its position ratio
     */
    public double ratio(int head) {
        return ratios[head];
    }

    /**
     * @param position a position index
     * @param head the head
     * @return the rescaled position fed to RoPE
     */
    public double position(int position, int head) {
        return position / ratios[head];
    }

    /**
     * Found-in-the-middle: the positional bias, the mean attention an irrelevant document gets
     * at each position.
     *
     * @param dummyAttention per trial, the attention a dummy document received at each position
     * @return the bias per position
     */
    public static double[] foundInTheMiddle(double[][] dummyAttention) {
        int positions = dummyAttention[0].length;
        double[] bias = new double[positions];
        for (double[] trial : dummyAttention) {
            for (int p = 0; p < positions; p++) {
                bias[p] += trial[p] / dummyAttention.length;
            }
        }
        return bias;
    }

    /**
     * Calibrated relevance: attention minus the positional bias.
     *
     * @param attention the attention each document (in position order) received
     * @param bias the bias per position
     * @return the calibrated relevance
     */
    public static double[] calibrate(double[] attention, double[] bias) {
        double[] out = new double[attention.length];
        for (int p = 0; p < out.length; p++) {
            out[p] = attention[p] - bias[p];
        }
        return out;
    }

    /**
     * Rotary position embedding of one vector: pairs {@code (2i, 2i + 1)} rotate by
     * {@code position * invFreq[i]}.
     *
     * @param x the vector
     * @param position the position (may be fractional)
     * @param invFreq the inverse frequencies, such as {@code Rope.frequencies()}
     * @return the rotated vector
     */
    public static float[] rotate(float[] x, double position, double[] invFreq) {
        float[] out = x.clone();
        for (int i = 0; i < invFreq.length && 2 * i + 1 < x.length; i++) {
            double a = position * invFreq[i];
            double c = Math.cos(a);
            double s = Math.sin(a);
            out[2 * i] = (float) (x[2 * i] * c - x[2 * i + 1] * s);
            out[2 * i + 1] = (float) (x[2 * i] * s + x[2 * i + 1] * c);
        }
        return out;
    }

    @Override
    public String toString() {
        return "Ms-PoE" + Arrays.toString(ratios);
    }
}
