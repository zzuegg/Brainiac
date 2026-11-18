package io.github.zzuegg.brainiac.experimental.vision;

/**
 * Optical-flow and disparity helpers on host arrays: the average end-point error (the
 * Euclidean distance between estimated and true flow vectors, averaged over pixels), the
 * share of pixels whose error exceeds a threshold, bilinear backward warping with border
 * clamping, and 2x pyramid down- and upsampling. A flow field of a {@code w x h} image is
 * {@code 2 w h} values: all {@code u} (horizontal) components, then all {@code v}.
 */
public final class FlowMetrics {

    private FlowMetrics() {
    }

    /**
     * The average end-point error.
     *
     * @param flow the estimate, {@code 2 n} values
     * @param truth the true flow, {@code 2 n} values
     * @return the mean of {@code |flow - truth|} over the {@code n} pixels
     */
    public static double endPointError(float[] flow, float[] truth) {
        int n = flow.length / 2;
        double s = 0;
        for (int i = 0; i < n; i++) {
            s += Math.hypot(flow[i] - truth[i], flow[n + i] - truth[n + i]);
        }
        return s / n;
    }

    /**
     * The average absolute error of a scalar field such as a disparity map.
     *
     * @param estimate the estimate
     * @param truth the truth
     * @return the mean absolute error
     */
    public static double meanAbsoluteError(float[] estimate, float[] truth) {
        double s = 0;
        for (int i = 0; i < estimate.length; i++) {
            s += Math.abs(estimate[i] - truth[i]);
        }
        return s / estimate.length;
    }

    /**
     * The share of entries whose absolute error exceeds a threshold, such as the "bad
     * pixel" rate of stereo benchmarks.
     *
     * @param estimate the estimate
     * @param truth the truth
     * @param threshold the error threshold
     * @return the share in [0, 1]
     */
    public static double badShare(float[] estimate, float[] truth, double threshold) {
        int bad = 0;
        for (int i = 0; i < estimate.length; i++) {
            bad += Math.abs(estimate[i] - truth[i]) > threshold ? 1 : 0;
        }
        return (double) bad / estimate.length;
    }

    /**
     * Bilinear sample with the coordinates clamped to the image.
     *
     * @param img the image, {@code w * h}
     * @param w the width
     * @param h the height
     * @param x the column
     * @param y the row
     * @return the interpolated value
     */
    public static float sample(float[] img, int w, int h, float x, float y) {
        x = Math.max(0, Math.min(w - 1, x));
        y = Math.max(0, Math.min(h - 1, y));
        int x0 = Math.min(w - 2, (int) x);
        int y0 = Math.min(h - 2, (int) y);
        x0 = Math.max(0, x0);
        y0 = Math.max(0, y0);
        float fx = Math.min(1, x - x0);
        float fy = Math.min(1, y - y0);
        int x1 = Math.min(w - 1, x0 + 1);
        int y1 = Math.min(h - 1, y0 + 1);
        return (1 - fy) * ((1 - fx) * img[y0 * w + x0] + fx * img[y0 * w + x1])
                + fy * ((1 - fx) * img[y1 * w + x0] + fx * img[y1 * w + x1]);
    }

    /**
     * Backward warping: output pixel {@code (x, y)} reads {@code img} at
     * {@code (x + u, y + v)}, so warping the second frame by the first frame's flow
     * reconstructs the first.
     *
     * @param img the image, {@code w * h}
     * @param flow the flow, {@code 2 w h}
     * @param w the width
     * @param h the height
     * @return the warped image
     */
    public static float[] warp(float[] img, float[] flow, int w, int h) {
        int n = w * h;
        float[] out = new float[n];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = y * w + x;
                out[i] = sample(img, w, h, x + flow[i], y + flow[n + i]);
            }
        }
        return out;
    }

    /**
     * Halves an image by 2x2 averaging (odd edges are dropped).
     *
     * @param img the image
     * @param w the width
     * @param h the height
     * @return the {@code (w / 2) x (h / 2)} image
     */
    public static float[] down(float[] img, int w, int h) {
        int w2 = w / 2;
        int h2 = h / 2;
        float[] out = new float[w2 * h2];
        for (int y = 0; y < h2; y++) {
            for (int x = 0; x < w2; x++) {
                out[y * w2 + x] = 0.25f * (img[2 * y * w + 2 * x] + img[2 * y * w + 2 * x + 1]
                        + img[(2 * y + 1) * w + 2 * x] + img[(2 * y + 1) * w + 2 * x + 1]);
            }
        }
        return out;
    }

    /**
     * Doubles a flow field (bilinear, with the vectors doubled too).
     *
     * @param flow the flow of a {@code w x h} image
     * @param w the width
     * @param h the height
     * @param w2 the new width
     * @param h2 the new height
     * @return the flow of the {@code w2 x h2} image
     */
    public static float[] upFlow(float[] flow, int w, int h, int w2, int h2) {
        int n = w * h;
        int n2 = w2 * h2;
        float[] u = java.util.Arrays.copyOfRange(flow, 0, n);
        float[] v = java.util.Arrays.copyOfRange(flow, n, 2 * n);
        float[] out = new float[2 * n2];
        float sx = (float) w / w2;
        float sy = (float) h / h2;
        for (int y = 0; y < h2; y++) {
            for (int x = 0; x < w2; x++) {
                float xs = (x + 0.5f) * sx - 0.5f;
                float ys = (y + 0.5f) * sy - 0.5f;
                out[y * w2 + x] = 2 * sample(u, w, h, xs, ys);
                out[n2 + y * w2 + x] = 2 * sample(v, w, h, xs, ys);
            }
        }
        return out;
    }
}
