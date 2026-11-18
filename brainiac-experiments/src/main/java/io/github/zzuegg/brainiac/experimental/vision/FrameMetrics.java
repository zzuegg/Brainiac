package io.github.zzuegg.brainiac.experimental.vision;

/**
 * Image-quality metrics for predicted frames and renders: mean squared error, PSNR for
 * signals in [0, 1], and the structural similarity index (Wang et al., 2004) over 7x7
 * windows with the usual constants {@code C1 = 0.01^2}, {@code C2 = 0.03^2}.
 */
public final class FrameMetrics {

    private FrameMetrics() {
    }

    /**
     * The mean squared error.
     *
     * @param a one image
     * @param b another of the same size
     * @return the MSE
     */
    public static double mse(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return s / a.length;
    }

    /**
     * The peak signal-to-noise ratio for values in [0, 1], estimates clamped to that range.
     *
     * @param estimate the estimate
     * @param truth the truth
     * @return the PSNR in dB
     */
    public static double psnr(float[] estimate, float[] truth) {
        double s = 0;
        for (int i = 0; i < estimate.length; i++) {
            double d = Math.max(0, Math.min(1, estimate[i])) - truth[i];
            s += d * d;
        }
        return 10 * Math.log10(estimate.length / Math.max(s, 1e-12));
    }

    /**
     * The mean structural similarity over all 7x7 windows inside the image.
     *
     * @param a one image, {@code w * h}
     * @param b another
     * @param w the width, at least 7
     * @param h the height, at least 7
     * @return SSIM, 1 for identical images
     */
    public static double ssim(float[] a, float[] b, int w, int h) {
        double c1 = 1e-4;
        double c2 = 9e-4;
        double total = 0;
        int windows = 0;
        int k = 7;
        for (int y = 0; y + k <= h; y++) {
            for (int x = 0; x + k <= w; x++) {
                double ma = 0;
                double mb = 0;
                for (int dy = 0; dy < k; dy++) {
                    for (int dx = 0; dx < k; dx++) {
                        ma += a[(y + dy) * w + x + dx];
                        mb += b[(y + dy) * w + x + dx];
                    }
                }
                int n = k * k;
                ma /= n;
                mb /= n;
                double va = 0;
                double vb = 0;
                double cov = 0;
                for (int dy = 0; dy < k; dy++) {
                    for (int dx = 0; dx < k; dx++) {
                        double da = a[(y + dy) * w + x + dx] - ma;
                        double db = b[(y + dy) * w + x + dx] - mb;
                        va += da * da;
                        vb += db * db;
                        cov += da * db;
                    }
                }
                va /= n - 1;
                vb /= n - 1;
                cov /= n - 1;
                total += (2 * ma * mb + c1) * (2 * cov + c2) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
                windows++;
            }
        }
        return total / windows;
    }
}
