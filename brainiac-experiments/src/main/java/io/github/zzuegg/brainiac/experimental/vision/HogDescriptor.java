package io.github.zzuegg.brainiac.experimental.vision;

/**
 * Histograms of oriented gradients (Dalal &amp; Triggs, CVPR 2005): centred-difference
 * gradients, unsigned orientations binned with linear interpolation (between bins) weighted
 * by magnitude into square cells, overlapping blocks of 2 x 2 cells normalised with L2-Hys
 * (L2, clip at 0.2, L2 again), and all blocks concatenated.
 */
public final class HogDescriptor {

    private final int side;
    private final int cell;
    private final int bins;

    /**
     * Creates the descriptor for square patches.
     *
     * @param side the patch side
     * @param cell the cell side (dividing {@code side})
     * @param bins the orientation bins over 0 to 180 degrees
     */
    public HogDescriptor(int side, int cell, int bins) {
        if (side % cell != 0) {
            throw new IllegalArgumentException("cell must divide side");
        }
        this.side = side;
        this.cell = cell;
        this.bins = bins;
    }

    /** @return the descriptor length */
    public int size() {
        int c = side / cell - 1;
        return c * c * 4 * bins;
    }

    /**
     * Computes the descriptor of a patch.
     *
     * @param patch {@code side * side} pixels, row-major
     * @return the descriptor
     */
    public double[] compute(float[] patch) {
        int cells = side / cell;
        double[][][] hist = new double[cells][cells][bins];
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                double gx = at(patch, x + 1, y) - at(patch, x - 1, y);
                double gy = at(patch, x, y + 1) - at(patch, x, y - 1);
                double mag = Math.hypot(gx, gy);
                if (mag == 0) {
                    continue;
                }
                double angle = Math.atan2(gy, gx);
                if (angle < 0) {
                    angle += Math.PI;
                }
                double pos = angle / Math.PI * bins - 0.5;
                int b0 = (int) Math.floor(pos);
                double frac = pos - b0;
                int lo = Math.floorMod(b0, bins);
                int hi = Math.floorMod(b0 + 1, bins);
                hist[y / cell][x / cell][lo] += mag * (1 - frac);
                hist[y / cell][x / cell][hi] += mag * frac;
            }
        }
        double[] out = new double[size()];
        int k = 0;
        for (int by = 0; by < cells - 1; by++) {
            for (int bx = 0; bx < cells - 1; bx++) {
                int start = k;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        for (int b = 0; b < bins; b++) {
                            out[k++] = hist[by + dy][bx + dx][b];
                        }
                    }
                }
                normalise(out, start, k);
                for (int i = start; i < k; i++) {
                    out[i] = Math.min(out[i], 0.2);
                }
                normalise(out, start, k);
            }
        }
        return out;
    }

    private static void normalise(double[] v, int from, int to) {
        double s = 1e-6;
        for (int i = from; i < to; i++) {
            s += v[i] * v[i];
        }
        s = Math.sqrt(s);
        for (int i = from; i < to; i++) {
            v[i] /= s;
        }
    }

    private double at(float[] p, int x, int y) {
        x = Math.max(0, Math.min(side - 1, x));
        y = Math.max(0, Math.min(side - 1, y));
        return p[y * side + x];
    }
}
