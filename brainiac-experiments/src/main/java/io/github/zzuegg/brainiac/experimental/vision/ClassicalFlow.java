package io.github.zzuegg.brainiac.experimental.vision;

/**
 * Training-free optical flow from the brightness-constancy equation
 * {@code I_x u + I_y v + I_t = 0}, which alone fixes only the flow component along the
 * image gradient (the aperture problem); each method adds an assumption:
 *
 * <ul>
 *   <li>{@link #hornSchunck}: a global smoothness term, minimised by Jacobi iterations
 *       {@code u = ubar - I_x (I_x ubar + I_y vbar + I_t) / (alpha^2 + I_x^2 + I_y^2)}, so
 *       textureless interiors are filled in from their edges;</li>
 *   <li>{@link #lucasKanade}: constant flow in a window, solved per pixel as a 2x2 least
 *       squares system, left at zero where the system is ill-conditioned (textureless
 *       windows).</li>
 * </ul>
 *
 * Both run coarse to fine on an image pyramid: the flow of a coarse level is doubled, the
 * second frame is warped by it ({@link FlowMetrics#warp}), and the finer level estimates
 * only the remaining increment, which lets small linearisations follow large motions
 * (Bouguet's pyramidal scheme).
 *
 * <p>Papers: Horn and Schunck, "Determining Optical Flow" (1981),
 * <a href="https://doi.org/10.1016/0004-3702(81)90024-2">doi:10.1016/0004-3702(81)90024-2</a>;
 * Lucas and Kanade, "An Iterative Image Registration Technique" (IJCAI 1981).
 */
public final class ClassicalFlow {

    private ClassicalFlow() {
    }

    /** One level's flow increment between a frame and the warped second frame. */
    private interface Solver {
        float[] solve(float[] a, float[] b, int w, int h);
    }

    /**
     * Horn-Schunck flow.
     *
     * @param first the first frame, {@code w * h}
     * @param second the second frame
     * @param w the width
     * @param h the height
     * @param alpha the smoothness weight, such as 0.5 for images in [0, 1]
     * @param iterations the Jacobi iterations per level and warp
     * @param levels the pyramid levels, 1 for a single scale
     * @return the flow from the first frame to the second, {@code 2 w h}
     */
    public static float[] hornSchunck(float[] first, float[] second, int w, int h, float alpha, int iterations,
            int levels) {
        return pyramid(first, second, w, h, levels, 2, (a, b, ww, hh) -> hsIncrement(a, b, ww, hh, alpha, iterations));
    }

    /**
     * Pyramidal Lucas-Kanade flow.
     *
     * @param first the first frame
     * @param second the second frame
     * @param w the width
     * @param h the height
     * @param radius the window radius (a {@code 2r+1} square)
     * @param levels the pyramid levels
     * @param iterations warp-and-solve iterations per level
     * @return the flow, {@code 2 w h}
     */
    public static float[] lucasKanade(float[] first, float[] second, int w, int h, int radius, int levels,
            int iterations) {
        return pyramid(first, second, w, h, levels, iterations, (a, b, ww, hh) -> lkIncrement(a, b, ww, hh, radius));
    }

    private static float[] pyramid(float[] first, float[] second, int w, int h, int levels, int warps, Solver s) {
        float[][] p1 = new float[levels][];
        float[][] p2 = new float[levels][];
        int[] ws = new int[levels];
        int[] hs = new int[levels];
        p1[0] = first;
        p2[0] = second;
        ws[0] = w;
        hs[0] = h;
        for (int l = 1; l < levels; l++) {
            p1[l] = FlowMetrics.down(p1[l - 1], ws[l - 1], hs[l - 1]);
            p2[l] = FlowMetrics.down(p2[l - 1], ws[l - 1], hs[l - 1]);
            ws[l] = ws[l - 1] / 2;
            hs[l] = hs[l - 1] / 2;
        }
        float[] flow = new float[2 * ws[levels - 1] * hs[levels - 1]];
        for (int l = levels - 1; l >= 0; l--) {
            if (l < levels - 1) {
                flow = FlowMetrics.upFlow(flow, ws[l + 1], hs[l + 1], ws[l], hs[l]);
            }
            for (int it = 0; it < warps; it++) {
                float[] warped = FlowMetrics.warp(p2[l], flow, ws[l], hs[l]);
                float[] d = s.solve(p1[l], warped, ws[l], hs[l]);
                for (int i = 0; i < flow.length; i++) {
                    flow[i] += d[i];
                }
            }
        }
        return flow;
    }

    /** Central-difference gradients of the average of two frames, and their difference. */
    private static float[][] derivatives(float[] a, float[] b, int w, int h) {
        int n = w * h;
        float[] ix = new float[n];
        float[] iy = new float[n];
        float[] it = new float[n];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = y * w + x;
                int xl = Math.max(0, x - 1);
                int xr = Math.min(w - 1, x + 1);
                int yu = Math.max(0, y - 1);
                int yd = Math.min(h - 1, y + 1);
                float gx = (a[y * w + xr] - a[y * w + xl] + b[y * w + xr] - b[y * w + xl]) / (2f * Math.max(1, xr - xl));
                float gy = (a[yd * w + x] - a[yu * w + x] + b[yd * w + x] - b[yu * w + x]) / (2f * Math.max(1, yd - yu));
                ix[i] = gx;
                iy[i] = gy;
                it[i] = b[i] - a[i];
            }
        }
        return new float[][] {ix, iy, it};
    }

    private static float[] hsIncrement(float[] a, float[] b, int w, int h, float alpha, int iterations) {
        int n = w * h;
        float[][] d = derivatives(a, b, w, h);
        float[] u = new float[n];
        float[] v = new float[n];
        float a2 = alpha * alpha;
        for (int it = 0; it < iterations; it++) {
            float[] nu = new float[n];
            float[] nv = new float[n];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int i = y * w + x;
                    float su = 0;
                    float sv = 0;
                    float wt = 0;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0) {
                                continue;
                            }
                            int xx = x + dx;
                            int yy = y + dy;
                            if (xx < 0 || yy < 0 || xx >= w || yy >= h) {
                                continue;
                            }
                            float k = dx == 0 || dy == 0 ? 1 / 6f : 1 / 12f;
                            su += k * u[yy * w + xx];
                            sv += k * v[yy * w + xx];
                            wt += k;
                        }
                    }
                    float ub = su / wt;
                    float vb = sv / wt;
                    float t = (d[0][i] * ub + d[1][i] * vb + d[2][i]) / (a2 + d[0][i] * d[0][i] + d[1][i] * d[1][i]);
                    nu[i] = ub - d[0][i] * t;
                    nv[i] = vb - d[1][i] * t;
                }
            }
            u = nu;
            v = nv;
        }
        float[] out = new float[2 * n];
        System.arraycopy(u, 0, out, 0, n);
        System.arraycopy(v, 0, out, n, n);
        return out;
    }

    private static float[] lkIncrement(float[] a, float[] b, int w, int h, int r) {
        int n = w * h;
        float[][] d = derivatives(a, b, w, h);
        float[] out = new float[2 * n];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double sxx = 0;
                double sxy = 0;
                double syy = 0;
                double sxt = 0;
                double syt = 0;
                for (int yy = Math.max(0, y - r); yy <= Math.min(h - 1, y + r); yy++) {
                    for (int xx = Math.max(0, x - r); xx <= Math.min(w - 1, x + r); xx++) {
                        int j = yy * w + xx;
                        sxx += d[0][j] * d[0][j];
                        sxy += d[0][j] * d[1][j];
                        syy += d[1][j] * d[1][j];
                        sxt += d[0][j] * d[2][j];
                        syt += d[1][j] * d[2][j];
                    }
                }
                double det = sxx * syy - sxy * sxy;
                double trace = sxx + syy;
                double minEig = trace / 2 - Math.sqrt(Math.max(0, trace * trace / 4 - det));
                if (minEig < 1e-3) {
                    continue;
                }
                int i = y * w + x;
                out[i] = (float) ((-syy * sxt + sxy * syt) / det);
                out[n + i] = (float) ((sxy * sxt - sxx * syt) / det);
            }
        }
        return out;
    }
}
