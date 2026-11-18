package io.github.zzuegg.brainiac.experimental.imaging;

/**
 * Classical edge-preserving smoothing filters on square images.
 *
 * <ul>
 *   <li>{@link Filter#BILATERAL} (Tomasi &amp; Manduchi, ICCV 1998): a neighbour average weighted
 *       by spatial distance and by intensity difference, so pixels across an edge barely
 *       contribute.</li>
 *   <li>{@link Filter#GUIDED} (He, Sun &amp; Tang, ECCV 2010): a local linear model
 *       {@code q = a I + b} of the output on a guide image fitted in every window, averaged;
 *       {@code O(1)} per pixel with box filters. {@link #guidedMasked} fits only on observed
 *       pixels, as residual interpolation for demosaicing needs.</li>
 *   <li>{@link Filter#PERONA_MALIK} (TPAMI 1990): anisotropic diffusion
 *       {@code x_t = div(g(|grad x|) grad x)} with {@code g(s) = exp(-(s/K)^2)}, smoothing inside
 *       regions but not across edges.</li>
 * </ul>
 */
public final class EdgePreservingFilters {

    /** The filters. */
    public enum Filter {
        /** The bilateral filter. */
        BILATERAL,
        /** The guided filter, self-guided. */
        GUIDED,
        /** Perona-Malik anisotropic diffusion. */
        PERONA_MALIK
    }

    private EdgePreservingFilters() {
    }

    /**
     * Applies a filter with settings derived from the noise level.
     *
     * @param filter the filter
     * @param x the image
     * @param side its side
     * @param sigma the noise standard deviation
     * @return the filtered image
     */
    public static double[] apply(Filter filter, double[] x, int side, double sigma) {
        return switch (filter) {
            case BILATERAL -> bilateral(x, side, 2, 1.5, 2.5 * sigma);
            case GUIDED -> guided(x, x, side, 2, 1.5 * sigma * sigma);
            case PERONA_MALIK -> peronaMalik(x, side, 2.5 * sigma, (int) Math.ceil(60 * sigma) + 5, 0.2);
        };
    }

    /**
     * The bilateral filter.
     *
     * @param x the image
     * @param side its side
     * @param radius the window radius
     * @param sigmaSpace the spatial standard deviation in pixels
     * @param sigmaRange the intensity standard deviation
     * @return the filtered image
     */
    public static double[] bilateral(double[] x, int side, int radius, double sigmaSpace, double sigmaRange) {
        double[] out = new double[x.length];
        double is = 1 / (2 * sigmaSpace * sigmaSpace);
        double ir = 1 / (2 * sigmaRange * sigmaRange);
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                double centre = x[r * side + c];
                double s = 0;
                double w = 0;
                for (int dy = -radius; dy <= radius; dy++) {
                    int rr = r + dy;
                    if (rr < 0 || rr >= side) {
                        continue;
                    }
                    for (int dx = -radius; dx <= radius; dx++) {
                        int cc = c + dx;
                        if (cc < 0 || cc >= side) {
                            continue;
                        }
                        double v = x[rr * side + cc];
                        double wt = Math.exp(-(dx * dx + dy * dy) * is - (v - centre) * (v - centre) * ir);
                        s += wt * v;
                        w += wt;
                    }
                }
                out[r * side + c] = s / w;
            }
        }
        return out;
    }

    /**
     * The guided filter.
     *
     * @param guide the guide image
     * @param p the image to filter
     * @param side the side
     * @param radius the window radius
     * @param eps the regulariser of the local slope
     * @return the filtered image
     */
    public static double[] guided(double[] guide, double[] p, int side, int radius, double eps) {
        int n = p.length;
        double[] mi = ImageMath.boxFilter(guide, side, radius);
        double[] mp = ImageMath.boxFilter(p, side, radius);
        double[] ip = new double[n];
        double[] ii = new double[n];
        for (int i = 0; i < n; i++) {
            ip[i] = guide[i] * p[i];
            ii[i] = guide[i] * guide[i];
        }
        double[] mip = ImageMath.boxFilter(ip, side, radius);
        double[] mii = ImageMath.boxFilter(ii, side, radius);
        double[] a = new double[n];
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            double cov = mip[i] - mi[i] * mp[i];
            double var = mii[i] - mi[i] * mi[i];
            a[i] = cov / (var + eps);
            b[i] = mp[i] - a[i] * mi[i];
        }
        double[] ma = ImageMath.boxFilter(a, side, radius);
        double[] mb = ImageMath.boxFilter(b, side, radius);
        double[] q = new double[n];
        for (int i = 0; i < n; i++) {
            q[i] = ma[i] * guide[i] + mb[i];
        }
        return q;
    }

    /**
     * The guided filter fitted only on observed pixels: each window's linear model
     * {@code p = a I + b} is fitted by least squares on the pixels where {@code observed} is
     * true, then applied everywhere (Kiku et al.'s residual interpolation).
     *
     * @param guide the guide, known everywhere
     * @param p the sparse image (values only where observed)
     * @param observed the mask
     * @param side the side
     * @param radius the window radius
     * @param eps the regulariser of the slope
     * @return the dense estimate
     */
    public static double[] guidedMasked(double[] guide, double[] p, boolean[] observed, int side, int radius,
            double eps) {
        int n = p.length;
        double[] m = new double[n];
        double[] mi = new double[n];
        double[] mp = new double[n];
        double[] mip = new double[n];
        double[] mii = new double[n];
        for (int i = 0; i < n; i++) {
            if (observed[i]) {
                m[i] = 1;
                mi[i] = guide[i];
                mp[i] = p[i];
                mip[i] = guide[i] * p[i];
                mii[i] = guide[i] * guide[i];
            }
        }
        double[] cm = ImageMath.boxFilter(m, side, radius);
        double[] ci = ImageMath.boxFilter(mi, side, radius);
        double[] cp = ImageMath.boxFilter(mp, side, radius);
        double[] cip = ImageMath.boxFilter(mip, side, radius);
        double[] cii = ImageMath.boxFilter(mii, side, radius);
        double[] a = new double[n];
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            double w = Math.max(cm[i], 1e-9);
            double ei = ci[i] / w;
            double ep = cp[i] / w;
            double cov = cip[i] / w - ei * ep;
            double var = cii[i] / w - ei * ei;
            a[i] = cov / (var + eps);
            b[i] = ep - a[i] * ei;
        }
        double[] ma = ImageMath.boxFilter(a, side, radius);
        double[] mb = ImageMath.boxFilter(b, side, radius);
        double[] q = new double[n];
        for (int i = 0; i < n; i++) {
            q[i] = ma[i] * guide[i] + mb[i];
        }
        return q;
    }

    /**
     * Perona-Malik diffusion with the exponential conductance and four neighbours.
     *
     * @param x the image
     * @param side its side
     * @param k the edge threshold
     * @param iterations the time steps
     * @param dt the step (at most 0.25)
     * @return the diffused image
     */
    public static double[] peronaMalik(double[] x, int side, double k, int iterations, double dt) {
        double[] u = x.clone();
        double ik = 1 / (k * k);
        for (int it = 0; it < iterations; it++) {
            double[] next = u.clone();
            for (int r = 0; r < side; r++) {
                for (int c = 0; c < side; c++) {
                    int i = r * side + c;
                    double flux = 0;
                    if (r > 0) {
                        double d = u[i - side] - u[i];
                        flux += Math.exp(-d * d * ik) * d;
                    }
                    if (r < side - 1) {
                        double d = u[i + side] - u[i];
                        flux += Math.exp(-d * d * ik) * d;
                    }
                    if (c > 0) {
                        double d = u[i - 1] - u[i];
                        flux += Math.exp(-d * d * ik) * d;
                    }
                    if (c < side - 1) {
                        double d = u[i + 1] - u[i];
                        flux += Math.exp(-d * d * ik) * d;
                    }
                    next[i] = u[i] + dt * flux;
                }
            }
            u = next;
        }
        return u;
    }
}
