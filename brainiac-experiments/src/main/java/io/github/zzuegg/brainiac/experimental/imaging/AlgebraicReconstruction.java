package io.github.zzuegg.brainiac.experimental.imaging;

/**
 * Algebraic (iterative) tomographic reconstruction on a {@link RadonTransform}.
 *
 * <ul>
 *   <li>{@link Method#ART}: Kaczmarz's method, projecting the estimate onto one measurement
 *       hyperplane {@code a_i . x = b_i} at a time, with relaxation.</li>
 *   <li>{@link Method#SART} (Andersen &amp; Kak, 1984): all rays at once,
 *       {@code x += lambda C A^T R (b - A x)} with {@code R} the inverse ray lengths and {@code C}
 *       the inverse column sums.</li>
 *   <li>{@link Method#SART_TV}: each SART sweep followed by a few Chambolle-Pock TV denoising
 *       steps ({@link PrimalDual}), the usual sparse-view recipe.</li>
 * </ul>
 * All keep the estimate non-negative.
 */
public final class AlgebraicReconstruction {

    /** The methods. */
    public enum Method {
        /** Kaczmarz row-action iterations. */
        ART,
        /** Simultaneous algebraic reconstruction. */
        SART,
        /** SART alternated with TV denoising. */
        SART_TV
    }

    private AlgebraicReconstruction() {
    }

    /**
     * Reconstructs an image.
     *
     * @param method the method
     * @param radon the operator
     * @param sinogram the measurements
     * @param sweeps passes over the data
     * @param relaxation the relaxation (about 1 for SART, 0.5 for ART)
     * @param tvWeight the TV weight of SART_TV's denoising step (ignored otherwise)
     * @return the image
     */
    public static double[] reconstruct(Method method, RadonTransform radon, double[] sinogram, int sweeps,
            double relaxation, double tvWeight) {
        int n = radon.inputSize();
        double[] x = new double[n];
        if (method == Method.ART) {
            int rows = radon.outputSize();
            double[][][] cache = new double[rows][][];
            double[] norms = new double[rows];
            for (int r = 0; r < rows; r++) {
                cache[r] = radon.row(r);
                for (double v : cache[r][1]) {
                    norms[r] += v * v;
                }
            }
            // a fixed permutation that interleaves angles (better conditioned than sequential order)
            int[] order = new int[rows];
            int angles = radon.angles();
            int side = radon.side();
            int k = 0;
            int stride = Math.max(1, (int) Math.round(angles * 0.381966));
            for (int i = 0; i < angles; i++) {
                int a = (int) ((long) i * stride % angles);
                for (int j = 0; j < side; j++) {
                    order[k++] = a * side + j;
                }
            }
            for (int s = 0; s < sweeps; s++) {
                for (int r : order) {
                    if (norms[r] == 0) {
                        continue;
                    }
                    double[] c = cache[r][0];
                    double[] v = cache[r][1];
                    double dot = 0;
                    for (int q = 0; q < c.length; q++) {
                        dot += v[q] * x[(int) c[q]];
                    }
                    double step = relaxation * (sinogram[r] - dot) / norms[r];
                    for (int q = 0; q < c.length; q++) {
                        int p = (int) c[q];
                        x[p] = Math.max(0, x[p] + step * v[q]);
                    }
                }
            }
            return x;
        }
        double[] rowSum = radon.rowSums();
        double[] ones = new double[radon.outputSize()];
        java.util.Arrays.fill(ones, 1);
        double[] colSum = radon.adjoint(ones);
        for (int s = 0; s < sweeps; s++) {
            double[] res = radon.apply(x);
            for (int i = 0; i < res.length; i++) {
                res[i] = rowSum[i] > 0 ? (sinogram[i] - res[i]) / rowSum[i] : 0;
            }
            double[] back = radon.adjoint(res);
            for (int i = 0; i < n; i++) {
                if (colSum[i] > 0) {
                    x[i] = Math.max(0, x[i] + relaxation * back[i] / colSum[i]);
                }
            }
            if (method == Method.SART_TV) {
                x = PrimalDual.denoise(PrimalDual.Model.TV_L2, x, radon.side(), tvWeight * Math.pow(0.93, s), 10);
            }
        }
        return x;
    }
}
