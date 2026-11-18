package io.github.zzuegg.brainiac.experimental.imaging;

/**
 * Compressed-sensing MRI (Lustig, Donoho &amp; Pauly, MRM 2007): undersampled k-space with
 * incoherent (random, variable-density) sampling and a sparsity prior, solved by proximal
 * methods.
 *
 * <ul>
 *   <li>{@link Prior#TV}: {@code min 1/2 |M F x - y|^2 + lambda TV(x)} by Chambolle-Pock
 *       ({@link PrimalDual}) with the exact k-space proximal step of
 *       {@link MriOperator#dataProx}.</li>
 *   <li>{@link Prior#WAVELET}: {@code min 1/2 |M F x - y|^2 + lambda |W x|_1} with an orthogonal
 *       Daubechies-4 transform ({@link Wavelets}) by FISTA (Beck &amp; Teboulle 2009); the
 *       coarse band is not penalised, and the transform is shifted by a different circular offset
 *       every iteration (random cycle spinning, which approximates the translation-invariant
 *       wavelet prior and removes blocky artefacts).</li>
 * </ul>
 */
public final class CompressedSensingMri {

    /** The sparsity priors. */
    public enum Prior {
        /** Total variation. */
        TV,
        /** l1 on orthogonal wavelet details. */
        WAVELET
    }

    private CompressedSensingMri() {
    }

    /**
     * Reconstructs an image.
     *
     * @param prior the prior
     * @param op the operator
     * @param y the measurements
     * @param lambda the regularisation weight
     * @param iterations the iterations
     * @return the image
     */
    public static double[] reconstruct(Prior prior, MriOperator op, double[] y, double lambda, int iterations) {
        double[] zeroFilled = op.adjoint(y);
        if (prior == Prior.TV) {
            return PrimalDual.tv(op.dataProx(y), zeroFilled, op.side(), lambda, iterations);
        }
        return fistaWavelet(op, y, op.side(), lambda, iterations, zeroFilled);
    }

    private static double[] shift(double[] x, int side, int dy, int dx) {
        double[] y = new double[x.length];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                y[r * side + c] = x[Math.floorMod(r + dy, side) * side + Math.floorMod(c + dx, side)];
            }
        }
        return y;
    }

    /**
     * FISTA with orthogonal-wavelet soft thresholding (cycle-spun) for any operator with
     * {@code |A| <= 1}.
     *
     * @param op the operator
     * @param y the measurements
     * @param side the image side
     * @param lambda the l1 weight
     * @param iterations the iterations
     * @param start the start
     * @return the image
     */
    public static double[] fistaWavelet(LinearOperator op, double[] y, int side, double lambda, int iterations,
            double[] start) {
        int levels = 3;
        int coarse = side >> levels;
        double[] x = start.clone();
        double[] z = x.clone();
        double t = 1;
        for (int it = 0; it < iterations; it++) {
            double[] r = op.apply(z);
            for (int i = 0; i < r.length; i++) {
                r[i] -= y[i];
            }
            double[] g = op.adjoint(r);
            double[] v = shift(ImageMath.axpy(z, -1, g), side, it * 3 % 8, it * 5 % 8);
            double[] c = Wavelets.forward2d(v, side, Wavelets.Wavelet.DB4, levels);
            for (int row = 0; row < side; row++) {
                for (int col = 0; col < side; col++) {
                    if (row < coarse && col < coarse) {
                        continue;
                    }
                    int i = row * side + col;
                    c[i] = Math.signum(c[i]) * Math.max(0, Math.abs(c[i]) - lambda);
                }
            }
            double[] next = shift(Wavelets.inverse2d(c, side, Wavelets.Wavelet.DB4, levels), side, -(it * 3 % 8),
                    -(it * 5 % 8));
            double tn = (1 + Math.sqrt(1 + 4 * t * t)) / 2;
            for (int i = 0; i < z.length; i++) {
                z[i] = next[i] + (t - 1) / tn * (next[i] - x[i]);
            }
            x = next;
            t = tn;
        }
        return x;
    }
}
