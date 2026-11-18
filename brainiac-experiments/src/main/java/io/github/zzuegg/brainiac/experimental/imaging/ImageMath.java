package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.experimental.data.DegradedImages;
import io.github.zzuegg.brainiac.tensor.spi.Fft;

/**
 * Small double-precision helpers on flat square images shared by the imaging blocks: 2-D DFTs
 * on core {@link Fft}, circular convolution and transfer functions, forward-difference
 * gradients and their adjoint divergence, box filters, and the PSNR/SSIM metrics (SSIM from
 * {@link DegradedImages}).
 */
public final class ImageMath {

    private ImageMath() {
    }

    /**
     * The 2-D DFT in place, unnormalised forward and {@code 1/(h w)} inverse.
     *
     * @param re real parts, {@code h x w} row-major
     * @param im imaginary parts
     * @param h the height
     * @param w the width
     * @param inverse whether to invert
     */
    public static void fft2(double[] re, double[] im, int h, int w, boolean inverse) {
        float[] rows = new float[2 * h * w];
        for (int i = 0; i < h * w; i++) {
            rows[2 * i] = (float) re[i];
            rows[2 * i + 1] = (float) im[i];
        }
        Fft.transform(rows, h, w, inverse);
        float[] cols = new float[2 * h * w];
        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                cols[2 * (c * h + r)] = rows[2 * (r * w + c)];
                cols[2 * (c * h + r) + 1] = rows[2 * (r * w + c) + 1];
            }
        }
        Fft.transform(cols, w, h, inverse);
        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                re[r * w + c] = cols[2 * (c * h + r)];
                im[r * w + c] = cols[2 * (c * h + r) + 1];
            }
        }
    }

    /**
     * The transfer function of a centred kernel on a {@code side x side} grid (circular).
     *
     * @param kernel the kernel, {@code k x k}
     * @param k its side
     * @param side the grid side
     * @return {@code {re, im}}
     */
    public static double[][] transfer(double[] kernel, int k, int side) {
        double[] re = new double[side * side];
        double[] im = new double[side * side];
        int c = k / 2;
        for (int ky = 0; ky < k; ky++) {
            for (int kx = 0; kx < k; kx++) {
                re[Math.floorMod(ky - c, side) * side + Math.floorMod(kx - c, side)] += kernel[ky * k + kx];
            }
        }
        fft2(re, im, side, side, false);
        return new double[][] {re, im};
    }

    /**
     * Circular convolution with a centred kernel.
     *
     * @param x the image, {@code side x side}
     * @param side its side
     * @param kernel the kernel, {@code k x k}
     * @param k its side
     * @return the blurred image
     */
    public static double[] convolve(double[] x, int side, double[] kernel, int k) {
        double[] y = new double[x.length];
        int c = k / 2;
        for (int r = 0; r < side; r++) {
            for (int col = 0; col < side; col++) {
                double s = 0;
                for (int ky = 0; ky < k; ky++) {
                    int rr = Math.floorMod(r - ky + c, side) * side;
                    for (int kx = 0; kx < k; kx++) {
                        double kv = kernel[ky * k + kx];
                        if (kv != 0) {
                            s += kv * x[rr + Math.floorMod(col - kx + c, side)];
                        }
                    }
                }
                y[r * side + col] = s;
            }
        }
        return y;
    }

    /**
     * Forward differences along x (columns), zero at the last column.
     *
     * @param x the image
     * @param side its side
     * @return the differences
     */
    public static double[] gradX(double[] x, int side) {
        double[] g = new double[x.length];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side - 1; c++) {
                g[r * side + c] = x[r * side + c + 1] - x[r * side + c];
            }
        }
        return g;
    }

    /**
     * Forward differences along y (rows), zero at the last row.
     *
     * @param x the image
     * @param side its side
     * @return the differences
     */
    public static double[] gradY(double[] x, int side) {
        double[] g = new double[x.length];
        for (int r = 0; r < side - 1; r++) {
            for (int c = 0; c < side; c++) {
                g[r * side + c] = x[(r + 1) * side + c] - x[r * side + c];
            }
        }
        return g;
    }

    /**
     * The divergence, minus the adjoint of ({@link #gradX}, {@link #gradY}).
     *
     * @param px the x component
     * @param py the y component
     * @param side the side
     * @return the divergence
     */
    public static double[] div(double[] px, double[] py, int side) {
        double[] d = new double[px.length];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                int i = r * side + c;
                double v = 0;
                if (c < side - 1) {
                    v += px[i];
                }
                if (c > 0) {
                    v -= px[i - 1];
                }
                if (r < side - 1) {
                    v += py[i];
                }
                if (r > 0) {
                    v -= py[i - side];
                }
                d[i] = v;
            }
        }
        return d;
    }

    /**
     * The isotropic total variation {@code sum |grad x|}.
     *
     * @param x the image
     * @param side its side
     * @return the total variation
     */
    public static double totalVariation(double[] x, int side) {
        double[] gx = gradX(x, side);
        double[] gy = gradY(x, side);
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            s += Math.hypot(gx[i], gy[i]);
        }
        return s;
    }

    /**
     * The mean over a {@code (2r+1)^2} window, clipped at the borders.
     *
     * @param x the image
     * @param side its side
     * @param radius the radius
     * @return the filtered image
     */
    public static double[] boxFilter(double[] x, int side, int radius) {
        double[] integral = new double[(side + 1) * (side + 1)];
        for (int r = 0; r < side; r++) {
            double row = 0;
            for (int c = 0; c < side; c++) {
                row += x[r * side + c];
                integral[(r + 1) * (side + 1) + c + 1] = integral[r * (side + 1) + c + 1] + row;
            }
        }
        double[] out = new double[x.length];
        for (int r = 0; r < side; r++) {
            int r0 = Math.max(0, r - radius);
            int r1 = Math.min(side, r + radius + 1);
            for (int c = 0; c < side; c++) {
                int c0 = Math.max(0, c - radius);
                int c1 = Math.min(side, c + radius + 1);
                double s = integral[r1 * (side + 1) + c1] - integral[r0 * (side + 1) + c1]
                        - integral[r1 * (side + 1) + c0] + integral[r0 * (side + 1) + c0];
                out[r * side + c] = s / ((r1 - r0) * (c1 - c0));
            }
        }
        return out;
    }

    /**
     * The Euclidean norm.
     *
     * @param x the vector
     * @return {@code |x|}
     */
    public static double norm(double[] x) {
        return Math.sqrt(dot(x, x));
    }

    /**
     * The inner product.
     *
     * @param a the first vector
     * @param b the second vector
     * @return {@code a . b}
     */
    public static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /**
     * The mean squared error.
     *
     * @param a the estimate
     * @param b the reference
     * @return the error
     */
    public static double mse(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return s / a.length;
    }

    /**
     * The peak signal-to-noise ratio for intensities in {@code [0, 1]}, estimate clipped.
     *
     * @param estimate the estimate
     * @param reference the reference
     * @return PSNR in dB
     */
    public static double psnr(double[] estimate, double[] reference) {
        double s = 0;
        for (int i = 0; i < estimate.length; i++) {
            double d = Math.max(0, Math.min(1, estimate[i])) - reference[i];
            s += d * d;
        }
        return 10 * Math.log10(1 / Math.max(1e-12, s / estimate.length));
    }

    /**
     * The structural similarity of square images (via {@link DegradedImages#ssim}).
     *
     * @param estimate the estimate
     * @param reference the reference
     * @param side the side
     * @return SSIM
     */
    public static double ssim(double[] estimate, double[] reference, int side) {
        return DegradedImages.ssim(toFloat(clip(estimate)), toFloat(reference), side);
    }

    /**
     * Clips to {@code [0, 1]}.
     *
     * @param x the values
     * @return a clipped copy
     */
    public static double[] clip(double[] x) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = Math.max(0, Math.min(1, x[i]));
        }
        return y;
    }

    /**
     * Converts to floats.
     *
     * @param x the values
     * @return a float copy
     */
    public static float[] toFloat(double[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = (float) x[i];
        }
        return y;
    }

    /**
     * Converts to doubles.
     *
     * @param x the values
     * @return a double copy
     */
    public static double[] toDouble(float[] x) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i];
        }
        return y;
    }

    /**
     * {@code a + s b}.
     *
     * @param a the first vector
     * @param s the scale
     * @param b the second vector
     * @return a new vector
     */
    public static double[] axpy(double[] a, double s, double[] b) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            y[i] = a[i] + s * b[i];
        }
        return y;
    }

    /**
     * Returns the best PSNR over integer shifts of the estimate up to {@code maxShift} (circular),
     * the usual way blind deconvolution results are scored because the kernel's centre is not
     * identifiable.
     *
     * @param estimate the estimate
     * @param reference the reference
     * @param side the side
     * @param maxShift the largest shift per axis
     * @return the best mean squared error
     */
    public static double shiftedMse(double[] estimate, double[] reference, int side, int maxShift) {
        double best = Double.MAX_VALUE;
        int border = maxShift;
        for (int dy = -maxShift; dy <= maxShift; dy++) {
            for (int dx = -maxShift; dx <= maxShift; dx++) {
                double s = 0;
                int n = 0;
                for (int r = border; r < side - border; r++) {
                    for (int c = border; c < side - border; c++) {
                        double e = Math.max(0, Math.min(1, estimate[Math.floorMod(r + dy, side) * side
                                + Math.floorMod(c + dx, side)])) - reference[r * side + c];
                        s += e * e;
                        n++;
                    }
                }
                best = Math.min(best, s / n);
            }
        }
        return best;
    }
}
