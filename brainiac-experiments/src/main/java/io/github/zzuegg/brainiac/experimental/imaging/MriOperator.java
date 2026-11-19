package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Single-coil Cartesian MRI: a real image's unitary 2-D DFT ({@code F / sqrt(n)}) restricted to
 * the sampled k-space locations, {@code y = M F x}. As a {@link LinearOperator} the measurement is
 * the real and imaginary parts {@code [re (n), im (n)]} with zeros at unsampled locations; the
 * adjoint is the real part of the zero-filled inverse transform. The batch versions
 * ({@link #forward(Var)}, {@link #adjoint(Var)}, {@link #normal(Var)}) run on core
 * {@link Var#fft2} so unrolled networks can be trained through them.
 *
 * <p>With a conjugate-symmetric mask the normal operator {@code A^T A} maps real images to real
 * images and is diagonal in k-space, so {@link #dataProx} solves the proximal step of
 * {@code 1/2 |A x - y|^2} exactly.
 */
public final class MriOperator implements LinearOperator {

    private final int side;
    private final boolean[] mask;
    private final Tensor maskTensor;

    /**
     * Creates the operator.
     *
     * @param side the image side
     * @param mask the sampled k-space locations, {@code side x side}
     */
    public MriOperator(int side, boolean[] mask) {
        this.side = side;
        this.mask = mask.clone();
        float[] m = new float[side * side * 2];
        for (int i = 0; i < side * side; i++) {
            m[2 * i] = mask[i] ? 1 : 0;
            m[2 * i + 1] = mask[i] ? 1 : 0;
        }
        this.maskTensor = Tensor.of(Shape.of(1, side, side, 2), m);
    }

    /** @return the image side */
    public int side() {
        return side;
    }

    /** @return the sampled k-space locations (a copy) */
    public boolean[] mask() {
        return mask.clone();
    }

    /** @return the sampled share of k-space */
    public double samplingRate() {
        int c = 0;
        for (boolean b : mask) {
            c += b ? 1 : 0;
        }
        return c / (double) mask.length;
    }

    @Override
    public int inputSize() {
        return side * side;
    }

    @Override
    public int outputSize() {
        return 2 * side * side;
    }

    @Override
    public double[] apply(double[] x) {
        int n = side * side;
        double[] re = x.clone();
        double[] im = new double[n];
        ImageMath.fft2(re, im, side, side, false);
        double s = 1 / Math.sqrt(n);
        double[] y = new double[2 * n];
        for (int i = 0; i < n; i++) {
            if (mask[i]) {
                y[i] = re[i] * s;
                y[n + i] = im[i] * s;
            }
        }
        return y;
    }

    @Override
    public double[] adjoint(double[] y) {
        int n = side * side;
        double[] re = new double[n];
        double[] im = new double[n];
        for (int i = 0; i < n; i++) {
            if (mask[i]) {
                re[i] = y[i];
                im[i] = y[n + i];
            }
        }
        ImageMath.fft2(re, im, side, side, true);
        double s = Math.sqrt(n);
        for (int i = 0; i < n; i++) {
            re[i] *= s;
        }
        return re;
    }

    /**
     * The proximal step of {@code 1/2 |A x - y|^2}, exact in k-space:
     * {@code X = (V + tau M Y) / (1 + tau M)}.
     *
     * @param y the measurements
     * @return the data term for {@link PrimalDual}
     */
    public PrimalDual.DataTerm dataProx(double[] y) {
        int n = side * side;
        double s = Math.sqrt(n);
        return new PrimalDual.DataTerm() {
            @Override
            public double[] prox(double[] v, double tau) {
                double[] re = v.clone();
                double[] im = new double[n];
                ImageMath.fft2(re, im, side, side, false);
                for (int i = 0; i < n; i++) {
                    if (mask[i]) {
                        re[i] = (re[i] + tau * s * y[i]) / (1 + tau);
                        im[i] = (im[i] + tau * s * y[n + i]) / (1 + tau);
                    }
                }
                ImageMath.fft2(re, im, side, side, true);
                return re;
            }

            @Override
            public double value(double[] x) {
                double[] r = apply(x);
                double e = 0;
                for (int i = 0; i < r.length; i++) {
                    e += (r[i] - y[i]) * (r[i] - y[i]);
                }
                return 0.5 * e;
            }
        };
    }

    /**
     * Packs flat measurements into the batch layout.
     *
     * @param y measurements {@code [re (n), im (n)]}
     * @return {@code [1, side, side, 2]}
     */
    public Tensor pack(double[] y) {
        int n = side * side;
        float[] v = new float[2 * n];
        for (int i = 0; i < n; i++) {
            v[2 * i] = (float) y[i];
            v[2 * i + 1] = (float) y[n + i];
        }
        return Tensor.of(Shape.of(1, side, side, 2), v);
    }

    /**
     * The forward model on a batch.
     *
     * @param x images {@code [b, 1, side, side]}
     * @return masked k-space {@code [b, side, side, 2]}
     */
    public Var forward(Var x) {
        int b = x.shape().dim(0);
        Var re = x.reshape(b, side, side, 1);
        Var c = Var.concat(3, re, Var.of(Tensor.zeros(Shape.of(b, side, side, 1))));
        return c.fft2(false).mul((float) (1 / Math.sqrt(side * side))).mul(maskTensor);
    }

    /**
     * The adjoint on a batch.
     *
     * @param k k-space {@code [b, side, side, 2]}
     * @return images {@code [b, 1, side, side]}
     */
    public Var adjoint(Var k) {
        int b = k.shape().dim(0);
        Var img = k.mul(maskTensor).fft2(true).mul((float) Math.sqrt(side * side));
        return img.slice(3, 0, 1).reshape(b, 1, side, side);
    }

    /**
     * The normal operator {@code A^T A} on a batch.
     *
     * @param x images {@code [b, 1, side, side]}
     * @return {@code A^T A x}
     */
    public Var normal(Var x) {
        return adjoint(forward(x));
    }
}
