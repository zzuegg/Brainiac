package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * One layer of a diffractive deep neural network (D2NN; Lin et al.,
 * <a href="https://arxiv.org/abs/1804.08711">1804.08711</a>): a passive, phase-only mask
 * followed by free-space propagation to the next plane. The optical field is complex,
 * stored as {@code [batch, n, n, 2]} (real, imaginary). The mask multiplies the field by
 * {@code exp(i phi)} with learnable phases {@code phi}; propagation over a distance
 * {@code z} uses the angular spectrum method with the core FFT:
 *
 * <pre>
 * E(z) = IFFT2( FFT2(E(0)) * exp(i 2 pi z sqrt(1/lambda^2 - fx^2 - fy^2)) )
 * </pre>
 *
 * with evanescent frequencies ({@code fx^2 + fy^2 > 1/lambda^2}) dropped. Lengths are in
 * units of the pixel pitch. The FFT makes the aperture periodic, so pad the input field when
 * light should not wrap around. A detector reads intensities {@code |E|^2} ({@link #intensity}).
 */
public final class DiffractiveLayer implements Layer {
    private final int size;
    private final Var phase;
    private final Tensor transfer;

    private DiffractiveLayer(int size, double wavelength, double distance, SplittableRandom random) {
        this.size = size;
        this.phase = Var.param(Tensor.uniform(Shape.of(size, size, 1), 0, (float) (2 * Math.PI), random));
        this.transfer = transfer(size, wavelength, distance);
    }

    /**
     * Creates a layer with random phases.
     *
     * @param size the side length n in pixels
     * @param wavelength the wavelength, in pixel pitches
     * @param distance the propagation distance after the mask, in pixel pitches
     * @param seed the seed
     * @return the layer
     */
    public static DiffractiveLayer of(int size, double wavelength, double distance, long seed) {
        return new DiffractiveLayer(size, wavelength, distance, new SplittableRandom(seed));
    }

    /**
     * Describes the layer for model builders; input {@code [n, n, 2]}.
     *
     * @param wavelength the wavelength in pixel pitches
     * @param distance the propagation distance in pixel pitches
     * @return the spec
     */
    public static LayerSpec spec(double wavelength, double distance) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new DiffractiveLayer(input.dim(0), wavelength, distance, random);
            }
        };
    }

    /** The angular spectrum transfer function as complex {@code [n, n, 2]}, in FFT frequency order. */
    private static Tensor transfer(int n, double wavelength, double distance) {
        float[] h = new float[n * n * 2];
        double k2 = 1 / (wavelength * wavelength);
        for (int u = 0; u < n; u++) {
            double fy = (u <= n / 2 ? u : u - n) / (double) n;
            for (int v = 0; v < n; v++) {
                double fx = (v <= n / 2 ? v : v - n) / (double) n;
                double q = k2 - fx * fx - fy * fy;
                if (q > 0) {
                    double phi = 2 * Math.PI * distance * Math.sqrt(q);
                    h[(u * n + v) * 2] = (float) Math.cos(phi);
                    h[(u * n + v) * 2 + 1] = (float) Math.sin(phi);
                }
            }
        }
        return Tensor.of(Shape.of(n, n, 2), h);
    }

    /**
     * Multiplies complex fields {@code [..., 2]}.
     *
     * @param a the first factor
     * @param b the second factor, broadcast against the first
     * @return {@code a b}
     */
    public static Var complexMultiply(Var a, Var b) {
        int last = a.shape().rank() - 1;
        int lastB = b.shape().rank() - 1;
        Var ar = a.slice(last, 0, 1);
        Var ai = a.slice(last, 1, 2);
        Var br = b.slice(lastB, 0, 1);
        Var bi = b.slice(lastB, 1, 2);
        return Var.concat(last, ar.mul(br).sub(ai.mul(bi)), ar.mul(bi).add(ai.mul(br)));
    }

    /**
     * The 2-D discrete Fourier transform of fields {@code [batch, n, n, 2]}, as two 1-D core
     * FFTs with a transpose between them.
     *
     * @param field the field
     * @param inverse whether to invert
     * @return the transform
     */
    public static Var fft2(Var field, boolean inverse) {
        Var rows = field.fft(inverse).permute(0, 2, 1, 3);
        return rows.fft(inverse).permute(0, 2, 1, 3);
    }

    /**
     * Free-space propagation by the angular spectrum method.
     *
     * @param field {@code [batch, n, n, 2]}
     * @param wavelength the wavelength in pixel pitches
     * @param distance the distance in pixel pitches
     * @return the field at the next plane
     */
    public static Var propagate(Var field, double wavelength, double distance) {
        Tensor h = transfer(field.shape().dim(1), wavelength, distance).to(field.value().device());
        return fft2(complexMultiply(fft2(field, false), Var.of(h)), true);
    }

    /**
     * The detected intensity.
     *
     * @param field {@code [batch, n, n, 2]}
     * @return {@code |E|^2}, {@code [batch, n, n]}
     */
    public static Var intensity(Var field) {
        Shape s = field.shape();
        return field.square().sum(3, false).reshape(s.dim(0), s.dim(1), s.dim(2));
    }

    @Override
    public Var forward(Var field) {
        Shape s = field.shape();
        if (s.rank() != 4 || s.dim(1) != size || s.dim(2) != size || s.dim(3) != 2) {
            throw new IllegalArgumentException("DiffractiveLayer expects [batch, " + size + ", " + size + ", 2], got " + s);
        }
        Var mask = Var.concat(2, phase.cos(), phase.sin());
        Var modulated = complexMultiply(field, mask);
        return fft2(complexMultiply(fft2(modulated, false), Var.of(transfer.to(field.value().device()))), true);
    }

    /** @return the learnable phases {@code [n, n, 1]} */
    public Var phase() {
        return phase;
    }

    @Override
    public List<Var> parameters() {
        return List.of(phase);
    }

    @Override
    public Shape outputShape(Shape inputShape) {
        return inputShape;
    }
}
