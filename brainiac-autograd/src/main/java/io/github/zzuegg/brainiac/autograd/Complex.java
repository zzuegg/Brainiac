package io.github.zzuegg.brainiac.autograd;

/**
 * Complex arithmetic on the layout {@link Var#fft} uses: real and imaginary parts in a last
 * axis of size 2. Everything is made of ordinary {@link Var} operations, so gradients flow
 * (as for real inputs, the gradient of the real loss with respect to each part).
 */
public final class Complex {

    private Complex() {
    }

    /**
     * @param z complex values {@code [..., 2]}
     * @return the real parts {@code [...]}
     */
    public static Var real(Var z) {
        return part(z, 0);
    }

    /**
     * @param z complex values {@code [..., 2]}
     * @return the imaginary parts {@code [...]}
     */
    public static Var imag(Var z) {
        return part(z, 1);
    }

    private static Var part(Var z, int i) {
        int rank = z.shape().rank();
        if (z.shape().dim(rank - 1) != 2) {
            throw new IllegalArgumentException("complex values need a last axis of 2, got " + z.shape());
        }
        int[] dims = java.util.Arrays.copyOf(z.shape().toArray(), rank - 1);
        return z.slice(rank - 1, i, i + 1).reshape(dims);
    }

    /**
     * Builds complex values from their parts.
     *
     * @param re real parts {@code [...]}
     * @param im imaginary parts, same shape
     * @return {@code [..., 2]}
     */
    public static Var of(Var re, Var im) {
        int rank = re.shape().rank();
        int[] dims = java.util.Arrays.copyOf(re.shape().toArray(), rank + 1);
        dims[rank] = 1;
        return Var.concat(rank, re.reshape(dims), im.reshape(dims));
    }

    /**
     * Element-wise product, broadcasting like {@link Var#mul(Var)}.
     *
     * @param a complex values {@code [..., 2]}
     * @param b complex values {@code [..., 2]}
     * @return {@code a b}
     */
    public static Var mul(Var a, Var b) {
        Var ar = real(a);
        Var ai = imag(a);
        Var br = real(b);
        Var bi = imag(b);
        return of(ar.mul(br).sub(ai.mul(bi)), ar.mul(bi).add(ai.mul(br)));
    }

    /**
     * @param z complex values {@code [..., 2]}
     * @return the conjugates
     */
    public static Var conj(Var z) {
        return of(real(z), imag(z).neg());
    }

    /**
     * The magnitudes {@code sqrt(re^2 + im^2)}; the gradient at 0 is 0.
     *
     * @param z complex values {@code [..., 2]}
     * @return {@code [...]}
     */
    public static Var abs(Var z) {
        return z.square().sum(-1, false).add(1e-30f).sqrt();
    }

    /**
     * Complex values from magnitudes and angles.
     *
     * @param magnitude {@code [...]}
     * @param angle in radians, same shape
     * @return {@code magnitude e^(i angle)}, {@code [..., 2]}
     */
    public static Var polar(Var magnitude, Var angle) {
        return of(magnitude.mul(angle.cos()), magnitude.mul(angle.sin()));
    }
}
