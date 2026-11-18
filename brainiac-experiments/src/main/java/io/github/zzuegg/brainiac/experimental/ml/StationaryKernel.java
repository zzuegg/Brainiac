package io.github.zzuegg.brainiac.experimental.ml;

/**
 * A stationary kernel {@code k(a, b) = variance * f(|a - b| / lengthscale)} from the RBF or
 * Matérn family, with the gradient of the kernel in its log lengthscale that
 * marginal-likelihood fitting of a Gaussian process needs (Rasmussen and Williams,
 * <a href="http://gaussianprocess.org/gpml/">GPML</a>, chapter 4). The gradient in the log
 * variance is the kernel itself.
 *
 * @param family the shape {@code f}
 * @param lengthscale the input scale, positive
 * @param variance the output scale (signal variance), positive
 */
public record StationaryKernel(Family family, double lengthscale, double variance) implements KernelFunction {

    /** The shape of the kernel as a function of the scaled distance {@code u}. */
    public enum Family {
        /** {@code exp(-u^2 / 2)}: infinitely smooth samples. */
        RBF,
        /** Matérn 3/2: {@code (1 + sqrt3 u) exp(-sqrt3 u)}, once-differentiable samples. */
        MATERN32,
        /** Matérn 5/2: {@code (1 + sqrt5 u + 5u^2/3) exp(-sqrt5 u)}, twice-differentiable samples. */
        MATERN52
    }

    private static final double SQRT3 = Math.sqrt(3);
    private static final double SQRT5 = Math.sqrt(5);

    /** Checks the hyperparameters. */
    public StationaryKernel {
        if (!(lengthscale > 0 && variance > 0)) {
            throw new IllegalArgumentException("lengthscale and variance must be positive");
        }
    }

    /**
     * An RBF kernel.
     *
     * @param lengthscale the input scale
     * @param variance the output scale
     * @return the kernel
     */
    public static StationaryKernel rbf(double lengthscale, double variance) {
        return new StationaryKernel(Family.RBF, lengthscale, variance);
    }

    /**
     * A Matérn 5/2 kernel.
     *
     * @param lengthscale the input scale
     * @param variance the output scale
     * @return the kernel
     */
    public static StationaryKernel matern52(double lengthscale, double variance) {
        return new StationaryKernel(Family.MATERN52, lengthscale, variance);
    }

    @Override
    public double value(double[] a, double[] b) {
        return variance * shape(Math.sqrt(MlMatrices.squaredDistance(a, b)) / lengthscale);
    }

    /**
     * The kernel's shape at a scaled distance.
     *
     * @param u {@code |a - b| / lengthscale}
     * @return {@code f(u)}
     */
    public double shape(double u) {
        return switch (family) {
            case RBF -> Math.exp(-0.5 * u * u);
            case MATERN32 -> (1 + SQRT3 * u) * Math.exp(-SQRT3 * u);
            case MATERN52 -> (1 + SQRT5 * u + 5 * u * u / 3) * Math.exp(-SQRT5 * u);
        };
    }

    /**
     * The derivative {@code d k / d log lengthscale} at a pair of points.
     *
     * @param a one point
     * @param b another
     * @return the derivative
     */
    public double dLogLengthscale(double[] a, double[] b) {
        double u = Math.sqrt(MlMatrices.squaredDistance(a, b)) / lengthscale;
        // d f(u) / d log l = f'(u) * (-u)
        double fPrime = switch (family) {
            case RBF -> -u * Math.exp(-0.5 * u * u);
            case MATERN32 -> -3 * u * Math.exp(-SQRT3 * u);
            case MATERN52 -> -5.0 / 3 * u * (1 + SQRT5 * u) * Math.exp(-SQRT5 * u);
        };
        return variance * fPrime * -u;
    }

    /**
     * The gradient of {@code k(a, b)} with respect to {@code a}, for backpropagating a kernel
     * loss into learned features (deep kernel learning).
     *
     * @param a the point to differentiate by
     * @param b the other point
     * @return {@code d k / d a}
     */
    public double[] gradient(double[] a, double[] b) {
        double u = Math.sqrt(MlMatrices.squaredDistance(a, b)) / lengthscale;
        // f'(u) / u, finite at u = 0 for every family
        double fOverU = switch (family) {
            case RBF -> -Math.exp(-0.5 * u * u);
            case MATERN32 -> -3 * Math.exp(-SQRT3 * u);
            case MATERN52 -> -5.0 / 3 * (1 + SQRT5 * u) * Math.exp(-SQRT5 * u);
        };
        double c = variance * fOverU / (lengthscale * lengthscale);
        double[] g = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            g[i] = c * (a[i] - b[i]);
        }
        return g;
    }

    /**
     * The same family with new hyperparameters.
     *
     * @param newLengthscale the input scale
     * @param newVariance the output scale
     * @return the kernel
     */
    public StationaryKernel with(double newLengthscale, double newVariance) {
        return new StationaryKernel(family, newLengthscale, newVariance);
    }
}
