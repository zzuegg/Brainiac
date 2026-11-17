package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * A continuous path through observations, for controlled differential equations: the
 * observations {@code [batch, points, channels]} are knots at parameter values
 * {@code s = 0, 1, ..., points - 1}, joined either linearly or by a natural cubic spline
 * (twice continuously differentiable, zero curvature at the ends). Irregular sampling is
 * handled by making the observation time one of the channels, so each sample can have its
 * own time stamps while all share the parameter {@code s}.
 *
 * <p>The spline coefficients are fitted once per path (a tridiagonal solve per sample and
 * channel); {@link #value} and {@link #derivative} then evaluate it anywhere.
 */
public final class ControlPath {

    /** How knots are joined. */
    public enum Interpolation {
        /** Straight lines: a piecewise constant derivative. */
        LINEAR,
        /** A natural cubic spline. */
        CUBIC
    }

    private final int batch;
    private final int points;
    private final int channels;
    private final float[] y;
    private final double[] second;
    private final Interpolation interpolation;

    /**
     * Fits a path.
     *
     * @param observations {@code [batch, points, channels]}, at least 2 points
     * @param interpolation linear or cubic
     */
    public ControlPath(Tensor observations, Interpolation interpolation) {
        Shape s = observations.shape();
        if (s.rank() != 3 || s.dim(1) < 2) {
            throw new IllegalArgumentException("need [batch, points >= 2, channels], got " + s);
        }
        this.batch = s.dim(0);
        this.points = s.dim(1);
        this.channels = s.dim(2);
        this.y = observations.toArray();
        this.interpolation = interpolation;
        this.second = new double[y.length];
        if (interpolation == Interpolation.CUBIC && points > 2) {
            fitSpline();
        }
    }

    /** Natural spline with unit spacing: M_{i-1} + 4 M_i + M_{i+1} = 6 (y_{i+1} - 2 y_i + y_{i-1}). */
    private void fitSpline() {
        int m = points - 2;
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                double[] diag = new double[m];
                double[] rhs = new double[m];
                for (int i = 0; i < m; i++) {
                    diag[i] = 4;
                    rhs[i] = 6 * (at(b, i + 2, c) - 2 * at(b, i + 1, c) + at(b, i, c));
                }
                // Thomas algorithm with unit off-diagonals
                for (int i = 1; i < m; i++) {
                    double w = 1 / diag[i - 1];
                    diag[i] -= w;
                    rhs[i] -= w * rhs[i - 1];
                }
                double[] sol = new double[m];
                for (int i = m - 1; i >= 0; i--) {
                    sol[i] = (rhs[i] - (i + 1 < m ? sol[i + 1] : 0)) / diag[i];
                }
                for (int i = 0; i < m; i++) {
                    second[(b * points + i + 1) * channels + c] = sol[i];
                }
            }
        }
    }

    private double at(int b, int i, int c) {
        return y[(b * points + i) * channels + c];
    }

    private double m(int b, int i, int c) {
        return second[(b * points + i) * channels + c];
    }

    /** @return the number of knots */
    public int points() {
        return points;
    }

    /** @return the parameter range end, {@code points - 1} */
    public float end() {
        return points - 1;
    }

    private int segment(double s) {
        return (int) Math.min(points - 2, Math.max(0, Math.floor(s)));
    }

    /**
     * Evaluates the path.
     *
     * @param s the parameter, in {@code [0, points - 1]}
     * @return {@code [batch, channels]}
     */
    public Tensor value(double s) {
        int i = segment(s);
        double u = s - i;
        float[] out = new float[batch * channels];
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                double y0 = at(b, i, c);
                double y1 = at(b, i + 1, c);
                double v = y0 + (y1 - y0) * u;
                if (interpolation == Interpolation.CUBIC) {
                    double m0 = m(b, i, c);
                    double m1 = m(b, i + 1, c);
                    v = m0 * Math.pow(1 - u, 3) / 6 + m1 * Math.pow(u, 3) / 6
                            + (y0 - m0 / 6) * (1 - u) + (y1 - m1 / 6) * u;
                }
                out[b * channels + c] = (float) v;
            }
        }
        return Tensor.of(Shape.of(batch, channels), out);
    }

    /**
     * Evaluates the path's derivative {@code dX/ds}.
     *
     * @param s the parameter, in {@code [0, points - 1]}
     * @return {@code [batch, channels]}
     */
    public Tensor derivative(double s) {
        int i = segment(s);
        double u = s - i;
        float[] out = new float[batch * channels];
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                double y0 = at(b, i, c);
                double y1 = at(b, i + 1, c);
                double d = y1 - y0;
                if (interpolation == Interpolation.CUBIC) {
                    double m0 = m(b, i, c);
                    double m1 = m(b, i + 1, c);
                    d = -m0 * (1 - u) * (1 - u) / 2 + m1 * u * u / 2 + d - (m1 - m0) / 6;
                }
                out[b * channels + c] = (float) d;
            }
        }
        return Tensor.of(Shape.of(batch, channels), out);
    }
}
