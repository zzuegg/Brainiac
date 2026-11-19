package io.github.zzuegg.brainiac.experimental.sensing;

import io.github.zzuegg.brainiac.autograd.Decompositions;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ArraySignals;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * SubspaceNet (Shmuel et al., 2023, <a href="https://arxiv.org/abs/2306.02271">arXiv:2306.02271</a>;
 * near-field DCD-MUSIC: Gast, Le Magoarou &amp; Shlezinger, 2025,
 * <a href="https://arxiv.org/abs/2504.00599">arXiv:2504.00599</a>): a small network maps the
 * snapshots' empirical autocorrelations {@code R_tau = 1/T sum x(t + tau) x(t)^H},
 * {@code tau = 0..L-1}, to a factor {@code B} and a Hermitian positive semi-definite surrogate
 * covariance {@code B B^H + eps I}; MUSIC or root-MUSIC ({@link ArrayProcessing}) then runs on the
 * surrogate. Training needs no true covariance: the loss is the MUSIC null spectrum
 * {@code |E_n^H a(theta_k)|^2} at the true directions (or near-field positions), with the noise
 * subspace {@code E_n} from core {@link Decompositions#eigh} on the real {@code 2M x 2M} embedding
 * of the surrogate, so gradients flow through the eigendecomposition.
 *
 * <p>{@link Geometry#NEAR_FIELD} trains with near-field steering vectors and estimates with
 * {@link #dcdMusic}: angles by far-field MUSIC, then each source's range by 1-D MUSIC over range;
 * {@link #music2d} is the 2-D MUSIC reference.
 */
public final class SubspaceNet {

    /** The array geometry the loss and the estimator assume. */
    public enum Geometry {
        /** Plane waves. */
        FAR_FIELD,
        /** Spherical waves (sources within the Fraunhofer distance). */
        NEAR_FIELD
    }

    /**
     * A training or test example.
     *
     * @param snapshots the snapshots
     * @param angles the true directions in degrees
     * @param ranges the true ranges in wavelengths (near field), or {@code null}
     */
    public record Example(ArraySignals.Snapshots snapshots, double[] angles, double[] ranges) {
    }

    private final int sensors;
    private final int sources;
    private final int lags;
    private final Geometry geometry;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;

    /**
     * Creates the network: autocorrelation features, one hidden tanh layer, the factor {@code B}.
     *
     * @param sensors {@code M}
     * @param sources {@code K}
     * @param lags {@code L}
     * @param hidden the hidden width
     * @param geometry the geometry
     * @param seed the seed
     */
    public SubspaceNet(int sensors, int sources, int lags, int hidden, Geometry geometry, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.sensors = sensors;
        this.sources = sources;
        this.lags = lags;
        this.geometry = geometry;
        int in = 2 * sensors * sensors * lags;
        int out = 2 * sensors * sensors;
        this.w1 = Var.param(Tensor.normal(Shape.of(in, hidden), 0, (float) Math.sqrt(1.0 / in), r));
        this.b1 = Var.param(Tensor.zeros(Shape.of(hidden)));
        this.w2 = Var.param(Tensor.normal(Shape.of(hidden, out), 0, (float) (0.3 / Math.sqrt(hidden)), r));
        float[] bias = new float[out];
        for (int i = 0; i < sensors; i++) {
            bias[i * sensors + i] = 1;
        }
        this.b2 = Var.param(Tensor.of(Shape.of(out), bias));
    }

    /**
     * The autocorrelation features, normalised by the trace of {@code R_0}.
     *
     * @param x the snapshots
     * @param lags {@code L}
     * @return {@code 2 M^2 L} values
     */
    public static float[] features(ArraySignals.Snapshots x, int lags) {
        int m = x.sensors();
        int t = x.count();
        float[] f = new float[2 * m * m * lags];
        double tr = 0;
        for (int i = 0; i < m; i++) {
            for (int s = 0; s < t; s++) {
                tr += x.re()[i][s] * x.re()[i][s] + x.im()[i][s] * x.im()[i][s];
            }
        }
        tr /= t * m;
        for (int l = 0; l < lags; l++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    double sr = 0;
                    double si = 0;
                    for (int s = 0; s + l < t; s++) {
                        double ar = x.re()[i][s + l];
                        double ai = x.im()[i][s + l];
                        double br = x.re()[j][s];
                        double bi = x.im()[j][s];
                        sr += ar * br + ai * bi;
                        si += ai * br - ar * bi;
                    }
                    int base = 2 * (l * m * m + i * m + j);
                    f[base] = (float) (sr / (t - l) / tr);
                    f[base + 1] = (float) (si / (t - l) / tr);
                }
            }
        }
        return f;
    }

    /** The embedded surrogate {@code E(B) E(B)^T + eps I}, {@code [b, 2M, 2M]}. */
    private Var embedded(Var features) {
        int b = features.shape().dim(0);
        int m = sensors;
        Var out = features.matmul(w1).add(b1).tanh().matmul(w2).add(b2).reshape(b, 2, m, m);
        Var br = out.slice(1, 0, 1).reshape(b, m, m);
        Var bi = out.slice(1, 1, 2).reshape(b, m, m);
        Var top = Var.concat(2, br, bi.neg());
        Var bottom = Var.concat(2, bi, br);
        Var e = Var.concat(1, top, bottom);
        float[] eye = new float[b * 4 * m * m];
        for (int k = 0; k < b; k++) {
            for (int i = 0; i < 2 * m; i++) {
                eye[k * 4 * m * m + i * 2 * m + i] = 1e-3f;
            }
        }
        return e.matmul(e.permute(0, 2, 1)).add(Var.of(Tensor.of(Shape.of(b, 2 * m, 2 * m), eye)));
    }

    private double[][] steering(double angle, double range) {
        double[] re = new double[sensors];
        double[] im = new double[sensors];
        for (int i = 0; i < sensors; i++) {
            double ph = geometry == Geometry.NEAR_FIELD
                    ? ArraySignals.nearFieldPhase(sensors, i, angle, range)
                    : Math.PI * i * Math.sin(Math.toRadians(angle));
            re[i] = Math.cos(ph);
            im[i] = Math.sin(ph);
        }
        return new double[][] {re, im};
    }

    /**
     * Trains through the eigendecomposition with the MUSIC null-spectrum loss.
     *
     * @param data draws a training example
     * @param steps the Adam steps
     * @param batch the batch size
     * @param lr the learning rate
     * @param seed the seed
     * @return the final loss
     */
    public double fit(Function<SplittableRandom, Example> data, int steps, int batch, float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        int in = 2 * sensors * sensors * lags;
        int m2 = 2 * sensors;
        int noise = 2 * (sensors - sources);
        double last = Double.NaN;
        for (int s = 0; s < steps; s++) {
            float[] x = new float[batch * in];
            float[] a = new float[batch * m2 * sources];
            for (int b = 0; b < batch; b++) {
                Example ex = data.apply(r);
                System.arraycopy(features(ex.snapshots(), lags), 0, x, b * in, in);
                for (int k = 0; k < sources; k++) {
                    double[][] st = steering(ex.angles()[k], ex.ranges() == null ? 0 : ex.ranges()[k]);
                    for (int i = 0; i < sensors; i++) {
                        a[b * m2 * sources + i * sources + k] = (float) (st[0][i] / Math.sqrt(sensors));
                        a[b * m2 * sources + (i + sensors) * sources + k] = (float) (st[1][i] / Math.sqrt(sensors));
                    }
                }
            }
            Var cov = embedded(Var.of(Tensor.of(Shape.of(batch, in), x)));
            Var[] eig = Decompositions.eigh(cov);
            Var en = eig[1].slice(2, 0, noise);
            Var proj = en.permute(0, 2, 1).matmul(Var.of(Tensor.of(Shape.of(batch, m2, sources), a)));
            Var loss = proj.square().sum().mul(1f / (batch * sources));
            loss.backward();
            opt.step(List.of(w1, b1, w2, b2));
            last = loss.value().toArray()[0];
        }
        return last;
    }

    /**
     * The learned surrogate covariance.
     *
     * @param x the snapshots
     * @return {@code B B^H + eps I}
     */
    public ArrayProcessing.Covariance surrogate(ArraySignals.Snapshots x) {
        float[] f = features(x, lags);
        float[] e = Var.noGrad(() -> embedded(Var.of(Tensor.of(Shape.of(1, f.length), f)))).value().toArray();
        int m = sensors;
        double[][] re = new double[m][m];
        double[][] im = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                re[i][j] = e[i * 2 * m + j];
                im[i][j] = e[(i + m) * 2 * m + j];
            }
        }
        return new ArrayProcessing.Covariance(re, im);
    }

    /** @return the trainable values */
    public long parameterCount() {
        return w1.value().size() + b1.value().size() + w2.value().size() + b2.value().size();
    }

    // ---------------------------------------------------------------- near field

    private static double[][][] nearSteering(int m, double angle, double[] ranges) {
        double[][] re = new double[ranges.length][m];
        double[][] im = new double[ranges.length][m];
        for (int g = 0; g < ranges.length; g++) {
            for (int i = 0; i < m; i++) {
                double ph = ArraySignals.nearFieldPhase(m, i, angle, ranges[g]);
                re[g][i] = Math.cos(ph);
                im[g][i] = Math.sin(ph);
            }
        }
        return new double[][][] {re, im};
    }

    private static double[] rangeGrid(double lo, double hi, int n) {
        double[] g = new double[n];
        for (int i = 0; i < n; i++) {
            g[i] = lo + (hi - lo) * i / (n - 1);
        }
        return g;
    }

    /**
     * DCD-MUSIC: far-field MUSIC for the angles, then 1-D MUSIC over range for each angle.
     *
     * @param r the covariance (sample or learned)
     * @param sources {@code K}
     * @param rangeLo the smallest range (wavelengths)
     * @param rangeHi the largest range
     * @return {angles ascending, ranges}
     */
    public static double[][] dcdMusic(ArrayProcessing.Covariance r, int sources, double rangeLo, double rangeHi) {
        double[] angles = ArrayProcessing.estimate(ArrayProcessing.Method.MUSIC, r, sources);
        double[] grid = rangeGrid(rangeLo, rangeHi, 120);
        double[] ranges = new double[angles.length];
        for (int k = 0; k < angles.length; k++) {
            double[][][] st = nearSteering(r.size(), angles[k], grid);
            double[] p = ArrayProcessing.music(r, sources, st[0], st[1]);
            int best = 0;
            for (int g = 1; g < p.length; g++) {
                if (p[g] > p[best]) {
                    best = g;
                }
            }
            ranges[k] = grid[best];
        }
        return new double[][] {angles, ranges};
    }

    /**
     * 2-D MUSIC over angle and range: the {@code K} highest local maxima of the joint spectrum.
     *
     * @param r the covariance
     * @param sources {@code K}
     * @param rangeLo the smallest range
     * @param rangeHi the largest range
     * @return {angles ascending, ranges}
     */
    public static double[][] music2d(ArrayProcessing.Covariance r, int sources, double rangeLo, double rangeHi) {
        double[] angles = new double[121];
        for (int i = 0; i < angles.length; i++) {
            angles[i] = -60 + i;
        }
        double[] ranges = rangeGrid(rangeLo, rangeHi, 40);
        double[][] p = new double[angles.length][];
        for (int i = 0; i < angles.length; i++) {
            double[][][] st = nearSteering(r.size(), angles[i], ranges);
            p[i] = ArrayProcessing.music(r, sources, st[0], st[1]);
        }
        List<double[]> peaks = new ArrayList<>();
        for (int i = 0; i < angles.length; i++) {
            for (int j = 0; j < ranges.length; j++) {
                boolean max = true;
                for (int di = -1; di <= 1 && max; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        int a = i + di;
                        int b = j + dj;
                        if ((di != 0 || dj != 0) && a >= 0 && a < angles.length && b >= 0 && b < ranges.length
                                && p[a][b] > p[i][j]) {
                            max = false;
                            break;
                        }
                    }
                }
                if (max) {
                    peaks.add(new double[] {p[i][j], angles[i], ranges[j]});
                }
            }
        }
        peaks.sort((x, y) -> Double.compare(y[0], x[0]));
        int k = Math.min(sources, peaks.size());
        double[][] sel = new double[k][];
        for (int i = 0; i < k; i++) {
            sel[i] = peaks.get(i);
        }
        Arrays.sort(sel, (x, y) -> Double.compare(x[1], y[1]));
        double[] a = new double[k];
        double[] rr = new double[k];
        for (int i = 0; i < k; i++) {
            a[i] = sel[i][1];
            rr[i] = sel[i][2];
        }
        return new double[][] {a, rr};
    }
}
