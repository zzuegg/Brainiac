package io.github.zzuegg.brainiac.experimental.sensing;

import io.github.zzuegg.brainiac.experimental.data.ArraySignals;
import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.tensor.spi.Fft;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Direction finding on a half-wavelength uniform linear array, steering vector
 * {@code a_m(theta) = exp(i pi m sin theta)}:
 * <ul>
 *   <li>{@link Method#DELAY_AND_SUM}: the beam power {@code a^H R a / M^2};</li>
 *   <li>{@link Method#MVDR} (Capon, Proc. IEEE 1969): {@code 1 / (a^H R^-1 a)};</li>
 *   <li>{@link Method#MUSIC} (Schmidt, IEEE TAP 1986): {@code 1 / (a^H E_n E_n^H a)};</li>
 *   <li>{@link Method#ROOT_MUSIC} (Barabell, ICASSP 1983): the roots of
 *       {@code a(1/z)^T E_n E_n^H a(z)} nearest the unit circle (Durand-Kerner);</li>
 *   <li>{@link Method#ESPRIT} (Roy &amp; Kailath, IEEE TASSP 1989): the eigenvalues of the
 *       least-squares shift {@code Psi} between the two subarrays' signal subspaces
 *       (characteristic polynomial by Faddeev-LeVerrier, then Durand-Kerner);</li>
 * </ul>
 * plus forward-backward averaging and spatial smoothing (Shan, Wax &amp; Kailath 1985) for coherent
 * sources, MDL/AIC source counting (Wax &amp; Kailath 1985), the stochastic Cramer-Rao bound
 * (Stoica &amp; Nehorai 1990), and the FFT spatial periodogram (core {@link Fft}).
 *
 * <p>Hermitian eigenproblems use core {@link Matrices#symmetricEigen} on the real
 * {@code 2M x 2M} embedding {@code [[Re, -Im], [Im, Re]]}: its top {@code 2K} eigenvectors span
 * the embedding of the complex {@code K}-dimensional signal subspace, so projectors and quadratic
 * forms carry over exactly.
 */
public final class ArrayProcessing {

    /** The estimators. */
    public enum Method {
        /** Conventional beamforming. */
        DELAY_AND_SUM,
        /** Capon's minimum-variance distortionless response. */
        MVDR,
        /** Multiple signal classification. */
        MUSIC,
        /** Polynomial-rooting MUSIC. */
        ROOT_MUSIC,
        /** Rotational invariance. */
        ESPRIT
    }

    /**
     * A Hermitian matrix as real and imaginary parts.
     *
     * @param re the real part
     * @param im the imaginary part
     */
    public record Covariance(double[][] re, double[][] im) {

        /** @return the size */
        public int size() {
            return re.length;
        }
    }

    private ArrayProcessing() {
    }

    /**
     * The sample covariance {@code X X^H / T}.
     *
     * @param x the snapshots
     * @return the covariance
     */
    public static Covariance covariance(ArraySignals.Snapshots x) {
        int m = x.sensors();
        int t = x.count();
        double[][] re = new double[m][m];
        double[][] im = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                double sr = 0;
                double si = 0;
                for (int k = 0; k < t; k++) {
                    sr += x.re()[i][k] * x.re()[j][k] + x.im()[i][k] * x.im()[j][k];
                    si += x.im()[i][k] * x.re()[j][k] - x.re()[i][k] * x.im()[j][k];
                }
                re[i][j] = sr / t;
                im[i][j] = si / t;
            }
        }
        return new Covariance(re, im);
    }

    /**
     * Forward-backward averaging {@code (R + J conj(R) J) / 2}.
     *
     * @param r the covariance
     * @return the averaged covariance
     */
    public static Covariance forwardBackward(Covariance r) {
        int m = r.size();
        double[][] re = new double[m][m];
        double[][] im = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                re[i][j] = 0.5 * (r.re()[i][j] + r.re()[m - 1 - i][m - 1 - j]);
                im[i][j] = 0.5 * (r.im()[i][j] - r.im()[m - 1 - i][m - 1 - j]);
            }
        }
        return new Covariance(re, im);
    }

    /**
     * Spatial smoothing: the average of the {@code M - L + 1} overlapping {@code L x L}
     * subarray covariances, optionally forward-backward averaged.
     *
     * @param r the covariance
     * @param subarray {@code L}
     * @param forwardBackward whether to also average forward-backward
     * @return the smoothed {@code L x L} covariance
     */
    public static Covariance spatialSmoothing(Covariance r, int subarray, boolean forwardBackward) {
        int m = r.size();
        int n = m - subarray + 1;
        double[][] re = new double[subarray][subarray];
        double[][] im = new double[subarray][subarray];
        for (int s = 0; s < n; s++) {
            for (int i = 0; i < subarray; i++) {
                for (int j = 0; j < subarray; j++) {
                    re[i][j] += r.re()[s + i][s + j] / n;
                    im[i][j] += r.im()[s + i][s + j] / n;
                }
            }
        }
        Covariance c = new Covariance(re, im);
        return forwardBackward ? forwardBackward(c) : c;
    }

    // ---------------------------------------------------------------- complex helpers

    private static double[][] embed(double[][] re, double[][] im) {
        int m = re.length;
        double[][] e = new double[2 * m][2 * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                e[i][j] = re[i][j];
                e[i + m][j + m] = re[i][j];
                e[i][j + m] = -im[i][j];
                e[i + m][j] = im[i][j];
            }
        }
        return e;
    }

    /** Eigenvectors of the embedding, columns sorted by descending eigenvalue. */
    private static double[][] embeddedVectors(Covariance r, double[] valuesOut) {
        double[][] e = embed(r.re(), r.im());
        int n = e.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                double s = 0.5 * (e[i][j] + e[j][i]);
                e[i][j] = s;
                e[j][i] = s;
            }
        }
        Matrices.SymmetricEigen eig = Matrices.symmetricEigen(e);
        double[][] vec = new double[n][];
        for (int c = 0; c < n; c++) {
            int src = n - 1 - c;
            vec[c] = new double[n];
            for (int i = 0; i < n; i++) {
                vec[c][i] = eig.vectors()[i][src];
            }
            if (valuesOut != null) {
                valuesOut[c] = eig.values()[src];
            }
        }
        return vec;
    }

    /** The complex signal-subspace projector {@code {re, im}} from the top {@code 2K} embedded vectors. */
    private static double[][][] signalProjector(Covariance r, int sources) {
        int m = r.size();
        double[][] vec = embeddedVectors(r, null);
        double[][] pr = new double[m][m];
        double[][] pi = new double[m][m];
        for (int c = 0; c < 2 * sources; c++) {
            double[] v = vec[c];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    pr[i][j] += v[i] * v[j];
                    pi[i][j] += v[i + m] * v[j];
                }
            }
        }
        return new double[][][] {pr, pi};
    }

    private static double quadratic(double[][] re, double[][] im, double[] ar, double[] ai) {
        int m = ar.length;
        double s = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                s += ar[i] * (re[i][j] * ar[j] - im[i][j] * ai[j]) + ai[i] * (re[i][j] * ai[j] + im[i][j] * ar[j]);
            }
        }
        return s;
    }

    private static double[][] steering(int m, double angleDeg) {
        double[] ar = new double[m];
        double[] ai = new double[m];
        double u = Math.PI * Math.sin(Math.toRadians(angleDeg));
        for (int i = 0; i < m; i++) {
            ar[i] = Math.cos(u * i);
            ai[i] = Math.sin(u * i);
        }
        return new double[][] {ar, ai};
    }

    // ---------------------------------------------------------------- spectra

    /**
     * A pseudo-spectrum on a grid of angles.
     *
     * @param method {@link Method#DELAY_AND_SUM}, {@link Method#MVDR} or {@link Method#MUSIC}
     * @param r the covariance
     * @param sources {@code K} (MUSIC)
     * @param gridDeg the angles
     * @return the spectrum
     */
    public static double[] spectrum(Method method, Covariance r, int sources, double[] gridDeg) {
        int m = r.size();
        double[][] qre;
        double[][] qim;
        switch (method) {
            case DELAY_AND_SUM -> {
                qre = r.re();
                qim = r.im();
            }
            case MVDR -> {
                double load = 0;
                for (int i = 0; i < m; i++) {
                    load += r.re()[i][i] * 1e-3 / m;
                }
                double[][] e = embed(r.re(), r.im());
                for (int i = 0; i < 2 * m; i++) {
                    e[i][i] += load;
                }
                double[][] inv = Matrices.inverse(e);
                qre = new double[m][m];
                qim = new double[m][m];
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < m; j++) {
                        qre[i][j] = inv[i][j];
                        qim[i][j] = inv[i + m][j];
                    }
                }
            }
            case MUSIC -> {
                double[][][] p = signalProjector(r, sources);
                qre = p[0];
                qim = p[1];
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < m; j++) {
                        qre[i][j] = (i == j ? 1 : 0) - qre[i][j];
                        qim[i][j] = -qim[i][j];
                    }
                }
            }
            default -> throw new IllegalArgumentException(method + " has no spectrum");
        }
        double[] out = new double[gridDeg.length];
        for (int g = 0; g < gridDeg.length; g++) {
            double[][] a = steering(m, gridDeg[g]);
            double q = quadratic(qre, qim, a[0], a[1]);
            out[g] = method == Method.DELAY_AND_SUM ? q / (m * m) : 1 / Math.max(q, 1e-12);
        }
        return out;
    }

    /**
     * The {@code K} highest local maxima of a spectrum, refined by parabolic interpolation.
     *
     * @param spectrum the values
     * @param gridDeg the angles (uniform)
     * @param sources {@code K}
     * @return the angles, ascending (fewer when fewer peaks exist)
     */
    public static double[] peaks(double[] spectrum, double[] gridDeg, int sources) {
        List<int[]> found = new ArrayList<>();
        for (int i = 1; i < spectrum.length - 1; i++) {
            if (spectrum[i] > spectrum[i - 1] && spectrum[i] >= spectrum[i + 1]) {
                found.add(new int[] {i});
            }
        }
        found.sort((a, b) -> Double.compare(spectrum[b[0]], spectrum[a[0]]));
        int k = Math.min(sources, found.size());
        double[] out = new double[k];
        double step = gridDeg[1] - gridDeg[0];
        for (int j = 0; j < k; j++) {
            int i = found.get(j)[0];
            double l = Math.log(spectrum[i - 1]);
            double c = Math.log(spectrum[i]);
            double rr = Math.log(spectrum[i + 1]);
            double den = l - 2 * c + rr;
            double off = den == 0 ? 0 : 0.5 * (l - rr) / den;
            out[j] = gridDeg[i] + Math.max(-0.5, Math.min(0.5, off)) * step;
        }
        Arrays.sort(out);
        return out;
    }

    /**
     * A uniform grid of angles.
     *
     * @param stepDeg the spacing
     * @return the angles from -90 to 90
     */
    public static double[] grid(double stepDeg) {
        int n = (int) Math.round(180 / stepDeg) + 1;
        double[] g = new double[n];
        for (int i = 0; i < n; i++) {
            g[i] = -90 + i * stepDeg;
        }
        return g;
    }

    /**
     * Estimates the directions.
     *
     * @param method the estimator
     * @param r the covariance (smoothed or not)
     * @param sources {@code K}
     * @return the angles in degrees, ascending
     */
    public static double[] estimate(Method method, Covariance r, int sources) {
        return switch (method) {
            case DELAY_AND_SUM, MVDR, MUSIC -> peaks(spectrum(method, r, sources, grid(0.25)), grid(0.25), sources);
            case ROOT_MUSIC -> rootMusic(r, sources);
            case ESPRIT -> esprit(r, sources);
        };
    }

    private static double[] rootMusic(Covariance r, int sources) {
        int m = r.size();
        double[][][] p = signalProjector(r, sources);
        int deg = 2 * m - 2;
        double[] cr = new double[deg + 1];
        double[] ci = new double[deg + 1];
        for (int k = -(m - 1); k <= m - 1; k++) {
            for (int i = 0; i < m; i++) {
                int j = i + k;
                if (j >= 0 && j < m) {
                    cr[k + m - 1] += (i == j ? 1 : 0) - p[0][i][j];
                    ci[k + m - 1] -= p[1][i][j];
                }
            }
        }
        double[][] roots = polynomialRoots(cr, ci);
        List<double[]> inside = new ArrayList<>();
        for (double[] z : roots) {
            double mod = Math.hypot(z[0], z[1]);
            if (mod <= 1 + 1e-9) {
                inside.add(new double[] {z[0], z[1], 1 - mod});
            }
        }
        inside.sort((a, b) -> Double.compare(a[2], b[2]));
        int k = Math.min(sources, inside.size());
        double[] out = new double[k];
        for (int j = 0; j < k; j++) {
            double arg = Math.atan2(inside.get(j)[1], inside.get(j)[0]);
            out[j] = Math.toDegrees(Math.asin(Math.max(-1, Math.min(1, arg / Math.PI))));
        }
        Arrays.sort(out);
        return out;
    }

    private static double[] esprit(Covariance r, int sources) {
        int m = r.size();
        double[][] vec = embeddedVectors(r, null);
        List<double[][]> basis = new ArrayList<>();
        for (int c = 0; c < 2 * sources && basis.size() < sources; c++) {
            double[] vr = Arrays.copyOfRange(vec[c], 0, m);
            double[] vi = Arrays.copyOfRange(vec[c], m, 2 * m);
            for (double[][] b : basis) {
                double pr = 0;
                double pi = 0;
                for (int i = 0; i < m; i++) {
                    pr += b[0][i] * vr[i] + b[1][i] * vi[i];
                    pi += b[0][i] * vi[i] - b[1][i] * vr[i];
                }
                for (int i = 0; i < m; i++) {
                    vr[i] -= pr * b[0][i] - pi * b[1][i];
                    vi[i] -= pr * b[1][i] + pi * b[0][i];
                }
            }
            double n = 0;
            for (int i = 0; i < m; i++) {
                n += vr[i] * vr[i] + vi[i] * vi[i];
            }
            if (n > 0.25) {
                n = Math.sqrt(n);
                for (int i = 0; i < m; i++) {
                    vr[i] /= n;
                    vi[i] /= n;
                }
                basis.add(new double[][] {vr, vi});
            }
        }
        int k = basis.size();
        double[][] e1r = new double[m - 1][k];
        double[][] e1i = new double[m - 1][k];
        double[][] e2r = new double[m - 1][k];
        double[][] e2i = new double[m - 1][k];
        for (int c = 0; c < k; c++) {
            for (int i = 0; i < m - 1; i++) {
                e1r[i][c] = basis.get(c)[0][i];
                e1i[i][c] = basis.get(c)[1][i];
                e2r[i][c] = basis.get(c)[0][i + 1];
                e2i[i][c] = basis.get(c)[1][i + 1];
            }
        }
        double[][][] g = mulH(e1r, e1i, e1r, e1i);
        double[][][] h = mulH(e1r, e1i, e2r, e2i);
        double[][][] gi = inverse(g[0], g[1]);
        double[][][] psi = mul(gi[0], gi[1], h[0], h[1]);
        double[][] lambda = eigenvaluesGeneral(psi[0], psi[1]);
        double[] out = new double[lambda.length];
        for (int j = 0; j < lambda.length; j++) {
            double arg = Math.atan2(lambda[j][1], lambda[j][0]);
            out[j] = Math.toDegrees(Math.asin(Math.max(-1, Math.min(1, arg / Math.PI))));
        }
        Arrays.sort(out);
        return out;
    }

    /** {@code A^H B}. */
    private static double[][][] mulH(double[][] ar, double[][] ai, double[][] br, double[][] bi) {
        int n = ar.length;
        int p = ar[0].length;
        int q = br[0].length;
        double[][] cr = new double[p][q];
        double[][] ci = new double[p][q];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < q; j++) {
                for (int k = 0; k < n; k++) {
                    cr[i][j] += ar[k][i] * br[k][j] + ai[k][i] * bi[k][j];
                    ci[i][j] += ar[k][i] * bi[k][j] - ai[k][i] * br[k][j];
                }
            }
        }
        return new double[][][] {cr, ci};
    }

    private static double[][][] mul(double[][] ar, double[][] ai, double[][] br, double[][] bi) {
        int p = ar.length;
        int n = br.length;
        int q = br[0].length;
        double[][] cr = new double[p][q];
        double[][] ci = new double[p][q];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < q; j++) {
                for (int k = 0; k < n; k++) {
                    cr[i][j] += ar[i][k] * br[k][j] - ai[i][k] * bi[k][j];
                    ci[i][j] += ar[i][k] * bi[k][j] + ai[i][k] * br[k][j];
                }
            }
        }
        return new double[][][] {cr, ci};
    }

    private static double[][][] inverse(double[][] re, double[][] im) {
        int m = re.length;
        double[][] inv = Matrices.inverse(embed(re, im));
        double[][] r = new double[m][m];
        double[][] i2 = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                r[i][j] = inv[i][j];
                i2[i][j] = inv[i + m][j];
            }
        }
        return new double[][][] {r, i2};
    }

    /**
     * Eigenvalues of a general complex matrix: the characteristic polynomial by Faddeev-LeVerrier,
     * then its roots.
     *
     * @param re the real part
     * @param im the imaginary part
     * @return the eigenvalues {@code [k][2]}
     */
    public static double[][] eigenvaluesGeneral(double[][] re, double[][] im) {
        int n = re.length;
        double[] cr = new double[n + 1];
        double[] ci = new double[n + 1];
        cr[n] = 1;
        double[][] mr = new double[n][n];
        double[][] mi = new double[n][n];
        for (int k = 1; k <= n; k++) {
            double[][][] am = mul(re, im, mr, mi);
            for (int i = 0; i < n; i++) {
                am[0][i][i] += cr[n - k + 1];
                am[1][i][i] += ci[n - k + 1];
            }
            mr = am[0];
            mi = am[1];
            double[][][] t = mul(re, im, mr, mi);
            double tr = 0;
            double ti = 0;
            for (int i = 0; i < n; i++) {
                tr += t[0][i][i];
                ti += t[1][i][i];
            }
            cr[n - k] = -tr / k;
            ci[n - k] = -ti / k;
        }
        return polynomialRoots(cr, ci);
    }

    /**
     * The roots of a complex polynomial {@code sum_j c_j z^j} by Durand-Kerner iteration.
     *
     * @param re the real parts of the coefficients, lowest degree first
     * @param im the imaginary parts
     * @return the roots {@code [degree][2]}
     */
    public static double[][] polynomialRoots(double[] re, double[] im) {
        int deg = re.length - 1;
        while (deg > 0 && Math.hypot(re[deg], im[deg]) < 1e-14) {
            deg--;
        }
        if (deg == 0) {
            return new double[0][];
        }
        double lr = re[deg];
        double li = im[deg];
        double l2 = lr * lr + li * li;
        double[] ar = new double[deg + 1];
        double[] ai = new double[deg + 1];
        for (int j = 0; j <= deg; j++) {
            ar[j] = (re[j] * lr + im[j] * li) / l2;
            ai[j] = (im[j] * lr - re[j] * li) / l2;
        }
        double[] zr = new double[deg];
        double[] zi = new double[deg];
        for (int j = 0; j < deg; j++) {
            double ang = 2 * Math.PI * j / deg + 0.4;
            zr[j] = 0.9 * Math.cos(ang);
            zi[j] = 0.9 * Math.sin(ang);
        }
        for (int it = 0; it < 1000; it++) {
            double change = 0;
            for (int j = 0; j < deg; j++) {
                double pr = 1;
                double pi = 0;
                for (int c = deg - 1; c >= 0; c--) {
                    double nr = pr * zr[j] - pi * zi[j] + ar[c];
                    double ni = pr * zi[j] + pi * zr[j] + ai[c];
                    pr = nr;
                    pi = ni;
                }
                double qr = 1;
                double qi = 0;
                for (int k = 0; k < deg; k++) {
                    if (k != j) {
                        double dr = zr[j] - zr[k];
                        double di = zi[j] - zi[k];
                        double nr = qr * dr - qi * di;
                        double ni = qr * di + qi * dr;
                        qr = nr;
                        qi = ni;
                    }
                }
                double q2 = qr * qr + qi * qi;
                if (q2 == 0) {
                    q2 = 1e-30;
                }
                double sr = (pr * qr + pi * qi) / q2;
                double si = (pi * qr - pr * qi) / q2;
                zr[j] -= sr;
                zi[j] -= si;
                change = Math.max(change, Math.hypot(sr, si));
            }
            if (change < 1e-13) {
                break;
            }
        }
        double[][] out = new double[deg][];
        for (int j = 0; j < deg; j++) {
            out[j] = new double[] {zr[j], zi[j]};
        }
        return out;
    }

    // ---------------------------------------------------------------- source number and bounds

    /**
     * The covariance's eigenvalues.
     *
     * @param r the covariance
     * @return the {@code M} eigenvalues, descending
     */
    public static double[] eigenvalues(Covariance r) {
        int m = r.size();
        double[] all = new double[2 * m];
        embeddedVectors(r, all);
        double[] out = new double[m];
        for (int i = 0; i < m; i++) {
            out[i] = all[2 * i];
        }
        return out;
    }

    /**
     * The number of sources by MDL or AIC (Wax &amp; Kailath 1985).
     *
     * @param mdl MDL (true) or AIC
     * @param eigenvalues the eigenvalues, descending
     * @param snapshots {@code T}
     * @return the estimate
     */
    public static int sourceCount(boolean mdl, double[] eigenvalues, int snapshots) {
        int m = eigenvalues.length;
        int best = 0;
        double bestScore = Double.MAX_VALUE;
        for (int k = 0; k < m; k++) {
            double logGeo = 0;
            double arith = 0;
            for (int i = k; i < m; i++) {
                double l = Math.max(eigenvalues[i], 1e-12);
                logGeo += Math.log(l) / (m - k);
                arith += l / (m - k);
            }
            double ll = -snapshots * (m - k) * (logGeo - Math.log(arith));
            double pen = k * (2.0 * m - k);
            double score = ll + (mdl ? 0.5 * pen * Math.log(snapshots) : pen);
            if (score < bestScore) {
                bestScore = score;
                best = k;
            }
        }
        return best;
    }

    /**
     * The stochastic Cramer-Rao bound (Stoica &amp; Nehorai 1990) for uncorrelated sources of
     * equal power in unit white noise.
     *
     * @param sensors {@code M}
     * @param anglesDeg the directions
     * @param snrDb the per-source SNR
     * @param snapshots {@code T}
     * @return the per-source standard deviation bound in degrees
     */
    public static double[] crb(int sensors, double[] anglesDeg, double snrDb, int snapshots) {
        int m = sensors;
        int k = anglesDeg.length;
        double p = Math.pow(10, snrDb / 10);
        double[][] ar = new double[m][k];
        double[][] ai = new double[m][k];
        double[][] dr = new double[m][k];
        double[][] di = new double[m][k];
        for (int j = 0; j < k; j++) {
            double th = Math.toRadians(anglesDeg[j]);
            for (int i = 0; i < m; i++) {
                double ph = Math.PI * i * Math.sin(th);
                ar[i][j] = Math.cos(ph);
                ai[i][j] = Math.sin(ph);
                double f = Math.PI * i * Math.cos(th);
                dr[i][j] = -f * ai[i][j];
                di[i][j] = f * ar[i][j];
            }
        }
        double[][][] aha = mulH(ar, ai, ar, ai);
        double[][][] ahaInv = inverse(aha[0], aha[1]);
        double[][][] t = mul(ar, ai, ahaInv[0], ahaInv[1]);
        double[][] ahR = new double[k][m];
        double[][] ahI = new double[k][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                ahR[j][i] = ar[i][j];
                ahI[j][i] = -ai[i][j];
            }
        }
        double[][][] proj = mul(t[0], t[1], ahR, ahI);
        double[][] perpR = new double[m][m];
        double[][] perpI = new double[m][m];
        double[][] rr = new double[m][m];
        double[][] ri = new double[m][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                perpR[i][j] = (i == j ? 1 : 0) - proj[0][i][j];
                perpI[i][j] = -proj[1][i][j];
                double sr = 0;
                double si = 0;
                for (int c = 0; c < k; c++) {
                    sr += p * (ar[i][c] * ar[j][c] + ai[i][c] * ai[j][c]);
                    si += p * (ai[i][c] * ar[j][c] - ar[i][c] * ai[j][c]);
                }
                rr[i][j] = sr + (i == j ? 1 : 0);
                ri[i][j] = si;
            }
        }
        double[][][] dpd = mulH(dr, di, mul(perpR, perpI, dr, di)[0], mul(perpR, perpI, dr, di)[1]);
        double[][][] rInv = inverse(rr, ri);
        double[][][] arA = mulH(ar, ai, mul(rInv[0], rInv[1], ar, ai)[0], mul(rInv[0], rInv[1], ar, ai)[1]);
        double[][] f = new double[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                double br = p * p * arA[0][j][i];
                double bi = p * p * arA[1][j][i];
                f[i][j] = 2 * snapshots * (dpd[0][i][j] * br - dpd[1][i][j] * bi);
            }
        }
        double[][] c = Matrices.inverse(f);
        double[] out = new double[k];
        for (int j = 0; j < k; j++) {
            out[j] = Math.toDegrees(Math.sqrt(Math.max(c[j][j], 0)));
        }
        return out;
    }

    /**
     * The spatial periodogram: each snapshot's zero-padded FFT over the array (core {@link Fft}),
     * powers averaged, the {@code K} highest peaks mapped to angles.
     *
     * @param x the snapshots
     * @param fftSize the padded length (a power of two)
     * @param sources {@code K}
     * @return the angles, ascending
     */
    public static double[] periodogram(ArraySignals.Snapshots x, int fftSize, int sources) {
        int m = x.sensors();
        int t = x.count();
        float[] data = new float[2 * fftSize * t];
        for (int s = 0; s < t; s++) {
            for (int i = 0; i < m; i++) {
                data[2 * (s * fftSize + i)] = (float) x.re()[i][s];
                data[2 * (s * fftSize + i) + 1] = (float) x.im()[i][s];
            }
        }
        Fft.transform(data, t, fftSize, false);
        double[] power = new double[fftSize];
        for (int s = 0; s < t; s++) {
            for (int f = 0; f < fftSize; f++) {
                double re = data[2 * (s * fftSize + f)];
                double im = data[2 * (s * fftSize + f) + 1];
                power[f] += re * re + im * im;
            }
        }
        double[] spec = new double[fftSize + 2];
        double[] ang = new double[fftSize + 2];
        for (int j = 0; j < fftSize + 2; j++) {
            int f = Math.floorMod(j - 1 - fftSize / 2, fftSize);
            double u = 2.0 * (j - 1 - fftSize / 2) / fftSize;
            spec[j] = power[f] + 1e-30;
            ang[j] = u;
        }
        List<Integer> found = new ArrayList<>();
        for (int j = 1; j < spec.length - 1; j++) {
            if (spec[j] > spec[j - 1] && spec[j] >= spec[j + 1]) {
                found.add(j);
            }
        }
        found.sort((a, b) -> Double.compare(spec[b], spec[a]));
        int k = Math.min(sources, found.size());
        double[] out = new double[k];
        for (int i = 0; i < k; i++) {
            int j = found.get(i);
            double l = Math.log(spec[j - 1]);
            double c = Math.log(spec[j]);
            double rr = Math.log(spec[j + 1]);
            double den = l - 2 * c + rr;
            double u = ang[j] + (den == 0 ? 0 : 0.5 * (l - rr) / den) * 2.0 / fftSize;
            out[i] = Math.toDegrees(Math.asin(Math.max(-1, Math.min(1, u))));
        }
        Arrays.sort(out);
        return out;
    }
}
