package io.github.zzuegg.brainiac.experimental.data;

import java.util.SplittableRandom;

/**
 * Synthetic sensing signals (wave 13, items 13.4.13-13.4.16):
 * <ul>
 *   <li>{@link #ula}: snapshots of a half-wavelength uniform linear array with {@code K}
 *       narrowband sources, uncorrelated or coherent, optional per-sensor gain/phase
 *       miscalibration, and a near-field (spherical wavefront) variant {@link #nearField};</li>
 *   <li>{@link #systemIdentification}: white or AR(1)-coloured input through an unknown FIR system
 *       that is fixed, switches abruptly or drifts;</li>
 *   <li>{@link #echo}: a speech-like coloured far-end signal through a synthetic room impulse
 *       response, with near-end double-talk bursts;</li>
 *   <li>{@link #sinusoids}: sums of sinusoids in AR(2) coloured noise, regularly or irregularly
 *       sampled, with the true noise spectrum.</li>
 * </ul>
 */
public final class ArraySignals {

    private ArraySignals() {
    }

    static double gaussian(SplittableRandom r) {
        double u = 1 - r.nextDouble();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r.nextDouble());
    }

    // ---------------------------------------------------------------- arrays

    /**
     * Complex array snapshots {@code [sensors][snapshots]}.
     *
     * @param re the real parts
     * @param im the imaginary parts
     * @param angles the true directions in degrees
     * @param ranges the true ranges in wavelengths (near field), or {@code null}
     */
    public record Snapshots(double[][] re, double[][] im, double[] angles, double[] ranges) {

        /** @return the number of sensors */
        public int sensors() {
            return re.length;
        }

        /** @return the number of snapshots */
        public int count() {
            return re[0].length;
        }
    }

    /**
     * Far-field snapshots {@code x(t) = A(theta) s(t) + n(t)} on a half-wavelength ULA,
     * {@code a_m(theta) = g_m exp(i (pi m sin theta + phi_m))}.
     *
     * @param sensors {@code M}
     * @param anglesDeg the directions in degrees
     * @param snapshots {@code T}
     * @param snrDb the per-source signal-to-noise ratio in dB
     * @param coherent whether all sources carry the same waveform (with random fixed phases)
     * @param miscalibration the standard deviation of the gain (relative) and phase (radians) errors
     * @param r the random source
     * @return the snapshots
     */
    public static Snapshots ula(int sensors, double[] anglesDeg, int snapshots, double snrDb, boolean coherent,
            double miscalibration, SplittableRandom r) {
        double[][] phase = new double[sensors][anglesDeg.length];
        double[] gain = new double[sensors];
        double[] offset = new double[sensors];
        for (int m = 0; m < sensors; m++) {
            gain[m] = 1 + miscalibration * gaussian(r);
            offset[m] = miscalibration * gaussian(r);
            for (int k = 0; k < anglesDeg.length; k++) {
                phase[m][k] = Math.PI * m * Math.sin(Math.toRadians(anglesDeg[k]));
            }
        }
        return synthesise(phase, gain, offset, snapshots, snrDb, coherent, anglesDeg, null, r);
    }

    /**
     * Near-field snapshots with spherical wavefronts: sensor {@code m} at {@code d_m = (m - (M-1)/2) / 2}
     * wavelengths, phase {@code -2 pi (|p_k - d_m| - r_k)} for a source at range {@code r_k} and
     * angle {@code theta_k} from the array centre.
     *
     * @param sensors {@code M}
     * @param anglesDeg the directions in degrees
     * @param rangesWavelengths the ranges in wavelengths
     * @param snapshots {@code T}
     * @param snrDb the per-source SNR in dB
     * @param r the random source
     * @return the snapshots
     */
    public static Snapshots nearField(int sensors, double[] anglesDeg, double[] rangesWavelengths, int snapshots,
            double snrDb, SplittableRandom r) {
        double[][] phase = new double[sensors][anglesDeg.length];
        double[] gain = new double[sensors];
        java.util.Arrays.fill(gain, 1);
        for (int m = 0; m < sensors; m++) {
            for (int k = 0; k < anglesDeg.length; k++) {
                phase[m][k] = nearFieldPhase(sensors, m, anglesDeg[k], rangesWavelengths[k]);
            }
        }
        return synthesise(phase, gain, new double[sensors], snapshots, snrDb, false, anglesDeg, rangesWavelengths,
                r);
    }

    /**
     * The near-field steering phase of one sensor.
     *
     * @param sensors {@code M}
     * @param m the sensor
     * @param angleDeg the direction
     * @param range the range in wavelengths
     * @return the phase in radians
     */
    public static double nearFieldPhase(int sensors, int m, double angleDeg, double range) {
        double d = (m - (sensors - 1) / 2.0) / 2;
        double th = Math.toRadians(angleDeg);
        double dist = Math.sqrt(range * range + d * d - 2 * range * d * Math.sin(th));
        return -2 * Math.PI * (dist - range);
    }

    private static Snapshots synthesise(double[][] phase, double[] gain, double[] offset, int snapshots,
            double snrDb, boolean coherent, double[] angles, double[] ranges, SplittableRandom r) {
        int m = phase.length;
        int k = angles.length;
        double amp = Math.pow(10, snrDb / 20);
        double[][] sre = new double[k][snapshots];
        double[][] sim = new double[k][snapshots];
        double[] fixed = new double[k];
        for (int j = 0; j < k; j++) {
            fixed[j] = 2 * Math.PI * r.nextDouble();
        }
        for (int t = 0; t < snapshots; t++) {
            double cr = gaussian(r) / Math.sqrt(2);
            double ci = gaussian(r) / Math.sqrt(2);
            for (int j = 0; j < k; j++) {
                if (coherent) {
                    sre[j][t] = amp * (cr * Math.cos(fixed[j]) - ci * Math.sin(fixed[j]));
                    sim[j][t] = amp * (cr * Math.sin(fixed[j]) + ci * Math.cos(fixed[j]));
                } else {
                    sre[j][t] = amp * gaussian(r) / Math.sqrt(2);
                    sim[j][t] = amp * gaussian(r) / Math.sqrt(2);
                }
            }
        }
        double[][] re = new double[m][snapshots];
        double[][] im = new double[m][snapshots];
        for (int s = 0; s < m; s++) {
            for (int t = 0; t < snapshots; t++) {
                double xr = gaussian(r) / Math.sqrt(2);
                double xi = gaussian(r) / Math.sqrt(2);
                for (int j = 0; j < k; j++) {
                    double p = phase[s][j] + offset[s];
                    double ar = gain[s] * Math.cos(p);
                    double ai = gain[s] * Math.sin(p);
                    xr += ar * sre[j][t] - ai * sim[j][t];
                    xi += ar * sim[j][t] + ai * sre[j][t];
                }
                re[s][t] = xr;
                im[s][t] = xi;
            }
        }
        return new Snapshots(re, im, angles.clone(), ranges == null ? null : ranges.clone());
    }

    // ---------------------------------------------------------------- adaptive filtering

    /**
     * A system-identification record: {@code d(n) = w(n)^T u(n) + v(n)}.
     *
     * @param input {@code u}
     * @param desired {@code d}
     * @param weights the true filter at each time {@code [n][taps]}
     */
    public record SystemRecord(double[] input, double[] desired, double[][] weights) {
    }

    /** How the unknown system changes. */
    public enum Change {
        /** Fixed. */
        NONE,
        /** Replaced by a new random system halfway. */
        ABRUPT,
        /** A random walk. */
        DRIFT
    }

    /**
     * System identification of a random FIR system with exponentially decaying taps.
     *
     * @param taps the filter length
     * @param n the samples
     * @param colour the AR(1) coefficient of the input (0 for white)
     * @param change the system's change
     * @param noise the measurement noise
     * @param r the random source
     * @return the record
     */
    public static SystemRecord systemIdentification(int taps, int n, double colour, Change change, double noise,
            SplittableRandom r) {
        double[] u = new double[n];
        double prev = 0;
        double scale = Math.sqrt(1 - colour * colour);
        for (int i = 0; i < n; i++) {
            prev = colour * prev + scale * gaussian(r);
            u[i] = prev;
        }
        double[] w = randomFilter(taps, r);
        double[] w2 = randomFilter(taps, r);
        double[][] ws = new double[n][];
        double[] d = new double[n];
        for (int i = 0; i < n; i++) {
            if (change == Change.ABRUPT && i == n / 2) {
                w = w2;
            } else if (change == Change.DRIFT) {
                w = w.clone();
                for (int j = 0; j < taps; j++) {
                    w[j] += 0.002 * gaussian(r) * Math.exp(-j / (taps / 3.0));
                }
            }
            ws[i] = w;
            double s = 0;
            for (int j = 0; j < taps && j <= i; j++) {
                s += w[j] * u[i - j];
            }
            d[i] = s + noise * gaussian(r);
        }
        return new SystemRecord(u, d, ws);
    }

    private static double[] randomFilter(int taps, SplittableRandom r) {
        double[] w = new double[taps];
        double n = 0;
        for (int j = 0; j < taps; j++) {
            w[j] = gaussian(r) * Math.exp(-j / (taps / 3.0));
            n += w[j] * w[j];
        }
        for (int j = 0; j < taps; j++) {
            w[j] /= Math.sqrt(n);
        }
        return w;
    }

    /**
     * An echo-cancellation record: {@code mic = h * far + near + noise}.
     *
     * @param far the far-end (loudspeaker) signal
     * @param mic the microphone signal
     * @param echo the echo component {@code h * far}
     * @param near the near-end talker (zero outside double-talk)
     * @param path the room impulse response
     */
    public record EchoRecord(double[] far, double[] mic, double[] echo, double[] near, double[] path) {
    }

    /**
     * Echo cancellation with a synthetic room impulse response (a delay, then exponentially decaying
     * random reflections), speech-like far-end and near-end signals (AR(2)-coloured noise with a
     * syllable-rate envelope), and double-talk in the second half when asked.
     *
     * @param taps the impulse response length
     * @param n the samples
     * @param doubleTalk whether the near end talks during the second half
     * @param noise the microphone noise
     * @param r the random source
     * @return the record
     */
    public static EchoRecord echo(int taps, int n, boolean doubleTalk, double noise, SplittableRandom r) {
        double[] h = new double[taps];
        int delay = taps / 8;
        double norm = 0;
        for (int j = delay; j < taps; j++) {
            h[j] = gaussian(r) * Math.exp(-(j - delay) / (taps / 5.0));
            norm += h[j] * h[j];
        }
        for (int j = 0; j < taps; j++) {
            h[j] *= 0.4 / Math.sqrt(norm);
        }
        double[] far = speech(n, r);
        double[] near = speech(n, r);
        double[] echo = new double[n];
        double[] mic = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < taps && j <= i; j++) {
                s += h[j] * far[i - j];
            }
            echo[i] = s;
            if (!doubleTalk || i < n / 2) {
                near[i] = 0;
            }
            mic[i] = s + near[i] + noise * gaussian(r);
        }
        return new EchoRecord(far, mic, echo, near, h);
    }

    private static double[] speech(int n, SplittableRandom r) {
        double[] x = new double[n];
        double a1 = 1.6;
        double a2 = -0.8;
        double x1 = 0;
        double x2 = 0;
        double phase = 2 * Math.PI * r.nextDouble();
        for (int i = 0; i < n; i++) {
            double v = a1 * x1 + a2 * x2 + gaussian(r);
            x2 = x1;
            x1 = v;
            double env = 0.3 + Math.pow(Math.max(Math.sin(2 * Math.PI * i / 800.0 + phase), 0), 2);
            x[i] = 0.25 * v * env;
        }
        return x;
    }

    // ---------------------------------------------------------------- spectra

    /**
     * Samples of sinusoids in AR(2) noise.
     *
     * @param t the sample times (unit spacing when regular)
     * @param x the values
     * @param frequencies the line frequencies in cycles per unit time
     * @param amplitudes the line amplitudes
     * @param ar the noise's AR(2) coefficients {@code {a1, a2}}
     * @param noise the noise's innovation standard deviation
     */
    public record Sinusoids(double[] t, double[] x, double[] frequencies, double[] amplitudes, double[] ar,
            double noise) {

        /**
         * The noise's power spectral density (two-sided, per unit frequency).
         *
         * @param f the frequency
         * @return {@code sigma^2 / |1 - a1 e^{-i w} - a2 e^{-2 i w}|^2}
         */
        public double noisePsd(double f) {
            double w = 2 * Math.PI * f;
            double re = 1 - ar[0] * Math.cos(w) - ar[1] * Math.cos(2 * w);
            double im = ar[0] * Math.sin(w) + ar[1] * Math.sin(2 * w);
            return noise * noise / (re * re + im * im);
        }
    }

    /**
     * Sums of sinusoids with random phases in AR(2) noise.
     *
     * @param n the samples
     * @param frequencies the line frequencies (cycles per sample, in {@code (0, 0.5)})
     * @param amplitudes their amplitudes
     * @param ar the AR(2) coefficients of the noise
     * @param noise the innovation standard deviation
     * @param irregular the share of samples dropped at random (plus two gaps of 10% each), 0 for regular
     * @param r the random source
     * @return the record
     */
    public static Sinusoids sinusoids(int n, double[] frequencies, double[] amplitudes, double[] ar, double noise,
            double irregular, SplittableRandom r) {
        int total = irregular > 0 ? (int) Math.round(n / (1 - irregular) / 0.8) : n;
        double[] phases = new double[frequencies.length];
        for (int k = 0; k < phases.length; k++) {
            phases[k] = 2 * Math.PI * r.nextDouble();
        }
        double[] full = new double[total];
        double x1 = 0;
        double x2 = 0;
        for (int burn = 0; burn < 200; burn++) {
            double v = ar[0] * x1 + ar[1] * x2 + noise * gaussian(r);
            x2 = x1;
            x1 = v;
        }
        for (int i = 0; i < total; i++) {
            double v = ar[0] * x1 + ar[1] * x2 + noise * gaussian(r);
            x2 = x1;
            x1 = v;
            double s = v;
            for (int k = 0; k < frequencies.length; k++) {
                s += amplitudes[k] * Math.cos(2 * Math.PI * frequencies[k] * i + phases[k]);
            }
            full[i] = s;
        }
        if (irregular <= 0) {
            double[] t = new double[n];
            for (int i = 0; i < n; i++) {
                t[i] = i;
            }
            return new Sinusoids(t, full, frequencies.clone(), amplitudes.clone(), ar.clone(), noise);
        }
        boolean[] keep = new boolean[total];
        int g1 = (int) (0.2 * total);
        int g2 = (int) (0.6 * total);
        int gap = (int) (0.1 * total);
        int count = 0;
        for (int i = 0; i < total; i++) {
            boolean inGap = (i >= g1 && i < g1 + gap) || (i >= g2 && i < g2 + gap);
            keep[i] = !inGap && r.nextDouble() >= irregular;
            if (keep[i]) {
                count++;
            }
        }
        double[] t = new double[count];
        double[] x = new double[count];
        int c = 0;
        for (int i = 0; i < total; i++) {
            if (keep[i]) {
                t[c] = i;
                x[c] = full[i];
                c++;
            }
        }
        return new Sinusoids(t, x, frequencies.clone(), amplitudes.clone(), ar.clone(), noise);
    }
}
