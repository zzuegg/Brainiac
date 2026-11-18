package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Padding;
import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Seismic phase pickers for three-component traces (Z, N, E).
 *
 * <ul>
 *   <li>{@link #staLta}: the ratio of a short-term to a long-term average of the squared
 *       signal (Allen, BSSA 1978); P is the first trigger on Z, S the first trigger on the
 *       horizontal energy after the P coda has started.</li>
 *   <li>{@link #arAic}: the STA/LTA trigger refined by the AR-AIC picker (Sleeman and van Eck,
 *       PEPI 1999): an autoregressive model fitted to the stretch before the window predicts
 *       the window, and the onset is the split of the prediction residuals that minimises
 *       {@code k log var_1 + (n - k - 1) log var_2}.</li>
 *   <li>{@link PhaseNet}: a small 1-D U-Net (Zhu and Beroza, GJI 2019) built from core
 *       {@link Conv} layers (strided convolutions down, transposed convolutions up, skip
 *       concatenations), trained by cross-entropy against Gaussian-shaped P, S and noise
 *       targets; picks are the peaks of the P and S probability traces.</li>
 * </ul>
 */
public final class PhasePicking {

    /**
     * Picks of one trace.
     *
     * @param p the P sample, or -1
     * @param s the S sample, or -1
     */
    public record Picks(int p, int s) {
    }

    private PhasePicking() {
    }

    /**
     * The STA/LTA ratio of a characteristic function (the LTA window ends where the STA
     * window starts).
     *
     * @param cf a non-negative characteristic function such as the squared signal
     * @param sta the short window in samples
     * @param lta the long window in samples
     * @return the ratio per sample (0 before three STA windows)
     */
    public static double[] ratio(double[] cf, int sta, int lta) {
        int n = cf.length;
        double[] c = new double[n + 1];
        for (int i = 0; i < n; i++) {
            c[i + 1] = c[i] + cf[i];
        }
        double[] r = new double[n];
        // the LTA window may be truncated at the start of the trace, down to two STA windows
        for (int t = 3 * sta - 1; t < n; t++) {
            double s = (c[t + 1] - c[t + 1 - sta]) / sta;
            int from = Math.max(0, t + 1 - sta - lta);
            double l = (c[t + 1 - sta] - c[from]) / (t + 1 - sta - from);
            r[t] = s / (l + 1e-12);
        }
        return r;
    }

    private static double[] energy(double[][] x, int... channels) {
        double[] e = new double[x[0].length];
        for (int c : channels) {
            for (int i = 0; i < e.length; i++) {
                e[i] += x[c][i] * x[c][i];
            }
        }
        return e;
    }

    private static int firstAbove(double[] r, double on, int from) {
        for (int t = Math.max(0, from); t < r.length; t++) {
            if (r[t] > on) {
                return t;
            }
        }
        return -1;
    }

    /**
     * STA/LTA picks.
     *
     * @param x {@code [3][samples]}
     * @param rate the sampling rate
     * @param on the trigger threshold
     * @return the picks
     */
    public static Picks staLta(double[][] x, double rate, double on) {
        int sta = (int) Math.round(rate);
        int lta = (int) Math.round(5 * rate);
        int p = firstAbove(ratio(energy(x, 0), sta, lta), on, 0);
        if (p < 0) {
            return new Picks(-1, -1);
        }
        int s = firstAbove(ratio(energy(x, 1, 2), sta, lta), on, p + (int) (0.5 * rate));
        return new Picks(p, s);
    }

    /**
     * STA/LTA triggers refined by AR-AIC.
     *
     * @param x {@code [3][samples]}
     * @param rate the sampling rate
     * @param on the STA/LTA trigger threshold
     * @param order the autoregressive order
     * @return the picks
     */
    public static Picks arAic(double[][] x, double rate, double on, int order) {
        Picks trig = staLta(x, rate, on);
        if (trig.p() < 0) {
            return trig;
        }
        int n = x[0].length;
        int fit = (int) rate;
        int pFrom = Math.max(fit + order, trig.p() - (int) (2 * rate));
        int pTo = Math.min(n, trig.p() + (int) (0.5 * rate));
        int p = aicSplit(x, new int[] {0}, pFrom - fit, pFrom, pTo, order);
        int sFrom = p + (int) (0.5 * rate);
        int sTo = Math.min(n, p + (int) (8 * rate));
        int s = sTo - sFrom > 2 * order + 4 ? aicSplit(x, new int[] {1, 2}, sFrom - fit, sFrom, sTo, order) : -1;
        return new Picks(p, s);
    }

    /**
     * The AR-AIC onset: fits AR({@code order}) by least squares on {@code [fitFrom, from)},
     * predicts {@code [from, to)} and returns the split of the residuals with the least AIC.
     */
    static int aicSplit(double[][] x, int[] channels, int fitFrom, int from, int to, int order) {
        fitFrom = Math.max(order, fitFrom);
        double[][] ata = new double[order][order];
        double[] atb = new double[order];
        for (int c : channels) {
            for (int t = fitFrom; t < from; t++) {
                for (int i = 0; i < order; i++) {
                    atb[i] += x[c][t - 1 - i] * x[c][t];
                    for (int j = 0; j < order; j++) {
                        ata[i][j] += x[c][t - 1 - i] * x[c][t - 1 - j];
                    }
                }
            }
        }
        for (int i = 0; i < order; i++) {
            ata[i][i] += 1e-6 * (1 + ata[i][i]);
        }
        double[][] inv = Matrices.inverse(ata);
        double[] a = new double[order];
        for (int i = 0; i < order; i++) {
            for (int j = 0; j < order; j++) {
                a[i] += inv[i][j] * atb[j];
            }
        }
        int m = to - from;
        double[] e2 = new double[m];
        for (int c : channels) {
            for (int t = from; t < to; t++) {
                double pred = 0;
                for (int i = 0; i < order; i++) {
                    pred += a[i] * x[c][t - 1 - i];
                }
                double e = x[c][t] - pred;
                e2[t - from] += e * e;
            }
        }
        double[] cum = new double[m + 1];
        for (int i = 0; i < m; i++) {
            cum[i + 1] = cum[i] + e2[i];
        }
        double best = Double.POSITIVE_INFINITY;
        int arg = from;
        for (int k = order + 1; k < m - order - 1; k++) {
            double v1 = cum[k] / k;
            double v2 = (cum[m] - cum[k]) / (m - k);
            double aic = k * Math.log(v1 + 1e-12) + (m - k - 1) * Math.log(v2 + 1e-12);
            if (aic < best) {
                best = aic;
                arg = from + k;
            }
        }
        return arg;
    }

    /** A PhaseNet-style 1-D U-Net picker. */
    public static final class PhaseNet {

        private final List<Layer> layers = new ArrayList<>();
        private final Layer in1;
        private final Layer down1;
        private final Layer down2;
        private final Layer up2;
        private final Layer mid2;
        private final Layer up1;
        private final Layer mid1;
        private final Layer out;

        /**
         * An untrained network.
         *
         * @param width the channels of the first level
         * @param seed the seed
         */
        public PhaseNet(int width, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            int length = 64;
            int w = width;
            in1 = add(Conv.of(w, 7).padding(Padding.SAME).activation(Activation.RELU).create(Shape.of(3, length), r));
            down1 = add(Conv.of(2 * w, 7).stride(2).padding(Padding.SAME).activation(Activation.RELU)
                    .create(Shape.of(w, length), r));
            down2 = add(Conv.of(4 * w, 7).stride(2).padding(Padding.SAME).activation(Activation.RELU)
                    .create(Shape.of(2 * w, length / 2), r));
            up2 = add(Conv.transposed(2 * w, 2).stride(2).activation(Activation.RELU)
                    .create(Shape.of(4 * w, length / 4), r));
            mid2 = add(Conv.of(2 * w, 7).padding(Padding.SAME).activation(Activation.RELU)
                    .create(Shape.of(4 * w, length / 2), r));
            up1 = add(Conv.transposed(w, 2).stride(2).activation(Activation.RELU)
                    .create(Shape.of(2 * w, length / 2), r));
            mid1 = add(Conv.of(w, 7).padding(Padding.SAME).activation(Activation.RELU)
                    .create(Shape.of(2 * w, length), r));
            out = add(Conv.of(3, 1).create(Shape.of(w, length), r));
        }

        private Layer add(Layer l) {
            layers.add(l);
            return l;
        }

        private List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            for (Layer l : layers) {
                p.addAll(l.parameters());
            }
            return p;
        }

        private Var logits(Var x) {
            Var a = in1.forward(x);
            Var b = down1.forward(a);
            Var c = down2.forward(b);
            Var u2 = mid2.forward(Var.concat(1, up2.forward(c), b));
            Var u1 = mid1.forward(Var.concat(1, up1.forward(u2), a));
            return out.forward(u1);
        }

        private static float[] normalised(double[][] x) {
            int n = x[0].length;
            float[] v = new float[3 * n];
            for (int c = 0; c < 3; c++) {
                double m = 0;
                double s = 0;
                for (double d : x[c]) {
                    m += d / n;
                }
                for (double d : x[c]) {
                    s += (d - m) * (d - m) / n;
                }
                s = Math.sqrt(s) + 1e-9;
                for (int i = 0; i < n; i++) {
                    v[c * n + i] = (float) ((x[c][i] - m) / s);
                }
            }
            return v;
        }

        /**
         * Trains by cross-entropy on Gaussian targets ({@code sigma} 0.1 s), on random crops
         * that contain the P or the S arrival when there is one (the network is fully convolutional, so
         * it then runs on whole traces).
         *
         * @param traces training traces
         * @param rate the sampling rate
         * @param length the crop length (a multiple of 4)
         * @param steps Adam steps
         * @param batch traces per step
         * @param learningRate the step
         * @param seed the seed
         * @return this network
         */
        public PhaseNet train(List<ScienceSims.SeismicTrace> traces, double rate, int length, int steps, int batch,
                double learningRate, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            Optimizer opt = Optimizer.adam((float) learningRate);
            List<Var> params = parameters();
            double sigma = 0.1 * rate;
            for (int s = 0; s < steps; s++) {
                float[] in = new float[batch * 3 * length];
                float[] target = new float[batch * 3 * length];
                for (int b = 0; b < batch; b++) {
                    ScienceSims.SeismicTrace tr = traces.get(r.nextInt(traces.size()));
                    int full = tr.data()[0].length;
                    int anchor = tr.pIndex() < 0 ? -1 : r.nextBoolean() ? tr.pIndex() : tr.sIndex();
                    int start = anchor < 0 ? r.nextInt(full - length + 1)
                            : Math.clamp(anchor - (long) r.nextInt(length), 0, full - length);
                    float[] all = normalised(tr.data());
                    for (int c = 0; c < 3; c++) {
                        System.arraycopy(all, c * full + start, in, (b * 3 + c) * length, length);
                    }
                    for (int j = 0; j < length; j++) {
                        int i = start + j;
                        double gp = tr.pIndex() < 0 ? 0 : Math.exp(-0.5 * Math.pow((i - tr.pIndex()) / sigma, 2));
                        double gs = tr.sIndex() < 0 ? 0 : Math.exp(-0.5 * Math.pow((i - tr.sIndex()) / sigma, 2));
                        target[(b * 3) * length + j] = (float) gp;
                        target[(b * 3 + 1) * length + j] = (float) gs;
                        target[(b * 3 + 2) * length + j] = (float) Math.max(0, 1 - gp - gs);
                    }
                }
                Tensor xt = Tensor.of(Shape.of(batch, 3, length), in);
                Tensor yt = Tensor.of(Shape.of(batch, 3, length), target);
                opt.minimize(params, () -> logits(Var.of(xt)).logSoftmax(1).mul(yt).sum().neg()
                        .mul(1f / (batch * length)));
            }
            return this;
        }

        /**
         * @param x {@code [3][length]}, the length a multiple of 4
         * @return {@code [3][length]} probabilities of P, S and noise
         */
        public double[][] probabilities(double[][] x) {
            int length = x[0].length;
            Tensor xt = Tensor.of(Shape.of(1, 3, length), normalised(x));
            float[] p = Var.noGrad(() -> logits(Var.of(xt)).softmax(1)).value().toArray();
            double[][] out = new double[3][length];
            for (int c = 0; c < 3; c++) {
                for (int i = 0; i < length; i++) {
                    out[c][i] = p[c * length + i];
                }
            }
            return out;
        }

        /**
         * Picks the highest P and S probability peaks above a threshold.
         *
         * @param x {@code [3][length]}
         * @param threshold the probability threshold
         * @return the picks
         */
        public Picks pick(double[][] x, double threshold) {
            return picks(probabilities(x), threshold);
        }

        /**
         * Picks from precomputed probabilities.
         *
         * @param p {@code [3][length]} from {@link #probabilities}
         * @param threshold the probability threshold
         * @return the picks
         */
        public static Picks picks(double[][] p, double threshold) {
            return new Picks(argmaxAbove(p[0], threshold), argmaxAbove(p[1], threshold));
        }

        /**
         * @param p {@code [3][length]} from {@link #probabilities}
         * @return the highest P or S probability, a trace-level detection score
         */
        public static double detectionScore(double[][] p) {
            double m = 0;
            for (int i = 0; i < p[0].length; i++) {
                m = Math.max(m, Math.max(p[0][i], p[1][i]));
            }
            return m;
        }

        private static int argmaxAbove(double[] v, double threshold) {
            int arg = -1;
            double best = threshold;
            for (int i = 0; i < v.length; i++) {
                if (v[i] > best) {
                    best = v[i];
                    arg = i;
                }
            }
            return arg;
        }
    }
}
