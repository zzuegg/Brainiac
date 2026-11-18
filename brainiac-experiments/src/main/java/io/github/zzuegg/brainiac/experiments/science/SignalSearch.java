package io.github.zzuegg.brainiac.experiments.science;

import io.github.zzuegg.brainiac.experimental.anomaly.Pelt;
import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.experimental.science.MatchedFilter;
import io.github.zzuegg.brainiac.experimental.science.PhasePicking;
import io.github.zzuegg.brainiac.experimental.science.TransitSearch;
import io.github.zzuegg.brainiac.experimental.science.TransitVetter;
import io.github.zzuegg.brainiac.experimental.timeseries.RandomConvolutionFeatures;
import io.github.zzuegg.brainiac.experimental.timeseries.RidgeClassifier;
import io.github.zzuegg.brainiac.experimental.train.RidgeRegression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Searches for known signal shapes in noise (wave 13, items 13.3.13–13.3.15): seismic phase
 * picking, chirp detection by matched filtering, and transit search with CNN vetting.
 */
public final class SignalSearch {

    private static final double[] SNRS = {-5, 0, 5, 10, 20};

    private SignalSearch() {
    }

    // ================================================================== 13.3.13 phase picking

    /**
     * Phase picking: STA/LTA, AR-AIC and the PhaseNet-style U-Net against PELT on the log
     * energy, and the trace-level detection task against ROCKET with a ridge classifier.
     *
     * @param rate the sampling rate (30 s traces)
     * @param train training traces (events and a fifth noise-only)
     * @param test test traces per SNR level, plus as many noise-only traces in total
     * @param steps U-Net Adam steps
     * @param seed the seed
     * @return method to {P recall, P precision, P residual sd (s), S recall, S precision, S
     *     residual sd (s), false picks on noise, S recall at SNR {@code <= 0} dB}
     */
    public static Map<String, double[]> picking(double rate, int train, int test, int steps, long seed) {
        int n = (int) (30 * rate);
        SplittableRandom r = new SplittableRandom(seed);
        List<ScienceSims.SeismicTrace> tr = new ArrayList<>();
        for (int i = 0; i < train; i++) {
            tr.add(ScienceSims.seismicTrace(n, rate, SNRS[r.nextInt(SNRS.length)], i % 5 != 0, seed * 1000 + i));
        }
        List<ScienceSims.SeismicTrace> te = new ArrayList<>();
        for (double snr : SNRS) {
            for (int i = 0; i < test; i++) {
                te.add(ScienceSims.seismicTrace(n, rate, snr, true, seed * 7919 + te.size()));
            }
        }
        for (int i = 0; i < test; i++) {
            te.add(ScienceSims.seismicTrace(n, rate, 0, false, seed * 7919 + te.size()));
        }
        PhasePicking.PhaseNet net = new PhasePicking.PhaseNet(8, seed).train(tr, rate, 256, steps, 8, 5e-3, seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        int tol = (int) Math.round(0.1 * rate);
        double[][][] probs = new double[te.size()][][];
        for (int i = 0; i < te.size(); i++) {
            probs[i] = net.probabilities(te.get(i).data());
        }
        String[] names = {"STA/LTA", "AR-AIC", "PhaseNet-lite U-Net", "PELT on log energy"};
        for (String name : names) {
            PickScore sc = new PickScore();
            for (int i = 0; i < te.size(); i++) {
                ScienceSims.SeismicTrace t = te.get(i);
                PhasePicking.Picks p = switch (name) {
                    case "STA/LTA" -> PhasePicking.staLta(t.data(), rate, 3);
                    case "AR-AIC" -> PhasePicking.arAic(t.data(), rate, 3, 4);
                    case "PhaseNet-lite U-Net" -> PhasePicking.PhaseNet.picks(probs[i], 0.3);
                    default -> pelt(t.data(), rate);
                };
                sc.add(t, p, tol, rate);
            }
            out.put(name, sc.result());
        }
        // trace-level detection: event vs noise
        int[] trLabel = new int[train];
        float[][][] trX = new float[train][][];
        for (int i = 0; i < train; i++) {
            trLabel[i] = tr.get(i).pIndex() >= 0 ? 1 : 0;
            trX[i] = toFloat(tr.get(i).data());
        }
        float[][][] teX = new float[te.size()][][];
        int[] teLabel = new int[te.size()];
        for (int i = 0; i < te.size(); i++) {
            teX[i] = toFloat(te.get(i).data());
            teLabel[i] = te.get(i).pIndex() >= 0 ? 1 : 0;
        }
        RandomConvolutionFeatures rocket = RandomConvolutionFeatures.rocket(100, n, 3, seed);
        RidgeClassifier ridge = RidgeClassifier.fit(rocket.transform(trX), trLabel, 2, 10);
        int[] pr = ridge.predict(rocket.transform(teX));
        int[] staDet = new int[te.size()];
        int[] netDet = new int[te.size()];
        for (int i = 0; i < te.size(); i++) {
            staDet[i] = PhasePicking.staLta(te.get(i).data(), rate, 3).p() >= 0 ? 1 : 0;
            netDet[i] = PhasePicking.PhaseNet.detectionScore(probs[i]) > 0.3 ? 1 : 0;
        }
        out.put("trace detection accuracy {STA/LTA, U-Net, ROCKET + ridge}",
                new double[] {RidgeClassifier.accuracy(staDet, teLabel), RidgeClassifier.accuracy(netDet, teLabel),
                    RidgeClassifier.accuracy(pr, teLabel)});
        return out;
    }

    // ================================================================== 13.3.14 chirps

    private static final double CHIRP_RATE = 256;
    private static final int CHIRP_N = 256;
    private static final double F0 = 30;
    private static final double FMAX = 120;
    private static final double PHI = 0.9;

    private static double[] segment(double snr, double tauLo, double tauHi, double glitchSnr, MatchedFilter truth,
            double[] tauOut, SplittableRandom r) {
        double[] x = ScienceSims.colouredNoise(CHIRP_N, PHI, r);
        if (snr > 0) {
            double tau = tauLo + (tauHi - tauLo) * r.nextDouble();
            int start = r.nextInt(Math.max(1, CHIRP_N - (int) (tau * CHIRP_RATE) - 8));
            double[] h = ScienceSims.chirp(CHIRP_N, CHIRP_RATE, tau, F0, FMAX, start, 2 * Math.PI * r.nextDouble());
            double a = snr / truth.optimalSnr(h);
            for (int i = 0; i < CHIRP_N; i++) {
                x[i] += a * h[i];
            }
            if (tauOut != null) {
                tauOut[0] = tau;
            }
        }
        if (glitchSnr > 0) {
            double[] g = ScienceSims.glitch(CHIRP_N, CHIRP_RATE, 32 + r.nextInt(CHIRP_N - 64),
                    40 + 60 * r.nextDouble(), 0.01 + 0.02 * r.nextDouble());
            double a = glitchSnr / truth.optimalSnr(g);
            for (int i = 0; i < CHIRP_N; i++) {
                x[i] += a * g[i];
            }
        }
        return x;
    }

    /**
     * Chirp detection in coloured noise: the matched filter over a minimal-match bank (with and
     * without the chi-squared veto), a 1-D CNN on whitened data, an excess-power detector and
     * ROCKET with ridge regression. Thresholds come from Gaussian noise-only segments.
     *
     * @param noiseSegments noise-only segments for the thresholds
     * @param perSnr signal segments per SNR in {4, 6, 8, 10, 12}
     * @param trainSegments CNN and ROCKET training segments (half with chirps at SNR 5-12)
     * @param cnnSteps CNN Adam steps
     * @param seed the seed
     * @return detector to {Pd at FAR 1e-2 for SNR 4, 6, 8, 10, 12; Pd at the loudest-noise
     *     threshold for SNR 8; Pd at FAR 1e-2 for out-of-range chirps at SNR 8; false-alarm rate
     *     on glitches at the FAR 1e-2 threshold; microseconds per second of data}, plus the bank
     *     size and the chirp-duration error
     */
    public static Map<String, double[]> chirps(int noiseSegments, int perSnr, int trainSegments, int cnnSteps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        MatchedFilter truth = new MatchedFilter(MatchedFilter.ar1Psd(CHIRP_N, PHI), CHIRP_RATE, F0, FMAX);
        double[] psd = MatchedFilter.welch(ScienceSims.colouredNoise(CHIRP_N * 16, PHI, r), CHIRP_N);
        double tauLo = 0.25;
        double tauHi = 0.7;
        MatchedFilter mf = new MatchedFilter(psd, CHIRP_RATE, F0, FMAX).placeBank(tauLo, tauHi, 0.9);
        double[] snrs = {4, 6, 8, 10, 12};
        List<double[]> eval = new ArrayList<>();
        List<String> kind = new ArrayList<>();
        List<Double> tauTrue = new ArrayList<>();
        List<Double> snrOf = new ArrayList<>();
        for (int i = 0; i < noiseSegments; i++) {
            eval.add(segment(0, 0, 0, 0, truth, null, r));
            kind.add("noise");
            tauTrue.add(Double.NaN);
            snrOf.add(0.0);
        }
        for (double snr : snrs) {
            for (int i = 0; i < perSnr; i++) {
                double[] t = new double[1];
                eval.add(segment(snr, tauLo, tauHi, 0, truth, t, r));
                kind.add("in");
                tauTrue.add(t[0]);
                snrOf.add(snr);
            }
        }
        for (int i = 0; i < perSnr; i++) {
            eval.add(segment(8, 0.7, 0.95, 0, truth, null, r));
            kind.add("out");
            tauTrue.add(Double.NaN);
            snrOf.add(8.0);
        }
        for (int i = 0; i < perSnr * 2; i++) {
            eval.add(segment(0, 0, 0, 10 + 10 * r.nextDouble(), truth, null, r));
            kind.add("glitch");
            tauTrue.add(Double.NaN);
            snrOf.add(0.0);
        }
        int m = eval.size();
        double[][] whitened = new double[m][];
        for (int i = 0; i < m; i++) {
            whitened[i] = mf.whiten(eval.get(i));
        }
        // training set for the CNN and ROCKET: Gaussian noise and in-range chirps, no glitches
        double[][] trW = new double[trainSegments][];
        int[] trY = new int[trainSegments];
        for (int i = 0; i < trainSegments; i++) {
            trY[i] = i % 2;
            trW[i] = mf.whiten(segment(trY[i] == 1 ? 5 + 7 * r.nextDouble() : 0, tauLo, tauHi, 0, truth, null, r));
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        double seconds = CHIRP_N / CHIRP_RATE;
        long t0 = System.nanoTime();
        double[] mfSnr = new double[m];
        double[] mfRw = new double[m];
        double tauErr = 0;
        int tauCount = 0;
        for (int i = 0; i < m; i++) {
            MatchedFilter.Detection d = mf.search(eval.get(i));
            mfSnr[i] = d.snr();
            mfRw[i] = d.reweighted();
            if ("in".equals(kind.get(i)) && snrOf.get(i) >= 8) {
                tauErr += Math.abs(d.tau() - tauTrue.get(i)) / tauTrue.get(i);
                tauCount++;
            }
        }
        double mfMicros = (System.nanoTime() - t0) / 1e3 / m / seconds;
        out.put("matched filter", roc(mfSnr, kind, snrOf, snrs, mfMicros));
        out.put("matched filter + chi2 veto", roc(mfRw, kind, snrOf, snrs, mfMicros));
        long t1 = System.nanoTime();
        MatchedFilter.CnnDetector cnn = new MatchedFilter.CnnDetector(seed).train(trW, trY, cnnSteps, 16, 3e-3, seed);
        long t2 = System.nanoTime();
        double[] cnnScore = cnn.scores(whitened);
        double cnnMicros = (System.nanoTime() - t2) / 1e3 / m / seconds;
        out.put("1-D CNN", roc(cnnScore, kind, snrOf, snrs, cnnMicros));
        t0 = System.nanoTime();
        double[] ep = new double[m];
        for (int i = 0; i < m; i++) {
            ep[i] = mf.excessPower(eval.get(i), (int) (0.25 * CHIRP_RATE));
        }
        out.put("excess power", roc(ep, kind, snrOf, snrs, (System.nanoTime() - t0) / 1e3 / m / seconds));
        RandomConvolutionFeatures rocket = RandomConvolutionFeatures.rocket(100, CHIRP_N, 1, seed);
        double[][] fTr = rocket.transform(univariate(trW));
        t0 = System.nanoTime();
        double[][] fTe = rocket.transform(univariate(whitened));
        double[] rs = ridgeScores(fTr, trY, fTe, 1.0);
        out.put("ROCKET + ridge", roc(rs, kind, snrOf, snrs, (System.nanoTime() - t0) / 1e3 / m / seconds));
        out.put("bank {templates, mean |tau error| / tau at SNR >= 8, CNN training s}",
                new double[] {mf.bank().length, tauErr / Math.max(1, tauCount), (t2 - t1) / 1e9});
        return out;
    }

    private static float[][][] univariate(double[][] x) {
        float[][][] f = new float[x.length][1][];
        for (int i = 0; i < x.length; i++) {
            f[i][0] = new float[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                f[i][0][j] = (float) x[i][j];
            }
        }
        return f;
    }

    static double[] ridgeScores(double[][] train, int[] labels, double[][] test, double lambda) {
        int d = train[0].length;
        double[] mean = new double[d];
        double[] sd = new double[d];
        for (double[] row : train) {
            for (int j = 0; j < d; j++) {
                mean[j] += row[j] / train.length;
            }
        }
        for (double[] row : train) {
            for (int j = 0; j < d; j++) {
                sd[j] += (row[j] - mean[j]) * (row[j] - mean[j]) / train.length;
            }
        }
        for (int j = 0; j < d; j++) {
            sd[j] = Math.sqrt(sd[j]) + 1e-9;
        }
        RidgeRegression rr = RidgeRegression.of(d + 1, 1);
        for (int i = 0; i < train.length; i++) {
            rr.add(standardise(train[i], mean, sd), new double[] {2 * labels[i] - 1});
        }
        double[][] w = rr.solve(lambda);
        double[] out = new double[test.length];
        for (int i = 0; i < test.length; i++) {
            out[i] = RidgeRegression.predict(w, standardise(test[i], mean, sd))[0];
        }
        return out;
    }

    private static double[] standardise(double[] x, double[] mean, double[] sd) {
        double[] z = new double[x.length + 1];
        for (int j = 0; j < x.length; j++) {
            z[j] = (x[j] - mean[j]) / sd[j];
        }
        z[x.length] = 1;
        return z;
    }

    private static double[] roc(double[] score, List<String> kind, List<Double> snr, double[] snrs, double micros) {
        List<Double> noise = new ArrayList<>();
        for (int i = 0; i < score.length; i++) {
            if ("noise".equals(kind.get(i))) {
                noise.add(score[i]);
            }
        }
        double[] ns = noise.stream().mapToDouble(Double::doubleValue).toArray();
        Arrays.sort(ns);
        double thr = ns[(int) Math.floor(0.99 * (ns.length - 1))];
        double loudest = ns[ns.length - 1];
        double[] out = new double[snrs.length + 4];
        for (int k = 0; k < snrs.length; k++) {
            int hit = 0;
            int all = 0;
            for (int i = 0; i < score.length; i++) {
                if ("in".equals(kind.get(i)) && snr.get(i) == snrs[k]) {
                    all++;
                    hit += score[i] > thr ? 1 : 0;
                }
            }
            out[k] = hit / (double) all;
        }
        int hit = 0;
        int all = 0;
        int outHit = 0;
        int outAll = 0;
        int gl = 0;
        int glAll = 0;
        for (int i = 0; i < score.length; i++) {
            if ("in".equals(kind.get(i)) && snr.get(i) == 8) {
                all++;
                hit += score[i] > loudest ? 1 : 0;
            } else if ("out".equals(kind.get(i))) {
                outAll++;
                outHit += score[i] > thr ? 1 : 0;
            } else if ("glitch".equals(kind.get(i))) {
                glAll++;
                gl += score[i] > thr ? 1 : 0;
            }
        }
        out[snrs.length] = hit / (double) Math.max(1, all);
        out[snrs.length + 1] = outHit / (double) Math.max(1, outAll);
        out[snrs.length + 2] = gl / (double) Math.max(1, glAll);
        out[snrs.length + 3] = micros;
        return out;
    }

    // ================================================================== 13.3.15 transits

    private static double[] uniformGrid(double[] time, double[] y, double cadence, int length) {
        double[] s = new double[length];
        int[] c = new int[length];
        for (int i = 0; i < time.length; i++) {
            int k = (int) Math.round((time[i] - time[0]) / cadence);
            if (k < length) {
                s[k] += y[i];
                c[k]++;
            }
        }
        for (int k = 0; k < length; k++) {
            s[k] = c[k] > 0 ? s[k] / c[k] : 0;
        }
        return s;
    }

    /**
     * Transit search and vetting on 90-day light curves at a 30-minute cadence with gaps, red
     * noise and stellar rotation: BLS after moving-median or GP detrending against the FFT
     * periodogram (recovered-period rate within 1% by transit SNR), and the two-view CNN vetter
     * against ROCKET with ridge regression on the raw (detrended, regridded) light curve.
     *
     * @param planets light curves for the period-recovery table
     * @param vetTrain vetting training curves (half planets, half eclipsing binaries)
     * @param vetTest vetting test curves
     * @param vetSteps vetter Adam steps
     * @param seed the seed
     * @return rows of recovery rates by SNR bin and vetting scores
     */
    public static Map<String, double[]> transits(int planets, int vetTrain, int vetTest, int vetSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double days = 90;
        double noise = 0.001;
        double[] edges = {4, 7, 10, 20};
        int bins = edges.length - 1;
        double[][] hits = new double[3][bins];
        int[] totals = new int[bins];
        double blsSeconds = 0;
        for (int i = 0; i < planets; i++) {
            double period = Math.exp(Math.log(1) + (Math.log(20) - Math.log(1)) * r.nextDouble());
            int bin = i % bins;
            double snr = edges[bin] + (edges[bin + 1] - edges[bin]) * r.nextDouble();
            double duration = 0.08 * Math.pow(period, 1.0 / 3);
            double nIn = 0.9 * days / period * duration * 48;
            double depth = snr * noise / Math.sqrt(nIn);
            ScienceSims.LightCurve lc = ScienceSims.lightCurve(days, "planet", period, depth, noise, 0.005,
                    seed * 31 + i);
            totals[bin]++;
            long t0 = System.nanoTime();
            double[] med = TransitSearch.detrendMedian(lc.time(), lc.flux(), 1.0);
            TransitSearch.Result a = TransitSearch.bls(lc.time(), med, 1, 20, 200);
            blsSeconds += (System.nanoTime() - t0) / 1e9;
            double[] gp = TransitSearch.detrendGp(lc.time(), lc.flux(), 0.7, 0.4);
            TransitSearch.Result b = TransitSearch.bls(lc.time(), gp, 1, 20, 200);
            double f = TransitSearch.fftPeriod(lc.time(), med, 1.0 / 48, 1, 20);
            hits[0][bin] += Math.abs(a.period() / period - 1) < 0.01 ? 1 : 0;
            hits[1][bin] += Math.abs(b.period() / period - 1) < 0.01 ? 1 : 0;
            hits[2][bin] += Math.abs(f / period - 1) < 0.01 ? 1 : 0;
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] names = {"BLS + moving median", "BLS + GP", "FFT periodogram"};
        for (int m = 0; m < 3; m++) {
            double[] row = new double[bins];
            for (int b = 0; b < bins; b++) {
                row[b] = totals[b] == 0 ? Double.NaN : hits[m][b] / totals[b];
            }
            out.put("recovered period within 1% {SNR 4-7, 7-10, 10-20}: " + names[m], row);
        }
        out.put("curves per SNR bin, BLS + median s per curve", new double[] {totals[0], totals[1], totals[2],
            blsSeconds / planets});
        // vetting
        int total = vetTrain + vetTest;
        double[][][] views = new double[total][][];
        float[][][] raw = new float[total][1][];
        int[] label = new int[total];
        String[] kinds = {"planet", "grazing EB", "planet", "secondary EB"};
        for (int i = 0; i < total; i++) {
            String kind = kinds[i % 4];
            double period = Math.exp(Math.log(1) + (Math.log(20) - Math.log(1)) * r.nextDouble());
            double depth = 0.0007 + 0.0023 * r.nextDouble();
            ScienceSims.LightCurve lc = ScienceSims.lightCurve(days, kind, period, depth, noise, 0.005,
                    seed * 97 + i);
            label[i] = lc.label();
            double[] med = TransitSearch.detrendMedian(lc.time(), lc.flux(), 1.0);
            views[i] = TransitSearch.views(lc.time(), med, lc.period(), lc.epoch(), lc.duration(), 201, 61);
            double[] grid = uniformGrid(lc.time(), med, 1.0 / 24, 2160);
            raw[i][0] = new float[grid.length];
            for (int k = 0; k < grid.length; k++) {
                raw[i][0][k] = (float) (grid[k] / noise);
            }
        }
        double[][][] trV = Arrays.copyOfRange(views, 0, vetTrain);
        double[][][] teV = Arrays.copyOfRange(views, vetTrain, total);
        int[] trY = Arrays.copyOfRange(label, 0, vetTrain);
        int[] teY = Arrays.copyOfRange(label, vetTrain, total);
        TransitVetter vetter = new TransitVetter(seed).train(trV, trY, vetSteps, 32, 3e-3, seed);
        double[] cnn = vetter.scores(teV);
        RandomConvolutionFeatures rocket = RandomConvolutionFeatures.rocket(60, 2160, 1, seed);
        double[] rs = ridgeScores(rocket.transform(Arrays.copyOfRange(raw, 0, vetTrain)), trY,
                rocket.transform(Arrays.copyOfRange(raw, vetTrain, total)), 1.0);
        out.put("vetting {AUROC, precision at 95% recall}: two-view CNN", new double[] {auroc(cnn, teY),
            precisionAtRecall(cnn, teY, 0.95)});
        out.put("vetting {AUROC, precision at 95% recall}: ROCKET + ridge on the light curve",
                new double[] {auroc(rs, teY), precisionAtRecall(rs, teY, 0.95)});
        return out;
    }

    static double auroc(double[] score, int[] y) {
        double s = 0;
        long pairs = 0;
        for (int i = 0; i < score.length; i++) {
            if (y[i] != 1) {
                continue;
            }
            for (int j = 0; j < score.length; j++) {
                if (y[j] == 0) {
                    s += score[i] > score[j] ? 1 : score[i] == score[j] ? 0.5 : 0;
                    pairs++;
                }
            }
        }
        return s / Math.max(1, pairs);
    }

    private static double precisionAtRecall(double[] score, int[] y, double recall) {
        Integer[] order = new Integer[score.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(score[b], score[a]));
        int pos = 0;
        for (int v : y) {
            pos += v;
        }
        int tp = 0;
        for (int k = 0; k < order.length; k++) {
            tp += y[order[k]];
            if (tp >= recall * pos) {
                return tp / (double) (k + 1);
            }
        }
        return pos / (double) y.length;
    }

    private static float[][] toFloat(double[][] x) {
        float[][] f = new float[x.length][x[0].length];
        for (int c = 0; c < x.length; c++) {
            for (int i = 0; i < x[0].length; i++) {
                f[c][i] = (float) x[c][i];
            }
        }
        return f;
    }

    /** P is the first PELT change of the smoothed log energy, S the next upward change. */
    private static PhasePicking.Picks pelt(double[][] x, double rate) {
        int w = Math.max(1, (int) (0.2 * rate));
        int dec = w;
        int n = x[0].length;
        double[] e = new double[n / dec];
        for (int k = 0; k < e.length; k++) {
            double s = 0;
            int c = k * dec;
            for (int i = Math.max(0, c - w + 1); i <= c; i++) {
                s += x[0][i] * x[0][i] + x[1][i] * x[1][i] + x[2][i] * x[2][i];
            }
            e[k] = Math.log(s / w + 1e-9);
        }
        // penalty: 4 log n times the noise variance of the log energy (from the first two seconds; the windows do not overlap)
        int head = Math.min(e.length, 10);
        double hm = mean(e, 0, head);
        double hv = 0;
        for (int k = 0; k < head; k++) {
            hv += (e[k] - hm) * (e[k] - hm) / head;
        }
        int[] cps = new Pelt(Pelt.Cost.MEAN, 4 * Math.log(e.length) * Math.max(hv, 1e-3), 4).segment(e);
        int p = -1;
        int s = -1;
        double prevMean = mean(e, 0, cps.length > 0 ? cps[0] : e.length);
        for (int k = 0; k < cps.length; k++) {
            int end = k + 1 < cps.length ? cps[k + 1] : e.length;
            double m = mean(e, cps[k], end);
            if (m > prevMean + 0.5) {
                if (p < 0) {
                    p = cps[k] * dec;
                } else if (s < 0) {
                    s = cps[k] * dec;
                }
            }
            prevMean = m;
        }
        return new PhasePicking.Picks(p, s);
    }

    private static double mean(double[] v, int from, int to) {
        double s = 0;
        for (int i = from; i < to; i++) {
            s += v[i];
        }
        return s / Math.max(1, to - from);
    }

    private static final class PickScore {
        private int events;
        private int pHit;
        private int pPicks;
        private int sHit;
        private int sPicks;
        private int noise;
        private int noisePicks;
        private int lowEvents;
        private int lowSHit;
        private double pRes2;
        private double sRes2;

        void add(ScienceSims.SeismicTrace t, PhasePicking.Picks p, int tol, double rate) {
            if (t.pIndex() < 0) {
                noise++;
                noisePicks += p.p() >= 0 || p.s() >= 0 ? 1 : 0;
                pPicks += p.p() >= 0 ? 1 : 0;
                sPicks += p.s() >= 0 ? 1 : 0;
                return;
            }
            events++;
            boolean low = t.snrDb() <= 0;
            lowEvents += low ? 1 : 0;
            if (p.p() >= 0) {
                pPicks++;
                int d = p.p() - t.pIndex();
                if (Math.abs(d) <= tol) {
                    pHit++;
                    pRes2 += d * d / (rate * rate);
                }
            }
            if (p.s() >= 0) {
                sPicks++;
                int d = p.s() - t.sIndex();
                if (Math.abs(d) <= tol) {
                    sHit++;
                    sRes2 += d * d / (rate * rate);
                    lowSHit += low ? 1 : 0;
                }
            }
        }

        double[] result() {
            return new double[] {pHit / (double) events, pHit / Math.max(1.0, pPicks), Math.sqrt(pRes2 / Math.max(1, pHit)),
                sHit / (double) events, sHit / Math.max(1.0, sPicks), Math.sqrt(sRes2 / Math.max(1, sHit)),
                noisePicks / (double) Math.max(1, noise), lowSHit / (double) Math.max(1, lowEvents)};
        }
    }
}
