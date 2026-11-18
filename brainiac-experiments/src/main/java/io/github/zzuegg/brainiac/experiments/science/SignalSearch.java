package io.github.zzuegg.brainiac.experiments.science;

import io.github.zzuegg.brainiac.experimental.anomaly.Pelt;
import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.experimental.science.PhasePicking;
import io.github.zzuegg.brainiac.experimental.timeseries.RandomConvolutionFeatures;
import io.github.zzuegg.brainiac.experimental.timeseries.RidgeClassifier;

import java.util.ArrayList;
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
     *     residual sd (s), false picks on noise, S recall at SNR <= 0 dB}
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
