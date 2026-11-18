package io.github.zzuegg.brainiac.experiments.science;

import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.experimental.science.Hbv;
import io.github.zzuegg.brainiac.experimental.science.HybridHydrology;
import io.github.zzuegg.brainiac.experimental.train.RidgeRegression;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Rainfall–runoff modelling under warming (wave 13, item 13.3.11): a synthetic snowy catchment
 * whose "truth" is HBV with an extra slow groundwater store and multiplicative observation
 * noise. Models are trained on six cold years and scored on three cold and three warm
 * (+2 degrees, identical precipitation) test years: HBV calibrated by CMA-ES and by Nelder–Mead,
 * an LSTM on forcings, δ-HBV with static and with forcing-dependent parameters (started from
 * the CMA-ES calibration), a ridge regression on lagged forcings, and persistence. Reported:
 * NSE and KGE on both periods, and the error of the simulated change in mean flow and in the
 * high-flow quantile Q95 against the observed change.
 */
public final class Hydrology {

    private static final double[] TRUTH = {0, 3.5, 0.9, 0.05, 0.1, 250, 0.7, 2.5, 0.2, 0.08, 0.02, 20, 1.5, 2.5};
    private static final int YEAR = 365;

    private Hydrology() {
    }

    private static double[] observe(ScienceSims.Weather w, double noise, long seed) {
        double[] q = Hbv.simulate(TRUTH, w, 0.3, 0.005);
        SplittableRandom r = new SplittableRandom(seed);
        for (int t = 0; t < q.length; t++) {
            q[t] *= Math.exp(noise * r.nextGaussian() - noise * noise / 2);
        }
        return q;
    }

    private static double[] lagged(ScienceSims.Weather w, int t) {
        double[] p = w.precipitation();
        double[] temp = w.temperature();
        double[] f = new double[18];
        int k = 0;
        f[k++] = 1;
        for (int l = 0; l < 7; l++) {
            f[k++] = p[Math.max(0, t - l)];
        }
        f[k++] = window(p, t, 7, 30);
        f[k++] = window(p, t, 30, 90);
        f[k++] = window(p, t, 90, 180);
        f[k++] = temp[t];
        f[k++] = window(temp, t, 0, 7);
        f[k++] = window(temp, t, 0, 30);
        f[k++] = window(w.potentialEvaporation(), t, 0, 30);
        // cold-season precipitation over the last 180 days: a snowpack proxy
        double snow = 0;
        for (int l = 0; l < 180; l++) {
            int s = Math.max(0, t - l);
            snow += temp[s] < 0 ? p[s] : 0;
        }
        f[k++] = snow / 180;
        f[k++] = Math.sin(2 * Math.PI * (t % YEAR) / YEAR);
        f[k] = Math.cos(2 * Math.PI * (t % YEAR) / YEAR);
        return f;
    }

    private static double window(double[] v, int t, int from, int to) {
        double s = 0;
        for (int l = from; l < to; l++) {
            s += v[Math.max(0, t - l)];
        }
        return s / (to - from);
    }

    private static double q95(double[] q, int from) {
        double[] s = Arrays.copyOfRange(q, from, q.length);
        Arrays.sort(s);
        return s[(int) (0.95 * (s.length - 1))];
    }

    private static double mean(double[] q, int from) {
        double s = 0;
        for (int t = from; t < q.length; t++) {
            s += q[t];
        }
        return s / (q.length - from);
    }

    /**
     * Runs the comparison.
     *
     * @param trainYears training years (the first is warm-up)
     * @param testYears test years per period (plus one warm-up year)
     * @param warming degrees added in the warm test period
     * @param cmaGenerations CMA-ES generations (Nelder–Mead gets ten times as many iterations)
     * @param hybridEpochs δ-HBV full-series Adam steps
     * @param lstmSteps LSTM mini-batch steps
     * @param seed the seed
     * @return model to {NSE cold, NSE warm, KGE cold, KGE warm, |error| of change in mean
     *     flow, |error| of change in Q95, seconds}
     */
    public static Map<String, double[]> run(int trainYears, int testYears, double warming, int cmaGenerations,
            int hybridEpochs, int lstmSteps, long seed) {
        ScienceSims.Weather train = ScienceSims.weather(trainYears * YEAR, 0, seed);
        ScienceSims.Weather cold = ScienceSims.weather((testYears + 1) * YEAR, 0, seed + 1);
        ScienceSims.Weather warm = ScienceSims.weather((testYears + 1) * YEAR, warming, seed + 1);
        double[] obsTrain = observe(train, 0.1, seed + 2);
        double[] obsCold = observe(cold, 0.1, seed + 3);
        double[] obsWarm = observe(warm, 0.1, seed + 3);
        Map<String, double[]> out = new LinkedHashMap<>();
        double dMean = mean(obsWarm, YEAR) - mean(obsCold, YEAR);
        double dQ95 = q95(obsWarm, YEAR) - q95(obsCold, YEAR);
        out.put("observed {mean cold, mean warm, Q95 cold, Q95 warm}",
                new double[] {mean(obsCold, YEAR), mean(obsWarm, YEAR), q95(obsCold, YEAR), q95(obsWarm, YEAR)});

        long t0 = System.nanoTime();
        double[] pCma = Hbv.calibrate(obsTrain, train, YEAR, Hbv.Calibrator.CMA_ES, cmaGenerations, seed);
        double sCma = (System.nanoTime() - t0) / 1e9;
        out.put("HBV CMA-ES", score(Hbv.simulate(pCma, cold, 0, 0), Hbv.simulate(pCma, warm, 0, 0), obsCold,
                obsWarm, dMean, dQ95, sCma));

        t0 = System.nanoTime();
        double[] pNm = Hbv.calibrate(obsTrain, train, YEAR, Hbv.Calibrator.NELDER_MEAD, 10 * cmaGenerations, seed);
        out.put("HBV Nelder-Mead", score(Hbv.simulate(pNm, cold, 0, 0), Hbv.simulate(pNm, warm, 0, 0), obsCold,
                obsWarm, dMean, dQ95, (System.nanoTime() - t0) / 1e9));

        double[] unit = new double[pCma.length];
        for (int i = 0; i < unit.length; i++) {
            unit[i] = (pCma[i] - Hbv.LOWER[i]) / (Hbv.UPPER[i] - Hbv.LOWER[i]);
        }
        for (HybridHydrology.Model m : new HybridHydrology.Model[] {HybridHydrology.Model.DELTA_HBV_STATIC,
            HybridHydrology.Model.DELTA_HBV_DYNAMIC, HybridHydrology.Model.LSTM}) {
            t0 = System.nanoTime();
            HybridHydrology h = new HybridHydrology(m);
            if (m == HybridHydrology.Model.LSTM) {
                h.train(train, obsTrain, YEAR, lstmSteps, 1e-2, null, seed);
            } else {
                h.train(train, obsTrain, YEAR, hybridEpochs, 0.05, unit, seed);
            }
            double sec = (System.nanoTime() - t0) / 1e9 + (m == HybridHydrology.Model.LSTM ? 0 : sCma);
            String name = switch (m) {
                case LSTM -> "LSTM";
                case DELTA_HBV_STATIC -> "delta-HBV static (from CMA-ES)";
                case DELTA_HBV_DYNAMIC -> "delta-HBV dynamic (from CMA-ES)";
            };
            out.put(name, score(h.predict(cold), h.predict(warm), obsCold, obsWarm, dMean, dQ95, sec));
        }

        t0 = System.nanoTime();
        double[][] x = new double[trainYears * YEAR - YEAR][];
        double[][] y = new double[x.length][1];
        for (int t = YEAR; t < trainYears * YEAR; t++) {
            x[t - YEAR] = lagged(train, t);
            y[t - YEAR][0] = obsTrain[t];
        }
        double[][] wRidge = RidgeRegression.fit(x, y, 1.0);
        double[] rc = new double[obsCold.length];
        double[] rw = new double[obsWarm.length];
        for (int t = 0; t < rc.length; t++) {
            rc[t] = Math.max(0, RidgeRegression.predict(wRidge, lagged(cold, t))[0]);
            rw[t] = Math.max(0, RidgeRegression.predict(wRidge, lagged(warm, t))[0]);
        }
        out.put("ridge on lagged forcings", score(rc, rw, obsCold, obsWarm, dMean, dQ95,
                (System.nanoTime() - t0) / 1e9));

        double[] pc = new double[obsCold.length];
        double[] pw = new double[obsWarm.length];
        for (int t = 1; t < pc.length; t++) {
            pc[t] = obsCold[t - 1];
            pw[t] = obsWarm[t - 1];
        }
        out.put("persistence (uses yesterday's observed flow)", score(pc, pw, obsCold, obsWarm, dMean, dQ95, 0));
        return out;
    }

    private static double[] score(double[] simCold, double[] simWarm, double[] obsCold, double[] obsWarm,
            double dMean, double dQ95, double seconds) {
        double sMean = mean(simWarm, YEAR) - mean(simCold, YEAR);
        double sQ95 = q95(simWarm, YEAR) - q95(simCold, YEAR);
        return new double[] {Hbv.nse(simCold, obsCold, YEAR), Hbv.nse(simWarm, obsWarm, YEAR),
            Hbv.kge(simCold, obsCold, YEAR), Hbv.kge(simWarm, obsWarm, YEAR), Math.abs(sMean - dMean),
            Math.abs(sQ95 - dQ95), seconds};
    }
}
