package io.github.zzuegg.brainiac.experimental.calibrate;

import io.github.zzuegg.brainiac.experimental.train.Calibration;

/**
 * Ensemble temperature scaling, the parametric half of Mix-n-Match (Zhang, Kailkhura &amp; Han,
 * ICML 2020, <a href="https://arxiv.org/abs/2003.07329">arXiv:2003.07329</a>):
 * {@code p = w1 softmax(z / T) + w2 softmax(z) + w3 / k}, with {@code T} from temperature
 * scaling and the weights on the simplex chosen to minimise validation NLL (by a grid of
 * step {@code 0.02}). It keeps temperature scaling's data efficiency (four parameters, the
 * identity and temperature scaling are both members) but can mix in the uncalibrated
 * prediction and a uniform floor, which helps where temperature scaling alone over- or
 * under-shoots.
 */
public final class EnsembleTemperature implements Recalibrator {

    private final double temperature;
    private final double[] weights;

    private EnsembleTemperature(double temperature, double[] weights) {
        this.temperature = temperature;
        this.weights = weights;
    }

    /**
     * Fits the temperature and the mixing weights.
     *
     * @param logits validation logits
     * @param labels validation labels
     * @return the map
     */
    public static EnsembleTemperature fit(double[][] logits, int[] labels) {
        double t = Calibration.fitTemperature(logits, labels);
        int n = logits.length;
        int k = logits[0].length;
        double[] pt = new double[n];
        double[] p1 = new double[n];
        for (int i = 0; i < n; i++) {
            pt[i] = Calibration.softmax(logits[i], t)[labels[i]];
            p1[i] = Calibration.softmax(logits[i], 1)[labels[i]];
        }
        double best = Double.POSITIVE_INFINITY;
        double[] w = {1, 0, 0};
        int steps = 50;
        for (int a = 0; a <= steps; a++) {
            for (int b = 0; a + b <= steps; b++) {
                double w1 = a / (double) steps;
                double w2 = b / (double) steps;
                double w3 = 1 - w1 - w2;
                double nll = 0;
                for (int i = 0; i < n; i++) {
                    nll -= Math.log(Math.max(1e-12, w1 * pt[i] + w2 * p1[i] + w3 / k));
                }
                if (nll < best - 1e-12) {
                    best = nll;
                    w = new double[] {w1, w2, w3};
                }
            }
        }
        return new EnsembleTemperature(t, w);
    }

    /** @return the temperature */
    public double temperature() {
        return temperature;
    }

    /** @return the weights of the scaled, the uncalibrated and the uniform prediction */
    public double[] weights() {
        return weights.clone();
    }

    @Override
    public double[][] apply(double[][] logits) {
        double[][] out = new double[logits.length][];
        for (int i = 0; i < logits.length; i++) {
            double[] a = Calibration.softmax(logits[i], temperature);
            double[] b = Calibration.softmax(logits[i], 1);
            int k = a.length;
            out[i] = new double[k];
            for (int c = 0; c < k; c++) {
                out[i][c] = weights[0] * a[c] + weights[1] * b[c] + weights[2] / k;
            }
        }
        return out;
    }
}
