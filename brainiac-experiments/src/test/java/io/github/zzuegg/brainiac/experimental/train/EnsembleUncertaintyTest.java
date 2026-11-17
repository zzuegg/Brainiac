package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class EnsembleUncertaintyTest {

    static final float[][] CENTRES = {{2, 0}, {-1, 1.7f}, {-1, -1.7f}};

    static float[][] blobs(int n, int[] y, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][];
        for (int i = 0; i < n; i++) {
            y[i] = i % 3;
            x[i] = new float[] {CENTRES[y[i]][0] + (float) r.nextGaussian() * 0.6f,
                CENTRES[y[i]][1] + (float) r.nextGaussian() * 0.6f};
        }
        return x;
    }

    static float[][] far(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][];
        for (int i = 0; i < n; i++) {
            double a = r.nextDouble() * 2 * Math.PI;
            x[i] = new float[] {(float) (12 * Math.cos(a)), (float) (12 * Math.sin(a))};
        }
        return x;
    }

    static float[][] oneHot(int[] y) {
        float[][] t = new float[y.length][3];
        for (int i = 0; i < y.length; i++) {
            t[i][y[i]] = 1;
        }
        return t;
    }

    static void fit(Model m, float[][] x, int[] y, int epochs) {
        Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.02f)).batchSize(32).epochs(epochs)
                .fit(Dataset.of(x, oneHot(y)));
    }

    static double meanMi(PopulationEnsemble e, float[][] x) {
        double[] u = e.predict(x).uncertainties(PopulationEnsemble.Uncertainty.MUTUAL_INFORMATION);
        double s = 0;
        for (double v : u) {
            s += v / u.length;
        }
        return s;
    }

    @Test
    void mcDropoutDisagreesMoreAwayFromTheData() {
        int[] y = new int[300];
        float[][] x = blobs(300, y, 1);
        Model m = Model.sequential().input(2).dense(32, Activation.RELU).dropout(0.3f).dense(32, Activation.RELU)
                .dropout(0.3f).dense(3, Activation.SOFTMAX).seed(2).build();
        fit(m, x, y, 30);
        PopulationEnsemble mc = PopulationEnsemble.mcDropout(m, 20);
        assertEquals(20, mc.size());
        int[] ty = new int[150];
        float[][] test = blobs(150, ty, 3);
        PopulationEnsemble.Prediction p = mc.predict(test);
        assertTrue(Calibration.accuracy(p.mean(), ty) > 0.9);
        // members really are different masks
        assertTrue(Math.abs(p.members()[0][0][0] - p.members()[1][0][0]) > 1e-6);
        double in = meanMi(mc, test);
        double out = meanMi(mc, far(100, 4));
        assertTrue(out > 2 * in, "far " + out + " in " + in);
    }

    @Test
    void deepEnsembleBeatsItsMembersOnNllAndFlagsFarInputs() {
        int[] y = new int[200];
        float[][] x = blobs(200, y, 5);
        PopulationEnsemble e = PopulationEnsemble.deepEnsemble(i -> Model.sequential().input(2)
                .dense(16, Activation.TANH).dense(3, Activation.SOFTMAX).seed(10 + i).build(), 5, m -> fit(m, x, y, 25));
        int[] ty = new int[150];
        float[][] test = blobs(150, ty, 6);
        PopulationEnsemble.Prediction p = e.predict(test);
        double ensemble = Calibration.negativeLogLikelihood(p.mean(), ty);
        double members = 0;
        for (double[][] member : p.members()) {
            members += Calibration.negativeLogLikelihood(member, ty) / p.members().length;
        }
        assertTrue(ensemble <= members + 1e-9, ensemble + " vs " + members);
        assertTrue(meanMi(e, far(100, 7)) > meanMi(e, test));
    }
}
