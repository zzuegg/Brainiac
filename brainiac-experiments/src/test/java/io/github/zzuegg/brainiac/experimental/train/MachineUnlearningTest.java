package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class MachineUnlearningTest {

    static Dataset dataset(List<float[]> x, List<Integer> y) {
        float[][] t = new float[y.size()][3];
        for (int i = 0; i < t.length; i++) {
            t[i][y.get(i)] = 1;
        }
        return Dataset.of(x.toArray(new float[0][]), t);
    }

    @ParameterizedTest
    @EnumSource(MachineUnlearning.Method.class)
    void forgettingAClassKeepsTheOthers(MachineUnlearning.Method method) {
        int[] y = new int[300];
        float[][] x = EnsembleUncertaintyTest.blobs(300, y, 1);
        List<float[]> fx = new ArrayList<>();
        List<Integer> fy = new ArrayList<>();
        List<float[]> rx = new ArrayList<>();
        List<Integer> ry = new ArrayList<>();
        for (int i = 0; i < x.length; i++) {
            (y[i] == 2 ? fx : rx).add(x[i]);
            (y[i] == 2 ? fy : ry).add(y[i]);
        }
        Dataset forget = dataset(fx, fy);
        Dataset retain = dataset(rx, ry);
        Model m = Model.sequential().input(2).dense(16, Activation.TANH).dense(3, Activation.SOFTMAX).seed(2).build();
        Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.02f)).batchSize(32).epochs(30)
                .fit(Dataset.of(x, EnsembleUncertaintyTest.oneHot(y)));
        assertTrue(MachineUnlearning.accuracy(m, forget) > 0.9);
        List<MachineUnlearning.Step> steps = MachineUnlearning.of(m, Loss.CROSS_ENTROPY).method(method)
                .optimizer(() -> Optimizer.adam(0.01f)).steps(300).batchSize(32).forgetLossCeiling(4).seed(3)
                .run(forget, retain);
        assertTrue(!steps.isEmpty());
        double f = MachineUnlearning.accuracy(m, forget);
        double r = MachineUnlearning.accuracy(m, retain);
        assertTrue(f < 0.1, method + " forget accuracy " + f);
        assertTrue(r > 0.9, method + " retain accuracy " + r);
    }

    @Test
    void membershipInferenceSeesMemorisedSamples() {
        SplittableRandom rnd = new SplittableRandom(5);
        List<float[]> mx = new ArrayList<>();
        List<Integer> my = new ArrayList<>();
        List<float[]> nx = new ArrayList<>();
        List<Integer> ny = new ArrayList<>();
        for (int i = 0; i < 80; i++) {
            float[] p = {(float) rnd.nextGaussian(), (float) rnd.nextGaussian(), (float) rnd.nextGaussian()};
            (i < 40 ? mx : nx).add(p);
            (i < 40 ? my : ny).add(rnd.nextInt(3));   // pure noise labels: only memorisation fits them
        }
        Dataset members = dataset(mx, my);
        Model m = Model.sequential().input(3).dense(64, Activation.TANH).dense(3, Activation.SOFTMAX).seed(4).build();
        Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.02f)).batchSize(40).epochs(400)
                .fit(members);
        double auc = MachineUnlearning.membershipAuc(m, members, dataset(nx, ny));
        assertTrue(auc > 0.85, "auc " + auc);
        assertEquals(0.5, MachineUnlearning.auc(new double[] {1, 2}, new double[] {1, 2}), 1e-12);
        assertEquals(1, MachineUnlearning.auc(new double[] {0.1}, new double[] {1, 2}), 1e-12);
    }
}
