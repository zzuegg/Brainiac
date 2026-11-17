package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.train.MachineUnlearning;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Machine unlearning of a whole class. A classifier of four clusters is trained, then told
 * to forget one cluster's training samples with retain-regularised gradient ascent. The
 * outcome compares accuracy before and after unlearning on the forgotten class (both its
 * training samples and unseen ones) and on the other classes, so you can see the class
 * vanish while the rest stays.
 *
 * <p>Golatkar et al. 2020, "Eternal Sunshine of the Spotless Net: Selective Forgetting in
 * Deep Networks", <a href="https://arxiv.org/abs/1911.04933">arXiv:1911.04933</a>; Kurmanji
 * et al. 2023, "Towards Unbounded Machine Unlearning", <a href="https://arxiv.org/abs/2302.09880">arXiv:2302.09880</a>.
 */
public final class ClassForgetting {

    /**
     * The outcome.
     *
     * @param forgetBefore accuracy on the forget set before unlearning
     * @param forgetAfter accuracy on the forget set after
     * @param unseenForgottenClassAfter accuracy on unseen samples of the forgotten class after
     * @param retainBefore accuracy on the other classes' test samples before
     * @param retainAfter the same after
     * @param steps unlearning steps taken
     */
    public record Outcome(double forgetBefore, double forgetAfter, double unseenForgottenClassAfter,
            double retainBefore, double retainAfter, int steps) {
    }

    private static final float[][] CENTRES = {{-1.5f, -1.5f}, {1.5f, -1.5f}, {1.5f, 1.5f}, {-1.5f, 1.5f}};

    private ClassForgetting() {
    }

    private static Dataset data(int n, long seed, boolean forgotten, boolean others) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> x = new ArrayList<>();
        List<float[]> y = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int c = i % 4;
            float[] p = {CENTRES[c][0] + (float) r.nextGaussian() * 0.6f,
                CENTRES[c][1] + (float) r.nextGaussian() * 0.6f};
            if (c == 3 ? forgotten : others) {
                float[] t = new float[4];
                t[c] = 1;
                x.add(p);
                y.add(t);
            }
        }
        return Dataset.of(x.toArray(new float[0][]), y.toArray(new float[0][]));
    }

    /**
     * Runs the experiment.
     *
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(long seed) {
        Dataset all = data(400, seed, true, true);
        Dataset forget = data(400, seed, true, false);
        Dataset retain = data(400, seed, false, true);
        Dataset unseen = data(400, seed + 1, true, false);
        Dataset test = data(400, seed + 1, false, true);
        Model m = Model.sequential().input(2).dense(16, Activation.TANH).dense(4, Activation.SOFTMAX).seed(seed)
                .build();
        Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.02f)).batchSize(32).epochs(30).seed(seed)
                .fit(all);
        double fb = MachineUnlearning.accuracy(m, forget);
        double rb = MachineUnlearning.accuracy(m, test);
        int steps = MachineUnlearning.of(m, Loss.CROSS_ENTROPY).optimizer(() -> Optimizer.adam(0.01f)).steps(300)
                .forgetLossCeiling(4).seed(seed).run(forget, retain).size();
        return new Outcome(fb, MachineUnlearning.accuracy(m, forget), MachineUnlearning.accuracy(m, unseen), rb,
                MachineUnlearning.accuracy(m, test), steps);
    }
}
