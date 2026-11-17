package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.techniques.SineMetaRegression.Learner;
import io.github.zzuegg.brainiac.experiments.techniques.SineMetaRegression.Result;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class SineMetaRegressionTest {

    @Test
    void neuralProcessesRegressSinesWithCalibratedUncertainty() {
        Map<Learner, Result> r = new EnumMap<>(Learner.class);
        r.put(Learner.GP, SineMetaRegression.run(Learner.GP, 0, 1));
        r.put(Learner.MAML, SineMetaRegression.run(Learner.MAML, 1000, 1));
        r.put(Learner.CNP, SineMetaRegression.run(Learner.CNP, 900, 1));
        r.put(Learner.ANP, SineMetaRegression.run(Learner.ANP, 900, 1));
        System.out.printf("sine 10-shot MSE / NLL / 95%% coverage: CNP %.3f/%.2f/%.2f, "
                + "ANP %.3f/%.2f/%.2f | MAML %.3f/%.2f/%.2f, GP %.3f/%.2f/%.2f%n",
                r.get(Learner.CNP).mse(), r.get(Learner.CNP).nll(), r.get(Learner.CNP).coverage(),
                r.get(Learner.ANP).mse(), r.get(Learner.ANP).nll(), r.get(Learner.ANP).coverage(),
                r.get(Learner.MAML).mse(), r.get(Learner.MAML).nll(), r.get(Learner.MAML).coverage(),
                r.get(Learner.GP).mse(), r.get(Learner.GP).nll(), r.get(Learner.GP).coverage());
        double scratch = 4.1;   // MetaLearningExperiment's untrained baseline on the same waves
        assertTrue(r.get(Learner.ANP).mse() < 0.7 * scratch, r.toString());
        assertTrue(r.get(Learner.ANP).nll() < r.get(Learner.MAML).nll(), "an uncertainty head beats MAML on NLL");
        assertTrue(r.get(Learner.GP).mse() < r.get(Learner.CNP).mse(), "the GP is the gold standard");
    }
}
