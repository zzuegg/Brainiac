package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticRl.FineTuneResult;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticRl.FineTuner;
import org.junit.jupiter.api.Test;

class EvolutionFineTuningTest {

    @Test
    void evolutionStrategiesAgainstGrpoAndMezo() {
        StringBuilder sb = new StringBuilder();
        double base = 0;
        double es = 0;
        for (FineTuner m : FineTuner.values()) {
            double[] acc = new double[3];
            double kl = 0;
            for (int seed = 1; seed <= 3; seed++) {
                FineTuneResult r = ArithmeticRl.fineTune(ArithmeticTask.addition(4), m, 8192, seed);
                acc[seed - 1] = r.after();
                kl += r.kl() / 3;
                base = r.before();
            }
            double mean = (acc[0] + acc[1] + acc[2]) / 3;
            double sd = Math.sqrt(((acc[0] - mean) * (acc[0] - mean) + (acc[1] - mean) * (acc[1] - mean)
                    + (acc[2] - mean) * (acc[2] - mean)) / 3);
            if (m == FineTuner.ES) {
                es = mean;
            }
            sb.append(String.format("%s %.3f +- %.3f (KL %.3f); ", m, mean, sd, kl));
        }
        System.out.printf("addition fine-tuning from a %.2f base, 8192 reward evaluations, 3 seeds: %s%n", base, sb);
        assertTrue(es >= base, "ES does not lose accuracy");
    }
}
