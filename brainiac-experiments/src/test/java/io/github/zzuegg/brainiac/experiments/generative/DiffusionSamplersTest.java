package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class DiffusionSamplersTest {

    @Test
    void edmDpmSolverAndScoreSdeSamplersAgainstDdimAndFlowMatching() {
        DiffusionSamplers.Models models = DiffusionSamplers.train(1200, 1);
        StringBuilder line = new StringBuilder("ring distance (blobs) by NFE:");
        Map<String, double[]> q = new HashMap<>();
        for (int nfe : new int[] {6, 10, 20}) {
            line.append(" | NFE ").append(nfe).append(':');
            for (Map.Entry<String, float[][]> e : DiffusionSamplers.sample(models, nfe, 7).entrySet()) {
                double[] s = DiffusionSamplers.quality(e.getValue());
                q.put(e.getKey() + " " + nfe, s);
                line.append(String.format(" %s %.3f (%.0f)", e.getKey(), s[0], s[1]));
            }
        }
        double[] ddpm = DiffusionSamplers.quality(DiffusionSamplers.sampleDdpm(models, 7));
        line.append(String.format(" | DDPM 100 steps %.3f (%.0f)", ddpm[0], ddpm[1]));
        System.out.println(line);
        assertTrue(q.get("EDM DPM-Solver++ 10")[0] < q.get("EDM Heun 10")[0] * 0.6,
                "DPM-Solver++ beats Heun on the same EDM model at 10 NFE");
        assertTrue(q.get("EDM DPM-Solver++ 10")[0] < 0.3, "DPM-Solver++ lands on the blobs in 10 NFE");
        assertTrue(q.get("EDM Heun 20")[0] < 0.3, "EDM Heun lands on the blobs in 20 NFE");
        assertTrue(q.get("SDE PC 20")[1] >= 5 && q.get("SDE ODE 20")[1] >= 5, "score-SDE samplers cover the ring");
        assertTrue(q.get("SDE ODE 20")[0] < 0.4 && q.get("SDE PC 20")[0] < 0.6, "score-SDE samplers converge");
    }

    @Test
    void scoreSdeHasAnExactLikelihood() {
        double[] nll = DiffusionSamplers.sdeLikelihood(DensityBenchmark.Target.MOONS, 1000, 1);
        System.out.printf("moons held-out NLL nats via the VP probability-flow ODE: %.3f (%.0f NFE); "
                + "a standard normal scores about 2.1%n", nll[0], nll[1]);
        assertTrue(nll[0] < 2.0);
    }
}
