package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Wave 5, 3.15: pass@k up to k = 256 for the base, GRPO and a distilled model. */
class ReasoningCoverageTest {

    private static String curve(int[] ks, double[] v) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ks.length; i++) {
            sb.append(i == 0 ? "" : " ").append(ks[i]).append(':').append(String.format("%.2f", v[i]));
        }
        return sb.toString();
    }

    @Test
    void rlSharpensWhileDistillationWidens() {
        ReasoningCoverage.Result r = ReasoningCoverage.run(ArithmeticTask.addition(6), 200, 100, 256, 1);
        System.out.printf("pass@k base [%s], GRPO [%s], distilled [%s]; base overtakes GRPO at k=%d%n",
                curve(r.ks(), r.base()), curve(r.ks(), r.rl()), curve(r.ks(), r.distilled()), r.crossover());
        int last = r.ks().length - 1;
        assertTrue(r.rl()[0] > r.base()[0], "GRPO wins at pass@1");
        assertTrue(r.crossover() > 0, "the base catches up at some k");
        assertTrue(r.distilled()[last] > r.rl()[last], "distillation covers more at k=256");
    }
}
