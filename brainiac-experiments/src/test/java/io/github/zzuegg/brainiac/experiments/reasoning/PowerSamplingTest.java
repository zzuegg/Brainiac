package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Wave 6, 6.10: power-distribution MCMC sampling against low temperature and GRPO. */
class PowerSamplingTest {

    @Test
    void powerSamplingSharpensLikeRlWithoutTraining() {
        PowerSampling.Result r = PowerSampling.run(ArithmeticTask.addition(6), 4, 6, 120, 16, 1);
        StringBuilder sb = new StringBuilder();
        int last = r.ks().length - 1;
        for (PowerSampling.Arm a : r.arms()) {
            sb.append(String.format(" %s pass@1 %.2f pass@%d %.2f (%.1f tok/answer);", a.name(), a.passAtK()[0],
                    r.ks()[last], a.passAtK()[last], a.tokensPerAnswer()));
        }
        System.out.printf("addition(6):%s MH acceptance %.2f%n", sb, r.acceptance());
        PowerSampling.Result c = PowerSampling.chain(new ChainTask(4), 150, 4, 6, 8, 1);
        StringBuilder cb = new StringBuilder();
        int cl = c.ks().length - 1;
        for (PowerSampling.Arm a : c.arms()) {
            cb.append(String.format(" %s pass@1 %.2f pass@%d %.2f (%.1f tok);", a.name(), a.passAtK()[0],
                    c.ks()[cl], a.passAtK()[cl], a.tokensPerAnswer()));
        }
        System.out.printf("chain(4):%s MH acceptance %.2f%n", cb, c.acceptance());
        double base = r.arm("base T=1").passAtK()[0];
        double power = r.arm("power sampling").passAtK()[0];
        assertTrue(power > base + 0.05, "power sampling beats the base at pass@1: " + power + " vs " + base);
        assertTrue(r.arm("power sampling").passAtK()[last] >= r.arm("GRPO T=1").passAtK()[last] - 0.05,
                "power sampling keeps pass@k coverage");
        assertTrue(c.arm("power sampling").passAtK()[0] > c.arm("base T=1").passAtK()[0] + 0.05,
                "power sampling sharpens the chain base");
    }
}
