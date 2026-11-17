package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.EnumMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class NoisyPreferenceTest {

    private static double ls(double x) {
        return -Math.log1p(Math.exp(-x));
    }

    @Test
    void dpoFamilyHasItsClosedForms() {
        Var pc = Var.param(Tensor.of(1f));
        Var pr = Var.param(Tensor.of(0f));
        Var zero = Var.of(Tensor.of(0f));
        Var refLow = Var.of(Tensor.of(2f));   // chosen likelihood fell by 1 versus the reference
        double h = 1;
        assertEquals(-ls(0.5 * h), PreferenceLosses.dpo(pc, pr, zero, zero, 0.5f).value().item(), 1e-6);
        assertEquals(-(0.8 * ls(0.5) + 0.2 * ls(-0.5)),
                PreferenceLosses.conservativeDpo(pc, pr, zero, zero, 0.5f, 0.2f).value().item(), 1e-6);
        assertEquals((0.8 * -ls(0.5) - 0.2 * -ls(-0.5)) / 0.6,
                PreferenceLosses.robustDpo(pc, pr, zero, zero, 0.5f, 0.2f).value().item(), 1e-6);
        // DPOP: margin (1 - 2) - 0 = -1, minus lambda * (2 - 1) = 3 -> -4
        assertEquals(-ls(0.5 * -4), PreferenceLosses.dpoPositive(pc, pr, refLow, zero, 0.5f, 3).value().item(), 1e-5);
        assertEquals(0.5f * (1 + 0.6f * (1 - 0.5f)), PreferenceLosses.adaptiveBeta(new float[] {1}, 0.5f, 0.6f, 0.5f),
                1e-6);
    }

    @Test
    void dpoVariantsUnderTwentyPercentFlippedLabels() {
        HookedDecoder base = PeftShootoutTest.base();
        Map<PreferenceTuning.DpoVariant, PreferenceTuning.NoisyResult> res =
                new EnumMap<>(PreferenceTuning.DpoVariant.class);
        for (PreferenceTuning.DpoVariant v : PreferenceTuning.DpoVariant.values()) {
            PreferenceTuning.NoisyResult r = PreferenceTuning.noisy(base, v, 0.2, 80, 4);
            res.put(v, r);
            System.out.printf("%-8s 20%% flipped: abs win rate %.3f, margin %.2f -> %.2f, chosen log-lik drift %+.2f%n",
                    v, r.winRate(), r.marginBefore(), r.marginAfter(), r.chosenDrift());
        }
        // The pretrained model strongly prefers the rejected style (margin about -34). Under
        // 20% noise, DPO stalls at the finite margin where flipped and clean gradients cancel
        // (beta h = ln 4); rDPO's de-biased loss keeps moving as on clean labels.
        PreferenceTuning.NoisyResult dpo = res.get(PreferenceTuning.DpoVariant.DPO);
        PreferenceTuning.NoisyResult rdpo = res.get(PreferenceTuning.DpoVariant.RDPO);
        PreferenceTuning.NoisyResult dpop = res.get(PreferenceTuning.DpoVariant.DPOP);
        for (PreferenceTuning.NoisyResult r : res.values()) {
            assertTrue(r.marginAfter() > r.marginBefore(), r.toString());
        }
        assertTrue(rdpo.marginAfter() > dpo.marginAfter() + 10 && rdpo.winRate() >= dpo.winRate(), rdpo.toString());
        assertTrue(dpop.chosenDrift() > dpo.chosenDrift(), "DPOP keeps the chosen likelihood up " + dpop);
    }
}
