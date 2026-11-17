package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experiments.language.UlmfitTransfer.Method;
import io.github.zzuegg.brainiac.experiments.language.UlmfitTransfer.Result;

import org.junit.jupiter.api.Test;

class UlmfitTransferTest {

    @Test
    void ulmfitTradesTargetFitForRetention() {
        HookedDecoder base = CountingTasks.pretrained(150, 3);
        StringBuilder line = new StringBuilder("Stride 1,2 -> 5 fine-tuning, target loss / source forgetting:");
        Result[] r = new Result[Method.values().length];
        for (Method m : Method.values()) {
            r[m.ordinal()] = UlmfitTransfer.run(base, m, 60, 9);
            line.append(String.format(" %s %.3f/%+.3f;", m, r[m.ordinal()].targetLoss(), r[m.ordinal()].forgetting()));
        }
        line.append(String.format(" (source loss before %.3f)", r[0].sourceBefore()));
        System.out.println(line);
        Result full = r[Method.FULL.ordinal()];
        Result ulmfit = r[Method.ULMFIT.ordinal()];
        assertTrue(ulmfit.targetLoss() < 0.75f * CountingTasks.evaluate(base, 12, 109, PeftShootout.NEW_STRIDE),
                "ULMFiT adapts to the new stride");
        assertTrue(ulmfit.forgetting() < full.forgetting(), "and forgets less of the source strides than full fine-tuning");
    }
}
