package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.LogitStabilizers;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Map;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class DecodingShowdownTest {

    @Test
    void everySpeedupIsLosslessAndMostSaveFullPasses() {
        Map<String, DecodingShowdown.Outcome> out = DecodingShowdown.run(PeftShootoutTest.base(), new int[] {2, 4, 6},
                16, 50, 1);
        int[] greedy = out.get("greedy").tokens();
        int faster = 0;
        for (Map.Entry<String, DecodingShowdown.Outcome> e : out.entrySet()) {
            assertArrayEquals(greedy, e.getValue().tokens(), e.getKey());
            faster += e.getValue().passes() < 16 ? 1 : 0;
        }
        assertTrue(faster >= 3, out.toString());
    }

    @Test
    void zLossPullsTheNormalizerToZeroAndSoftCapBounds() {
        Var logits = Var.param(Tensor.normal(Shape.of(4, 6), 3, 1, new SplittableRandom(1)));
        float first = LogitStabilizers.zLoss(logits, 1).value().item();
        for (int i = 0; i < 50; i++) {
            Var z = LogitStabilizers.zLoss(logits, 1);
            z.backward();
            logits.assign(logits.value().sub(logits.grad().mul(1f)));
            logits.zeroGrad();
        }
        assertTrue(LogitStabilizers.zLoss(logits, 1).value().item() < 0.1f * first);
        float[] capped = LogitStabilizers.softCap(Var.of(Tensor.of(100f, -100f, 1f)), 30).value().toArray();
        assertTrue(capped[0] < 30 && capped[1] > -30 && Math.abs(capped[2] - 1) < 0.01);
    }
}
