package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class CrossLayerKvTest {

    @Test
    void sharedCachesStayCloseToTheUnsharedStack() {
        List<KvEfficientAttention.Row> rows = KvEfficientAttention.acrossLayers(32, 120, 2);
        System.out.print(KvEfficientAttention.table(rows));
        KvEfficientAttention.Row unshared = rows.get(0);
        KvEfficientAttention.Row cla = rows.get(1);
        KvEfficientAttention.Row yoco = rows.get(2);
        System.out.printf("CLA val loss %.4f (cache %d), YOCO %.4f (cache %d) vs unshared %.4f (cache %d)%n",
                cla.validationLoss(), cla.cachedPerToken(), yoco.validationLoss(), yoco.cachedPerToken(),
                unshared.validationLoss(), unshared.cachedPerToken());
        assertTrue(cla.cachedPerToken() * 2 == unshared.cachedPerToken());
        assertTrue(yoco.cachedPerToken() * 4 == unshared.cachedPerToken());
        assertTrue(cla.validationLoss() < unshared.validationLoss() + 0.3);
        assertTrue(yoco.validationLoss() < unshared.validationLoss() + 0.3);
    }
}
