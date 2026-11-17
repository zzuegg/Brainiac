package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class KvEfficientAttentionTest {

    @Test
    void latentAttentionMatchesMhaWithAFractionOfTheCache() {
        List<KvEfficientAttention.Row> rows = KvEfficientAttention.acrossHeads(32, 250, 1);
        System.out.print(KvEfficientAttention.table(rows));
        KvEfficientAttention.Row mha = rows.get(0);
        KvEfficientAttention.Row gqa = rows.get(1);
        KvEfficientAttention.Row mla = rows.get(2);
        System.out.printf("MLA val loss %.4f with %d cached/token vs MHA %.4f with %d, GQA %.4f with %d%n",
                mla.validationLoss(), mla.cachedPerToken(), mha.validationLoss(), mha.cachedPerToken(),
                gqa.validationLoss(), gqa.cachedPerToken());
        assertTrue(mla.cachedPerToken() * 3 < mha.cachedPerToken());
        assertTrue(mla.validationLoss() < mha.validationLoss() + 0.4, "MLA close to MHA");
        assertTrue(mla.validationLoss() < Math.log(32) * 0.85, "MLA learns");
    }
}
