package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class CompressedKvShootoutTest {

    @Test
    void compressedBlocksCutTheCacheAtModestLoss() {
        List<CompressedKvShootout.Row> rows = CompressedKvShootout.run(16, 150, 1, false);
        System.out.print(CompressedKvShootout.table(rows));
        CompressedKvShootout.Row mha = rows.get(0);
        CompressedKvShootout.Row gqa = rows.get(1);
        CompressedKvShootout.Row mla = rows.get(2);
        CompressedKvShootout.Row infini = rows.get(3);
        CompressedKvShootout.Row v4 = rows.get(4);
        System.out.printf("val loss / cache per token: CSA/HCA %.4f / %.1f vs MHA %.4f / %.0f, GQA %.4f / %.0f, "
                + "MLA %.4f / %.0f, compressive %.4f / 0 (+%d fixed)%n", v4.validationLoss(), v4.cachedPerToken(),
                mha.validationLoss(), mha.cachedPerToken(), gqa.validationLoss(), gqa.cachedPerToken(),
                mla.validationLoss(), mla.cachedPerToken(), infini.validationLoss(), infini.constantCache());
        assertTrue(v4.cachedPerToken() < mla.cachedPerToken());
        assertTrue(v4.cachedPerToken() * 10 < mha.cachedPerToken());
        assertTrue(v4.validationLoss() < Math.log(32));
        assertTrue(v4.validationLoss() < mha.validationLoss() + 0.5);
    }
}
