package io.github.zzuegg.brainiac.experiments.serving;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Map;
import org.junit.jupiter.api.Test;

class SpeculativeShootoutTest {

    @Test
    void onPolicyTvdDistillationRaisesAcceptance() {
        Map<String, SpeculativeShootout.DraftRow> r = SpeculativeShootout.distillation(150, new long[] {1, 2, 3});
        r.forEach((k, v) -> System.out.printf("8.4.5 %-34s acceptance T=1 %.3f, greedy %.3f, tokens/pass %.2f%n", k,
                v.acceptanceT1(), v.acceptanceGreedy(), v.tokensPerPass()));
        assertTrue(r.get("DistillSpec on-policy TVD").acceptanceT1() > r.get("draft on data").acceptanceT1(),
                r.toString());
    }

    @Test
    void kSequentialSelectionIsLosslessAndGainsWithDrafts() {
        SpeculativeShootout.MultiDraftResult r = SpeculativeShootout.multiDraft(100_000, new long[] {1, 2, 3});
        System.out.printf("8.4.6 SpecTr K-SEQ tokens/pass k=1/2/4/8 %s; TV k=4 K-SEQ %.4f, naive %.4f,"
                + " plain sampling %.4f%n", Arrays.toString(Arrays.stream(r.tokensPerPass())
                .mapToObj(x -> String.format("%.2f", x)).toArray()), r.tvKSeq(), r.tvNaive(), r.tvSampling());
        assertTrue(r.tokensPerPass()[3] > r.tokensPerPass()[0], Arrays.toString(r.tokensPerPass()));
        assertTrue(r.tvKSeq() < 3 * r.tvSampling() + 0.005, r.toString());
        assertTrue(r.tvNaive() > 2 * r.tvKSeq(), r.toString());
    }
}
