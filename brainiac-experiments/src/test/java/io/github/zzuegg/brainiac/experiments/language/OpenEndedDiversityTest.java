package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Wave 5, 3.17: seed-conditioning against temperature sampling on "invent a walk", plus hivemind overlap. */
class OpenEndedDiversityTest {

    @Test
    void seedConditioningInventsMoreNovelWalks() {
        OpenEndedDiversity.Result r = OpenEndedDiversity.run(100, 300, 1);
        StringBuilder sb = new StringBuilder();
        for (OpenEndedDiversity.Outcome o : r.outcomes()) {
            sb.append(String.format("%s: valid %.2f, unique valid %.2f, novel %.2f; ", o.method(), o.valid(),
                    o.uniqueValid(), o.novelUniqueValid()));
        }
        System.out.printf("%sbaseline repeats %.2f of its valid samples, two baselines overlap %.2f (Jaccard)%n", sb,
                r.intraRepeat(), r.interOverlap());
        OpenEndedDiversity.Outcome t1 = r.outcomes().get(1);
        OpenEndedDiversity.Outcome seeded = r.outcomes().get(3);
        assertTrue(seeded.uniqueValid() > 0.2, "greedy decoding from random seeds is diverse");
        assertTrue(seeded.novelUniqueValid() > t1.novelUniqueValid(), "and more often novel than temperature 1");
        assertTrue(r.interOverlap() > 0.5, "two models trained alike say the same things");
    }
}
