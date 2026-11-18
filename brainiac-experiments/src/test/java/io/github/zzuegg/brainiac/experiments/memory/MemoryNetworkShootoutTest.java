package io.github.zzuegg.brainiac.experiments.memory;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.memory.MemoryNetworkShootout.Architecture;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class MemoryNetworkShootoutTest {

    static double[] run(Architecture a, int steps, StringBuilder line) {
        double[] acc = MemoryNetworkShootout.run(a, steps, 1);
        line.append(String.format(" %s %.2f/%.2f/%.2f;", a, acc[0], acc[1], acc[2]));
        return acc;
    }

    @Test
    void storiesAnswerTheirQuestions() {
        BabiLite.Batch b = BabiLite.generate(50, new SplittableRandom(1), 1, 2, 3);
        for (int i = 0; i < 50; i++) {
            assertTrue(b.answers()[i] >= 0 && b.answers()[i] < BabiLite.ANSWERS);
        }
    }

    @Test
    void hopsChainSupportingFacts() {
        StringBuilder line = new StringBuilder("bAbI-lite accuracy on 1/2/3 supporting facts:");
        double[] one = run(Architecture.MEMN2N_1_HOP, 400, line);
        double[] three = run(Architecture.MEMN2N_3_HOPS, 400, line);
        run(Architecture.MEMN2N_2_HOPS, 400, line);
        double[] kv = run(Architecture.KV_MEMNN, 400, line);
        System.out.println(line);
        assertTrue(three[0] > 0.9, "MemN2N answers single-fact questions");
        assertTrue(three[1] > one[1], "more hops help with two supporting facts");
        assertTrue(kv[0] > 0.6, "the key-value network learns");
    }
}
