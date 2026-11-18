package io.github.zzuegg.brainiac.experiments.memory;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.memory.MemoryNetworkShootout.Architecture;

import org.junit.jupiter.api.Test;

class MemoryBaselinesTest {

    @Test
    void transformerNtmAndDncOnBabiLite() {
        StringBuilder line = new StringBuilder("bAbI-lite baselines on 1/2/3 supporting facts (150 steps):");
        double[] t = MemoryNetworkShootoutTest.run(Architecture.TRANSFORMER, 150, line);
        double[] ntm = MemoryNetworkShootoutTest.run(Architecture.NTM, 150, line);
        double[] dnc = MemoryNetworkShootoutTest.run(Architecture.DNC, 150, line);
        StringBuilder mem = new StringBuilder();
        double[] memn2n = MemoryNetworkShootoutTest.run(Architecture.MEMN2N_3_HOPS, 150, mem);
        System.out.println(line + " vs" + mem);
        assertTrue(ntm[0] > 0.3 && dnc[0] > 0.3, "NTM and DNC beat chance (0.25) on task 1");
        assertTrue(memn2n[0] > t[0] && memn2n[0] > ntm[0] && memn2n[0] > dnc[0],
                "MemN2N beats every baseline on single-fact questions at the same budget");
    }
}
