package io.github.zzuegg.brainiac.experiments.evolution;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class EvolvedKernelsTest {

    @Test
    void stableSoftmaxIsRightWhereTheNaiveOneOverflows() {
        List<EvolvedKernels.Instr> stable = List.of(new EvolvedKernels.Instr(EvolvedKernels.Op.MAX, 0, 0, 0),
                new EvolvedKernels.Instr(EvolvedKernels.Op.SUBS, 1, 0, 0),
                new EvolvedKernels.Instr(EvolvedKernels.Op.EXP, 1, 1, 0),
                new EvolvedKernels.Instr(EvolvedKernels.Op.SUM, 1, 1, 0),
                new EvolvedKernels.Instr(EvolvedKernels.Op.RECIP, 1, 1, 0),
                new EvolvedKernels.Instr(EvolvedKernels.Op.MULS, 1, 1, 1));
        assertEquals(1.0, EvolvedKernels.softmaxCorrectness(stable));
        assertEquals(0.5, EvolvedKernels.softmaxCorrectness(EvolvedKernels.naiveSoftmax()));
        assertTrue(EvolvedKernels.matmulMisses(new EvolvedKernels.Matmul("ikj", 8, 32, 8)) > 0);
    }

    @Test
    void guidedEvolutionFindsTheStableKernelAndATiledMatmul() {
        EvolvedKernels.Result r = EvolvedKernels.run(8000, 120, 1);
        System.out.printf("evolved kernels: softmax fitness AlphaEvolve-style %.2f (at half budget %.2f) vs blind "
                + "mutation %.2f (half %.2f), regularized evolution %.2f, naive %.2f; best [%s]; matmul cache misses "
                + "%d vs blind %d, regularized %d, textbook ijk %d (%s)%n", r.softmaxGuided(), r.softmaxGuidedHalf(),
                r.softmaxRandom(), r.softmaxRandomHalf(), r.softmaxRegularized(), r.softmaxNaive(), r.softmaxBest(),
                r.matmulGuided(), r.matmulRandom(), r.matmulRegularized(), r.matmulNaive(), r.matmulBest());
        assertTrue(r.softmaxGuided() > 12, "a stable kernel is right on every test: " + r);
        assertTrue(r.softmaxGuided() >= r.softmaxRandom(), r.toString());
        assertTrue(r.softmaxGuided() >= r.softmaxRegularized(), r.toString());
        assertTrue(r.matmulGuided() < r.matmulNaive() / 2, r.toString());
    }
}
