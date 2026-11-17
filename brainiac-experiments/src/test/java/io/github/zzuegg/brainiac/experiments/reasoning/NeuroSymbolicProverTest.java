package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class NeuroSymbolicProverTest {

    @Test
    void engineNeedsTheAuxiliaryTerm() {
        // a = c and b = d prove ab = cd only through ad or cb
        NeuroSymbolicProver.Problem p = new NeuroSymbolicProver.Problem(
                List.of(new String[] {"a", "c"}, new String[] {"b", "d"}), new String[] {"ab", "cd"},
                List.of("ab", "bc", "cd"));
        assertEquals(-1, NeuroSymbolicProver.prove(p, 0, null, new java.util.SplittableRandom(1)));
        assertEquals(1, NeuroSymbolicProver.prove(p, 1, null, new java.util.SplittableRandom(1)));
    }

    @Test
    void neuralProposerProvesMoreThanTheEngineAlone() {
        NeuroSymbolicProver.Result r = NeuroSymbolicProver.run(90, 6, 300, 1);
        System.out.printf("toy algebra proof rate: engine + neural proposer %.3f (%.2f aux terms) vs random aux %.3f "
                + "vs engine alone %.3f%n", r.neural(), r.neuralAux(), r.random(), r.symbolic());
        assertTrue(r.neural() > r.symbolic() + 0.4, r.toString());
        assertTrue(r.neural() > r.random() + 0.1, r.toString());
    }
}
