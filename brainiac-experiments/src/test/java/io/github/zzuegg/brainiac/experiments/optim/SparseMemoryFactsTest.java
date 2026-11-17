package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.optim.SparseMemoryFacts.Method;
import io.github.zzuegg.brainiac.experiments.optim.SparseMemoryFacts.Outcome;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class SparseMemoryFactsTest {

    @Test
    void sparseMemorySlotsLearnNewFactsWithLittleForgetting() {
        Map<Method, Outcome> r = new EnumMap<>(Method.class);
        StringBuilder line = new StringBuilder("Fact stream (100 new facts in batches of 10), new/background accuracy:");
        for (Method m : Method.values()) {
            Outcome o = SparseMemoryFacts.run(m, 10, 60, 1);
            r.put(m, o);
            line.append(String.format(" %s %.2f/%.2f;", m, o.newFacts(), o.background()));
        }
        line.append(String.format(" (background after pretraining %.2f)", r.get(Method.FULL).backgroundBefore()));
        System.out.println(line);
        Outcome sparse = r.get(Method.SPARSE_MEMORY);
        Outcome full = r.get(Method.FULL);
        assertTrue(sparse.newFacts() > 0.5, line.toString());
        assertTrue(sparse.background() > full.background() + 0.1, "less forgetting than full fine-tuning: " + line);
    }
}
