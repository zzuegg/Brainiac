package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class HashingRetrievalTest {

    @Test
    void hashing() {
        List<HashingRetrieval.Row> rows = HashingRetrieval.run(80, 60, 10, new int[] {16, 32, 64}, 300, 91);
        StringBuilder line = new StringBuilder("11.5.16 mAP seen/unseen, P@50, us/query:");
        int bits = -1;
        for (HashingRetrieval.Row r : rows) {
            if (r.bits() != bits) {
                bits = r.bits();
                line.append(" | ").append(bits == 0 ? "float" : bits + " bits").append(':');
            }
            line.append(String.format(" %s %.3f/%.3f/%.3f/%.0f;", r.method(), r.seenMap(), r.unseenMap(),
                    r.precision(), r.micros()));
        }
        System.out.println(line);
        assertTrue(rows.get(0).seenMap() > 0.3, line.toString());
    }
}
