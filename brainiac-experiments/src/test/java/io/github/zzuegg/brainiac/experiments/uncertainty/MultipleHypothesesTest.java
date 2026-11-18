package io.github.zzuegg.brainiac.experiments.uncertainty;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class MultipleHypothesesTest {

    private static MultipleHypotheses.Row row(List<MultipleHypotheses.Row> rows, String m) {
        return rows.stream().filter(r -> r.method().equals(m)).findFirst().orElseThrow();
    }

    @Test
    void winnerTakesAllFindsTheModes() {
        for (MultipleHypotheses.Task t : new MultipleHypotheses.Task[] {MultipleHypotheses.Task.ARM,
            MultipleHypotheses.Task.TURN}) {
            List<MultipleHypotheses.Row> rows = MultipleHypotheses.run(t, 4, 300, 1);
            StringBuilder b = new StringBuilder("11.4.15 " + t + " oracle/coverage/NLL:");
            rows.forEach(r -> b.append(String.format(" %s %.3f/%.3f/%.3f;", r.method(), r.oracle(), r.coverage(),
                    r.nll())));
            System.out.println(b);
            assertTrue(row(rows, "wta").coverage() > row(rows, "mse").coverage(),
                    t + ": WTA heads cover more of the solution set than the mean regressor");
        }
    }
}
