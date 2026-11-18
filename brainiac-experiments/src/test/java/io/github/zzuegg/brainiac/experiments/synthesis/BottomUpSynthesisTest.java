package io.github.zzuegg.brainiac.experiments.synthesis;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.synthesis.ProgramSynthesisShootout.Method;
import io.github.zzuegg.brainiac.experiments.synthesis.ProgramSynthesisShootout.Row;

import java.util.List;

import org.junit.jupiter.api.Test;

/** Items 11.2.12-11.2.13: BUSTLE, CrossBeam and Probe against unguided bottom-up and DeepCoder. */
class BottomUpSynthesisTest {

    static Row row(List<Row> rows, Method m, String label) {
        return rows.stream().filter(r -> r.method() == m && r.label().equals(label)).findFirst().orElseThrow();
    }

    @Test
    void guidedBottomUpSearch() {
        List<Row> rows = ProgramSynthesisShootout.bottomUp(ProgramSynthesisShootoutTest.TASKS,
                new long[] {1_000, 10_000, 100_000}, 200, 100, 250, 1);
        System.out.println("Bottom-up synthesis, 30 tasks of 2-5 primitives: " + ProgramSynthesisShootoutTest.line(rows));
        String b = "budget 10000";
        System.out.printf("11.2.12-13 solved at 10^4 (len 4-5 at 10^5): bottom-up %.2f (%.2f), BUSTLE %.2f (%.2f),"
                + " CrossBeam %.2f (%.2f), Probe %.2f (%.2f), DeepCoder ordered %.2f (%.2f), enumerate %.2f (%.2f)%n",
                row(rows, Method.BOTTOM_UP, b).solved(), row(rows, Method.BOTTOM_UP, "len 4-5").solved(),
                row(rows, Method.BUSTLE, b).solved(), row(rows, Method.BUSTLE, "len 4-5").solved(),
                row(rows, Method.CROSSBEAM, b).solved(), row(rows, Method.CROSSBEAM, "len 4-5").solved(),
                row(rows, Method.PROBE, b).solved(), row(rows, Method.PROBE, "len 4-5").solved(),
                row(rows, Method.DEEPCODER_ORDERED, b).solved(), row(rows, Method.DEEPCODER_ORDERED, "len 4-5").solved(),
                row(rows, Method.ENUMERATE, b).solved(), row(rows, Method.ENUMERATE, "len 4-5").solved());
        assertTrue(row(rows, Method.BOTTOM_UP, "budget 100000").consistent()
                >= row(rows, Method.ENUMERATE, "budget 100000").consistent() - 0.05,
                "unit-cost bottom-up matches breadth-first enumeration");
    }
}
