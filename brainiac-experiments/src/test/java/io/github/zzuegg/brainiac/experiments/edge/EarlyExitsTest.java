package io.github.zzuegg.brainiac.experiments.edge;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class EarlyExitsTest {

    @Test
    void exitsSaveComputeAtNearFullAccuracy() {
        List<EarlyExits.ExitRow> rows = new ArrayList<>(EarlyExits.run("digits", 2_000, 300, 0.9, 1));
        rows.addAll(EarlyExits.run("shapes", 1_000, 200, 0.9, 2));
        for (EarlyExits.ExitRow r : rows) {
            System.out.printf("12.2.11 %-7s %-42s acc %.3f at %.2f of full MACs, area %.3f, overthinking %.3f,"
                    + " ECE %.3f, EEFP %.3f%n", r.task(), r.method(), r.accuracy(), r.macsShare(), r.area(),
                    r.overthinking(), r.ece(), r.eefp());
        }
        for (String task : new String[] {"digits", "shapes"}) {
            EarlyExits.ExitRow scaled = rows.stream().filter(r -> r.task().equals(task)
                    && r.method().startsWith("joint + temperature")).findFirst().orElseThrow();
            assertTrue(scaled.macsShare() < 0.95, scaled.toString());
            assertTrue(scaled.eefp() > 0.6, scaled.toString());
        }
    }
}
