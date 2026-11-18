package io.github.zzuegg.brainiac.experiments.shift;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class DomainAdaptationTest {

    static void print(String task, Map<DomainAdaptation.Method, DomainAdaptation.Result> r) {
        r.forEach((m, v) -> System.out.printf("DA %-14s %-11s target %.3f  source %.3f  classes %2d  gap closed %5.2f%n",
                task, m, v.targetAccuracy(), v.sourceAccuracy(), v.classesPredicted(),
                DomainAdaptation.gapClosed(r, m)));
    }

    @Test
    void shapesOntoTexturedBackgrounds() {
        Map<DomainAdaptation.Method, DomainAdaptation.Result> r = DomainAdaptation.adapt(
                DomainAdaptation.Task.SHAPES_TEXTURE, List.of(DomainAdaptation.Method.SOURCE_ONLY,
                        DomainAdaptation.Method.ORACLE, DomainAdaptation.Method.DANN, DomainAdaptation.Method.CDAN_E, DomainAdaptation.Method.CORAL,
                        DomainAdaptation.Method.MMD, DomainAdaptation.Method.ADABN, DomainAdaptation.Method.TENT,
                        DomainAdaptation.Method.SHOT), 250, 1);
        print("shapes", r);
        double src = r.get(DomainAdaptation.Method.SOURCE_ONLY).targetAccuracy();
        assertTrue(r.get(DomainAdaptation.Method.ORACLE).targetAccuracy() > src, "the oracle beats source-only");
        double best = Math.max(Math.max(r.get(DomainAdaptation.Method.DANN).targetAccuracy(),
                r.get(DomainAdaptation.Method.CORAL).targetAccuracy()), r.get(DomainAdaptation.Method.SHOT)
                .targetAccuracy());
        assertTrue(best > src, "some adaptation beats source-only: " + best + " vs " + src);
    }
}
