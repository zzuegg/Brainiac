package io.github.zzuegg.brainiac.experiments.shift;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class OodTrainingTest {

    @Test
    void outlierExposureRaisesFarOodAuroc() {
        Map<OodShootout.Training, List<OodShootout.Row>> rows = new EnumMap<>(OodShootout.Training.class);
        for (OodShootout.Training t : OodShootout.Training.values()) {
            rows.put(t, OodShootout.outlierExposure(t, 400, 1));
            for (OodShootout.Row r : rows.get(t)) {
                System.out.printf("OE %-20s near AUROC %.3f  far AUROC %.3f FPR95 %.3f  acc %.3f%n", r.name(),
                        r.nearAuroc(), r.farAuroc(), r.farFpr95(), r.accuracy());
            }
        }
        double plainFar = rows.get(OodShootout.Training.PLAIN).get(0).farAuroc();
        double oeFar = rows.get(OodShootout.Training.OE_UNIFORM).get(0).farAuroc();
        assertTrue(rows.get(OodShootout.Training.OE_UNIFORM).get(0).accuracy() > 0.85, "OE keeps accuracy");
        assertTrue(oeFar > plainFar, "OE " + oeFar + " vs plain " + plainFar);
    }

    @Test
    void openMaxAgainstMaxLogit() {
        OodShootout.OpenSetScores brief = OodShootout.openSet(false, 200, 1);
        OodShootout.OpenSetScores well = OodShootout.openSet(true, 200, 1);
        for (OodShootout.OpenSetScores s : List.of(brief, well)) {
            System.out.printf("Open set %-5s closed acc %.3f  AUROC max logit %.3f  OpenMax %.3f  conformal %.3f  "
                    + "open acc OpenMax %.3f  max logit %.3f%n", s == brief ? "brief" : "well", s.closedAccuracy(),
                    s.maxLogitAuroc(), s.openMaxAuroc(), s.conformalAuroc(), s.openMaxAccuracy(), s.maxLogitAccuracy());
        }
        assertTrue(brief.closedAccuracy() > 0.65, "closed accuracy " + brief.closedAccuracy());
        assertTrue(brief.openMaxAuroc() > 0.55 && brief.maxLogitAuroc() > 0.55, "both beat chance");
    }
}
