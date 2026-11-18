package io.github.zzuegg.brainiac.experiments.shift;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;

class OodShootoutTest {

    @Test
    void postHocScoresOnShapes() {
        List<OodShootout.Row> rows = OodShootout.postHoc(ShiftSuites.Family.SHAPES, 1);
        for (OodShootout.Row r : rows) {
            System.out.printf("OOD shapes %-14s near AUROC %.3f FPR95 %.3f  far AUROC %.3f FPR95 %.3f  acc %.3f  "
                    + "passes %d%n", r.name(), r.nearAuroc(), r.nearFpr95(), r.farAuroc(), r.farFpr95(), r.accuracy(),
                    r.passes());
        }
        Map<String, OodShootout.Row> by = rows.stream().collect(Collectors.toMap(OodShootout.Row::name,
                Function.identity()));
        assertTrue(by.get("MSP").accuracy() > 0.9, "classifier accuracy");
        assertTrue(by.get("MSP").nearAuroc() > 0.55, "MSP beats chance on near OOD");
        assertTrue(by.get("energy").farAuroc() > 0.52, "energy beats chance on far OOD");
        assertTrue(by.get("deep k-NN").farAuroc() > by.get("MSP").farAuroc(), "k-NN beats MSP on far OOD");
    }
}
