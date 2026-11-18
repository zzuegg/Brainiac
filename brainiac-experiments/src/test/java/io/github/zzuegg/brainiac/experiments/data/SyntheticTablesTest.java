package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.data.SyntheticTraining.TableSource;
import io.github.zzuegg.brainiac.experiments.data.SyntheticTraining.TabularScore;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class SyntheticTablesTest {

    @Test
    void tabularDiffusionAgainstCopulaSmoteAndReal() {
        Map<TableSource, TabularScore> r = new EnumMap<>(TableSource.class);
        for (TableSource s : new TableSource[] {TableSource.REAL, TableSource.GAUSSIAN_COPULA, TableSource.SMOTE,
            TableSource.TABDDPM}) {
            r.put(s, SyntheticTraining.tabular(s, 1500, 64, 1));
        }
        StringBuilder line = new StringBuilder("synthetic tables TSTR (acc / rare recall / balanced / DCR ratio):");
        r.forEach((s, t) -> line.append(String.format(" %s %.3f/%.3f/%.3f/%.2f", s, t.accuracy(), t.rareRecall(),
                t.balancedAccuracy(), t.closestRecordRatio())));
        System.out.println(line);
        TabularScore real = r.get(TableSource.REAL);
        TabularScore copula = r.get(TableSource.GAUSSIAN_COPULA);
        TabularScore smote = r.get(TableSource.SMOTE);
        TabularScore ddpm = r.get(TableSource.TABDDPM);
        assertTrue(real.balancedAccuracy() > ddpm.balancedAccuracy(), "real data is the upper bound");
        assertTrue(ddpm.balancedAccuracy() > copula.balancedAccuracy(), "TabDDPM keeps the label structure");
        assertTrue(smote.closestRecordRatio() < ddpm.closestRecordRatio(), "interpolation sits nearer real rows");
        assertTrue(ddpm.closestRecordRatio() > 0.7, "TabDDPM does not copy training rows");
    }
}
