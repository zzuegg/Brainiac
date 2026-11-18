package io.github.zzuegg.brainiac.experiments.shift;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;

class SelectivePredictionTest {

    @Test
    void riskCoverageOfSelectiveMethods() {
        List<SelectivePrediction.Row> rows = SelectivePrediction.selective(400, 1);
        for (SelectivePrediction.Row r : rows) {
            System.out.printf("Selective %-28s acc %.3f  AURC %.4f  risk@80 %.3f  | shifted AURC %.4f  risk@80 %.3f%n",
                    r.name(), r.accuracy(), r.aurc(), r.riskAt80(), r.shiftedAurc(), r.shiftedRiskAt80());
        }
        Map<String, SelectivePrediction.Row> by = rows.stream().collect(Collectors.toMap(SelectivePrediction.Row::name,
                Function.identity()));
        SelectivePrediction.Row sr = by.get("softmax response");
        assertTrue(sr.riskAt80() < 1 - sr.accuracy(), "abstaining lowers the risk");
        assertTrue(by.get("SelectiveNet").riskAt80() < 1 - by.get("SelectiveNet").accuracy(), "SelectiveNet selects");
    }

    @Test
    void learnedDeferralUsesTheExpert() {
        List<SelectivePrediction.DeferRow> rows = SelectivePrediction.defer(400, new double[] {0.2, 0.4}, 1);
        for (SelectivePrediction.DeferRow r : rows) {
            System.out.printf("Defer %-30s own rate %.3f acc %.3f  at budgets %s  expert ECE %.3f%n", r.name(),
                    r.ownRate(), r.ownAccuracy(), r.accuracyAt(), r.expertEce());
        }
        Map<String, SelectivePrediction.DeferRow> by = rows.stream().collect(Collectors.toMap(
                SelectivePrediction.DeferRow::name, Function.identity()));
        double alone = by.get("classifier alone").ownAccuracy();
        assertTrue(by.get("one-vs-all (Verma-Nalisnick)").accuracyAt().get(0.4) > alone, "deferral helps");
        assertTrue(by.get("softmax (Mozannar-Sontag)").accuracyAt().get(0.4) > alone, "deferral helps");
    }
}
