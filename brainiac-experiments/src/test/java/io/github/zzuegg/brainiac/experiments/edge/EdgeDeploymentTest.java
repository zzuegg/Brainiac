package io.github.zzuegg.brainiac.experiments.edge;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.edge.KeywordSpotters;
import java.util.List;
import org.junit.jupiter.api.Test;

class EdgeDeploymentTest {

    @Test
    void integerEngineIsBitExactAndCloseToFloat() {
        List<EdgeDeployment.IntegerRow> rows = EdgeDeployment.integer(250, 1);
        for (EdgeDeployment.IntegerRow r : rows) {
            System.out.printf("9.6.13 %-16s %-56s acc %.3f, agree %.3f, bit-exact %.2f, %7d B, %9d MACs, %8.1f us%n",
                    r.model(), r.method(), r.accuracy(), r.agreement(), r.bitExact(), r.bytes(), r.macs(),
                    r.microsPerSample());
        }
        for (EdgeDeployment.IntegerRow r : rows) {
            if (r.method().contains("integer engine")) {
                assertEquals(1.0, r.bitExact(), 1e-12, r.toString());
            }
        }
        EdgeDeployment.IntegerRow fl = rows.get(0);
        EdgeDeployment.IntegerRow qat = rows.get(4);
        assertTrue(qat.accuracy() > fl.accuracy() - 0.05, qat + " vs " + fl);
        assertTrue(qat.bytes() * 3 < fl.bytes(), "int8 is about 4x smaller");
    }

    @Test
    void patchInferenceCutsPeakMemoryExactly() {
        List<EdgeDeployment.PatchRow> rows = EdgeDeployment.patches(64, new int[] {2, 4}, 2);
        for (EdgeDeployment.PatchRow r : rows) {
            System.out.printf("9.6.14 %-22s grid %d peak %7d B, %9d MACs (+%.1f%%), max diff %.2e%n", r.network(),
                    r.grid(), r.peakBytes(), r.macs(), 100 * r.extraMacShare(), r.maxDifference());
        }
        for (EdgeDeployment.OrderRow r : EdgeDeployment.reordering(2)) {
            System.out.printf("9.6.14 %-46s given order %6d B, best %6d B (%s)%n", r.graph(), r.givenPeak(),
                    r.bestPeak(), r.bestOrder());
            assertTrue(r.bestPeak() <= r.givenPeak());
        }
        for (EdgeDeployment.BudgetRow r : EdgeDeployment.budget(32, 120, 2)) {
            System.out.printf("9.6.14 %-48s side %2d peak %6d B, %8d MACs, accuracy %.3f%n", r.method(), r.side(),
                    r.peakBytes(), r.macs(), r.accuracy());
        }
        EdgeDeployment.PatchRow layer = rows.get(0);
        EdgeDeployment.PatchRow patch = rows.get(2);
        assertTrue(patch.peakBytes() * 2 < layer.peakBytes(), patch + " vs " + layer);
        assertTrue(patch.maxDifference() < 1e-3, "patch execution matches layer by layer");
    }

    @Test
    void onceForAllSubnetsWorkWithoutRetraining() {
        EdgeDeployment.OfaResult res = EdgeDeployment.onceForAll(300, 3, 3);
        for (EdgeDeployment.OfaRow r : res.rows()) {
            System.out.printf("9.6.15 %-38s %-44s %8d MACs, peak %6d B, accuracy %.3f, %5d steps%n", r.method(),
                    r.subnet(), r.macs(), r.peakBytes(), r.accuracy(), r.trainingSteps());
        }
        for (EdgeDeployment.SearchRow r : res.search()) {
            System.out.printf("9.6.15 budget %-8s %-28s %-44s %8d MACs, predicted %.3f, measured %.3f%n", r.budget(),
                    r.method(), r.subnet(), r.macs(), r.predicted(), r.measured());
        }
        System.out.printf("9.6.15 predictor Kendall tau %.2f%n", res.predictorTau());
        double ofaSmall = res.rows().stream().filter(r -> r.method().startsWith("OFA")).skip(1).findFirst()
                .orElseThrow().accuracy();
        double bigSmall = res.rows().stream().filter(r -> r.method().startsWith("largest")).skip(1).findFirst()
                .orElseThrow().accuracy();
        assertTrue(ofaSmall > bigSmall, "progressive shrinking helps the smallest subnet: " + ofaSmall + " vs "
                + bigSmall);
    }

    @Test
    void sparseUpdateWithQasAdaptsUnderBudget() {
        List<EdgeDeployment.AdaptRow> rows = EdgeDeployment.adaptation(200, 120, 0.3, 4);
        for (EdgeDeployment.AdaptRow r : rows) {
            System.out.printf("9.6.16 %-52s %7d B, accuracy %.3f  [%s]%n", r.method(), r.trainingBytes(),
                    r.accuracy(), r.detail());
        }
        double none = rows.get(0).accuracy();
        double qas = rows.stream().filter(r -> r.method().startsWith("sparse update (")).findFirst().orElseThrow()
                .accuracy();
        assertTrue(qas > none + 0.1, "adaptation learns the new glyphs: " + qas + " vs " + none);
    }

    @Test
    void dsCnnKeywordSpotterFitsSmallBudget() {
        List<EdgeDeployment.KeywordRow> rows = EdgeDeployment.keywords(List.of(KeywordSpotters.BUDGETS.get(0)),
                KeywordSpotters.Family.values(), 200, 5);
        rows.addAll(EdgeDeployment.keywordBaselines(150, 3, 5));
        for (EdgeDeployment.KeywordRow r : rows) {
            System.out.printf("9.6.17 %-13s %-34s %7d B, %8d MACs, fits %-5s peak %6d B, float %.3f, int8 %.3f%n",
                    r.budget(), r.family(), r.weightBytes(), r.macs(), r.fits(), r.peakBytes(), r.floatAccuracy(),
                    r.int8Accuracy());
        }
        EdgeDeployment.KeywordRow ds = rows.stream().filter(r -> r.family().equals("DS_CNN")).findFirst()
                .orElseThrow();
        assertTrue(ds.fits(), ds.toString());
        assertTrue(ds.floatAccuracy() > 1.0 / EdgeDeployment.KWS_CLASSES + 0.2, ds.toString());
    }
}
