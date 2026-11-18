package io.github.zzuegg.brainiac.experiments.streaming;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class StreamShootoutWave12Test {

    private static double get(List<StreamShootout.ClusterRow> rows, String stream, String method) {
        return rows.stream().filter(r -> r.stream().equals(stream) && r.method().startsWith(method)).findFirst()
                .orElseThrow().ari();
    }

    @Test
    void streamClusterersTrackEvolvingBlobs() {
        List<StreamShootout.ClusterRow> rows = StreamShootout.clustering(6_000, 1);
        for (StreamShootout.ClusterRow r : rows) {
            System.out.printf("12.2.1 %-15s %-26s windowed ARI %.3f, %5d summaries, %9.0f points/s%n", r.stream(),
                    r.method(), r.ari(), r.summaries(), r.pointsPerSecond());
        }
        assertTrue(get(rows, "evolving blobs", "DenStream") > 0.5, "DenStream tracks the blobs");
        assertTrue(get(rows, "evolving blobs", "CluStream") > 0.4, "CluStream tracks the blobs");
        assertTrue(get(rows, "moons", "DenStream") > get(rows, "moons", "sequential k-means"),
                "density clusters beat k-means on moons");
    }

    @Test
    void streamingDetectorsFindAnomalies() {
        List<StreamShootout.AnomalyRow> rows = StreamShootout.anomalies(5_000, 2);
        for (StreamShootout.AnomalyRow r : rows) {
            System.out.printf("12.2.2 %-40s AUROC %.3f, AUPRC %.3f, AUROC after drift %.3f, %9.0f points/s%n",
                    r.method(), r.auroc(), r.auprc(), r.aurocAfterDrift(), r.pointsPerSecond());
        }
        for (StreamShootout.AnomalyRow r : rows) {
            assertTrue(r.auroc() > 0.6, r.toString());
        }
    }

    @Test
    void streamingPcaFindsTheSpikedSubspace() {
        List<StreamShootout.SubspaceRow> rows = StreamShootout.subspaces(40, 4, 3_000, 3);
        for (StreamShootout.SubspaceRow r : rows) {
            System.out.printf("12.2.3 %-15s %-36s sin(theta) to truth %.4f, to batch SVD %.2e, %7.1f us/point%n",
                    r.setting(), r.method(), r.sinTheta(), r.sinThetaBatch(), r.microsPerPoint());
        }
        StreamShootout.SubspaceRow svd = rows.stream()
                .filter(r -> r.setting().startsWith("large") && r.method().startsWith("incremental")).findFirst()
                .orElseThrow();
        StreamShootout.SubspaceRow cc = rows.stream()
                .filter(r -> r.setting().startsWith("large") && r.method().startsWith("CCIPCA")).findFirst()
                .orElseThrow();
        assertTrue(svd.sinThetaBatch() < 0.05, svd.toString());
        assertTrue(cc.sinTheta() < 0.5, cc.toString());
    }

    @Test
    void mondrianForestsLearnOnlineAndWidenInGaps() {
        List<StreamShootout.ForestRow> rows = StreamShootout.forests(1_500, 4);
        for (StreamShootout.ForestRow r : rows) {
            System.out.printf("12.2.4 %-22s %-44s score %.3f (first fifth %.3f), %.2f s%n", r.task(), r.method(),
                    r.score(), r.early(), r.seconds());
        }
        double[] gap = StreamShootout.gapUncertainty(300, 5);
        System.out.printf("12.2.4 predictive sd in the input gap / on the data: Mondrian %.2f, random forest %.2f%n",
                gap[0], gap[1]);
        StreamShootout.ForestRow mondrian = rows.get(0);
        StreamShootout.ForestRow hoeffding = rows.get(1);
        assertTrue(mondrian.score() > 0.5, mondrian.toString());
        assertTrue(mondrian.early() > hoeffding.early() - 0.05, "Mondrian is not worse early in the stream");
        assertTrue(gap[0] > 1, "Mondrian widens its predictive sd in the gap");
    }
}
