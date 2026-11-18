package io.github.zzuegg.brainiac.experiments.privacy;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.safety.PrivateTraining;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class FederatedFollowUpsTest {

    @Test
    void clusteringAndLocalNormalisationHandleHeterogeneity() {
        List<FederatedFollowUps.FedRow> rows = new ArrayList<>(FederatedFollowUps.permutedLabels(20, 4, 20, 10, 0.5,
                1));
        rows.addAll(FederatedFollowUps.featureShift(20, 20, 10, 0.5, 2));
        rows.addAll(FederatedFollowUps.labelSkew(0.1, 20, 20, 10, 0.5, 0.7, 3));
        for (FederatedFollowUps.FedRow r : rows) {
            System.out.printf("12.2.12 %-34s %-36s mean %.3f, worst client %.3f, rounds to target %3d, ARI %.2f%n",
                    r.setting(), r.method(), r.meanAccuracy(), r.worstClient(), r.roundsToTarget(), r.note());
        }
        double fedAvg = pick(rows, "permuted", "FedAvg").meanAccuracy();
        double ifca = pick(rows, "permuted", "IFCA (4").meanAccuracy();
        assertTrue(ifca > fedAvg + 0.2, "IFCA beats one model on permuted labels");
        assertTrue(pick(rows, "feature", "FedBN").meanAccuracy() > pick(rows, "feature", "FedAvg").meanAccuracy(),
                "FedBN beats FedAvg under feature shift");
    }

    private static FederatedFollowUps.FedRow pick(List<FederatedFollowUps.FedRow> rows, String setting,
            String method) {
        return rows.stream().filter(r -> r.setting().startsWith(setting) || r.setting().contains(setting))
                .filter(r -> r.method().startsWith(method)).findFirst().orElseThrow();
    }

    @Test
    void clippingStrategiesNeedNoTuning() {
        List<PrivateTraining.Clipping> rows = PrivateTraining.clipping(new double[] {1, 3, 8}, 100, 3, 0.05, 4);
        for (PrivateTraining.Clipping r : rows) {
            System.out.printf("12.2.13 eps %.0f %-42s accounted eps %.2f, accuracy %.3f, final C %.3f%n",
                    r.targetEpsilon(), r.method(), r.epsilon(), r.accuracy(), r.finalClipNorm());
        }
        for (PrivateTraining.Clipping r : rows) {
            assertTrue(r.epsilon() <= r.targetEpsilon() * 1.01, r.toString());
        }
        double best = rows.stream().filter(r -> r.targetEpsilon() == 8 && r.method().startsWith("DP-SGD"))
                .mapToDouble(PrivateTraining.Clipping::accuracy).max().orElseThrow();
        double auto = rows.stream().filter(r -> r.targetEpsilon() == 8 && r.method().startsWith("automatic"))
                .findFirst().orElseThrow().accuracy();
        assertTrue(auto > best - 0.15, "automatic clipping is near the best fixed C");
    }
}
