package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.AioliMixer;
import io.github.zzuegg.brainiac.experimental.data.BanditMixer;

import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class OnlineMixingTest {

    @Test
    void onlineMixingAgainstOfflineMixtures() {
        Map<DataCuration.OnlineMixing, DataCuration.OnlineMixRun> runs = new EnumMap<>(
                DataCuration.OnlineMixing.class);
        for (DataCuration.OnlineMixing m : DataCuration.OnlineMixing.values()) {
            runs.put(m, DataCuration.onlineMixture(m, 300, 1));
            System.out.println("online mixing, 300 steps: " + runs.get(m));
        }
        DataCuration.OnlineMixRun odm = runs.get(DataCuration.OnlineMixing.ODM);
        DataCuration.OnlineMixRun clipped = runs.get(DataCuration.OnlineMixing.ODM_CLIPPED);
        DataCuration.OnlineMixRun uniform = runs.get(DataCuration.OnlineMixing.UNIFORM);
        DataCuration.OnlineMixRun aioli = runs.get(DataCuration.OnlineMixing.AIOLI);
        DataCuration.OnlineMixRun doremi = runs.get(DataCuration.OnlineMixing.DOREMI);
        System.out.printf("online mixing: target loss ODM %.4f, clipped ODM %.4f, Aioli %.4f vs uniform %.4f, DoReMi "
                + "%.4f (incl. proxies); noisy-domain weight ODM %.2f vs clipped %.2f%n", odm.targetLoss(),
                clipped.targetLoss(), aioli.targetLoss(), uniform.targetLoss(), doremi.targetLoss(), odm.weights()[2],
                clipped.weights()[2]);
        for (DataCuration.OnlineMixRun r : runs.values()) {
            assertEquals(1.0, Arrays.stream(r.weights()).sum(), 1e-6, r.toString());
            assertTrue(Double.isFinite(r.targetLoss()), r.toString());
        }
        assertTrue(clipped.weights()[2] <= odm.weights()[2] + 1e-9, "clipping limits the noisy domain's pull");
        assertTrue(aioli.averageLoss() < uniform.averageLoss() + 0.1, "Aioli does not lose badly to uniform");
    }

    @Test
    void twoDomainsAgainstTheGridOptimum() {
        DataCuration.TwoDomain t = DataCuration.twoDomain(200, 2);
        System.out.printf("two domains, 200 steps: grid optimum share %.1f (loss %.4f), uniform %.4f; Aioli share %.2f "
                + "(loss %.4f, distance %.2f); ODM share %.2f (loss %.4f, distance %.2f)%n", t.gridOptimum(),
                t.gridLoss(), t.uniformLoss(), t.aioliShare(), t.aioliLoss(), Math.abs(t.aioliShare()
                        - t.gridOptimum()), t.odmShare(), t.odmLoss(), Math.abs(t.odmShare() - t.gridOptimum()));
        assertTrue(t.gridLoss() <= t.uniformLoss() + 1e-9);
    }

    @Test
    void banditAndAioliMechanics() {
        BanditMixer b = BanditMixer.of(3, 0.5);
        for (int i = 0; i < 200; i++) {
            b.update(new double[] {1, 3, 1}, new boolean[] {true, true, true});
        }
        assertTrue(b.weights()[1] > b.weights()[0], Arrays.toString(b.weights()));
        AioliMixer a = new AioliMixer(2, 0.5, 1e-6, 0.5, 9);
        double[] loss = {2, 2};
        for (int c = 0; c < 30; c++) {
            double[] m = a.proportions();
            loss = new double[] {loss[0] - 0.05 * m[0], loss[1] - 0.01 * m[1]};
            a.observe(loss);
        }
        assertTrue(a.weights()[0] > 0.5, "Aioli moves towards the group whose training lowers loss most: "
                + Arrays.toString(a.weights()));
    }
}
