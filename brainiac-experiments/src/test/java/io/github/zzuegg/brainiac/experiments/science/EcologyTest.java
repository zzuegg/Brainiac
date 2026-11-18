package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class EcologyTest {

    @Test
    void captureRecaptureAndOccupancyCorrectForDetection() {
        Map<String, double[]> a = Ecology.abundance(300, 0.3, 40, 1);
        System.out.printf("13.3.9 capture-recapture N=300 p=0.3 (40 studies): %s%n", Lines.of(a));
        Map<String, double[]> o = Ecology.occupancy(200, 3, 12, 2);
        System.out.printf("13.3.9 occupancy 200 sites x 3 visits (12 surveys): %s%n", Lines.of(o));
        assertTrue(Math.abs(a.get("Chapman (2 occasions) {rel bias, coverage}")[0]) < 0.05, Lines.of(a));
        assertTrue(a.get("naive distinct count {rel bias 2 occ, rel bias 5 occ}")[0] < -0.3, Lines.of(a));
        double[] m = o.get("occupancy model {mean-psi rel bias, cover b0, cover b1, cover a1}");
        double[] n = o.get("naive {share-detected rel bias, logistic slope on x (truth 1),"
                + " slope on detection covariate (truth 0)}");
        assertTrue(Math.abs(m[0]) < Math.abs(n[0]), Lines.of(o));
    }

    @Test
    void poissonProcessModelsMapTheIntensity() {
        Map<String, double[]> r = Ecology.distributions(48, 300, 3);
        System.out.printf("13.3.10 species distributions 48x48 {survey AUROC, log-intensity corr}: %s%n", Lines.of(r));
        assertTrue(r.get("unbiased: IPP")[1] > r.get("unbiased: KDE of locations")[1], Lines.of(r));
        assertTrue(r.get("road bias: IPP + bias covariate")[1] > r.get("road bias: IPP")[1], Lines.of(r));
    }
}
