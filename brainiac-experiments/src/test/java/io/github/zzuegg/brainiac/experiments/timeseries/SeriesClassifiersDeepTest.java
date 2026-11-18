package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SeriesClassifiersDeepTest {

    @Test
    void deepClassifiersOnWarpedTemplates() {
        Map<String, double[]> r = SeriesClassifiers.deep(SeriesClassifiers.split("warped", 90, 120, 64, 5), 50, 20, 5);
        r.forEach((k, v) -> System.out.printf("deep warped %s: acc %.3f (%.0f ms, %.0f params)%n", k, v[0], v[1],
                v[2]));
        assertTrue(r.get("INCEPTION_TIME")[0] > 0.6, "InceptionTime " + r.get("INCEPTION_TIME")[0]);
        assertTrue(r.get("LITE")[2] < r.get("INCEPTION_TIME")[2] / 3, "LITE is much smaller");
        double inception = r.get("INCEPTION_TIME")[0];
        assertTrue(inception > r.get("FCN")[0] && inception > r.get("RESNET1D")[0] && inception > r.get("TCN")[0],
                "InceptionTime beats FCN, ResNet and the TCN on warped templates");
    }
}
