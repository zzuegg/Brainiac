package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;

class KernelLearnersTest {

    @Test
    void randomWeightNetworks() {
        for (String name : new String[] {"stress", "friedman", "spirals", "digits"}) {
            KernelLearners.Task t = KernelLearners.task(name, 1);
            Map<String, double[]> r = KernelLearners.randomWeights(t, 256, 600, 1);
            r.forEach((k, v) -> System.out.printf("random weights %s %s: %s %.3f (%.0f ms)%n", name, k,
                    t.regression() ? "RMSE" : "acc", v[0], v[1]));
            double rvfl = r.get("RVFL")[0];
            double linear = r.get("linear ridge")[0];
            if (!name.equals("digits")) {
                assertTrue(t.regression() ? rvfl < linear : rvfl > linear, "RVFL beats the linear read-out on " + name);
            }
        }
    }

    @Test
    void multipleKernelLearning() {
        for (KernelLearners.KernelTask kt : new KernelLearners.KernelTask[] {KernelLearners.groups(300, 2),
            KernelLearners.multiScale(300, 3)}) {
            Map<String, double[]> r = KernelLearners.mkl(kt, 1, 2);
            r.forEach((k, v) -> System.out.printf("MKL %s %s: acc %.3f (%.0f ms), informative share %.2f, weights %s%n",
                    kt.task().name(), k, v[0], v[1], v[2],
                    Arrays.toString(Arrays.stream(v, 3, v.length).map(w -> Math.round(w * 100) / 100.0).toArray())));
            if (kt.task().name().equals("groups")) {
                assertTrue(r.get("ALIGNF")[2] > 0.5, "ALIGNF selects the informative group");
                assertTrue(r.get("EasyMKL")[2] > 0.25, "EasyMKL up-weights the informative group");
            }
        }
    }
}
