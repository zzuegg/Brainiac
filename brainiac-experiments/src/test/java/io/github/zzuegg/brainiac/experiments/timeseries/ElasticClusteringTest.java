package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.tsc.ElasticDistances;
import io.github.zzuegg.brainiac.experimental.tsc.SoftDtw;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ElasticClusteringTest {

    @Test
    void softDtwGradientMatchesFiniteDifferencesAndTendsToDtw() {
        double[] x = {0, 1, 3, 2, 0.5, -1};
        double[] y = {0.2, 0.8, 2.5, 2.5, 0, -0.5, -1};
        double[] g = SoftDtw.gradient(x, y, 0.5);
        for (int i = 0; i < x.length; i++) {
            double[] a = x.clone();
            double[] b = x.clone();
            a[i] += 1e-5;
            b[i] -= 1e-5;
            double fd = (SoftDtw.compute(a, y, 0.5).value() - SoftDtw.compute(b, y, 0.5).value()) / 2e-5;
            assertEquals(fd, g[i], 1e-4, "soft-DTW gradient " + i);
        }
        assertEquals(ElasticDistances.dtw(x, y, -1), SoftDtw.compute(x, y, 1e-3).value(), 1e-2);
        Var p = Var.param(Tensor.of(Shape.of(1, 6), 0, 1, 3, 2, 0.5f, -1));
        SoftDtw.loss(p, Tensor.of(Shape.of(1, 7), 0.2f, 0.8f, 2.5f, 2.5f, 0, -0.5f, -1), 0.5).backward();
        assertEquals(g[2], p.grad().toArray()[2], 1e-3);
    }

    @Test
    void shapeClusteringBeatsEuclideanKMeans() {
        Map<String, Double> r = ElasticClustering.clustering("phase", 60, 64, 1);
        r.forEach((k, v) -> System.out.printf("elastic clustering phase-shifted %s: ARI %.3f%n", k, v));
        assertTrue(r.get("k-Shape") > r.get("k-means (Euclidean)") + 0.1, "k-Shape vs k-means");
        Map<String, Double> w = ElasticClustering.clustering("warped", 60, 64, 2);
        w.forEach((k, v) -> System.out.printf("elastic clustering warped %s: ARI %.3f%n", k, v));
        assertTrue(w.get("DBA k-means") > w.get("k-means (Euclidean)"), "DBA k-means vs k-means");
    }

    @Test
    void barycentresAndSoftDtwForecasts() {
        Map<String, Double> a = ElasticClustering.averaging(12, 64, 3);
        a.forEach((k, v) -> System.out.printf("elastic averaging %s: %.3f%n", k, v));
        assertTrue(a.get("DBA") < a.get("Euclidean mean"), "DBA closer to the template than the mean");
        Map<String, double[]> f = ElasticClustering.forecasting(300, 4);
        f.forEach((k, v) -> System.out.printf("soft-DTW forecasting %s: MSE %.3f DTW %.3f peak %.2f (true %.2f)%n", k,
                v[0], v[1], v[2], v[3]));
        assertTrue(f.get("soft-DTW")[2] > f.get("MSE")[2], "soft-DTW keeps sharper peaks");
    }
}
