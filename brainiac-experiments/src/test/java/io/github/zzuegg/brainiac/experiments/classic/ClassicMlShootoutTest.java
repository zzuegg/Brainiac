package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.classic.ClassicMlShootout.Points;
import io.github.zzuegg.brainiac.experiments.classic.ClassicMlShootout.Row;

import java.util.Map;

import org.junit.jupiter.api.Test;

class ClassicMlShootoutTest {

    private static String line(Map<String, Row> rows, String format) {
        StringBuilder sb = new StringBuilder();
        rows.forEach((k, v) -> sb.append(k).append(' ').append(String.format(format, v.score())).append(" (")
                .append(String.format("%.0f ms, %d", v.millis(), v.size())).append(") "));
        return sb.toString().trim();
    }

    @Test
    void kernelSvmAndRandomFeaturesSolveSpiralsWhereLinearModelsFail() {
        Points train = ClassicMlShootout.spirals(80, 3, 0.15, 1);
        Points test = ClassicMlShootout.spirals(50, 3, 0.15, 2);
        Map<String, Row> r = ClassicMlShootout.kernelMachines(train, test, 10, 3);
        System.out.printf("spirals test accuracy: %s%n", line(r, "%.3f"));
        assertTrue(r.get("SVM-RBF").score() > 0.9, r.toString());
        assertTrue(r.get("SVM-RBF").score() > r.get("logistic").score() + 0.3, r.toString());
        assertTrue(r.get("RFF+linear").score() > r.get("linear SVM").score() + 0.3, r.toString());
        assertTrue(r.get("SVM-RBF").size() < train.size() * 2, "support vectors " + r.get("SVM-RBF").size());
    }

    @Test
    void kernelMachinesOnCirclesIncludingTheGpClassifier() {
        Points train = ClassicMlShootout.circles(200, 0.08, 0.5, 4);
        Points test = ClassicMlShootout.circles(200, 0.08, 0.5, 5);
        Map<String, Row> r = ClassicMlShootout.kernelMachines(train, test, 2, 6);
        System.out.printf("circles test accuracy: %s%n", line(r, "%.3f"));
        assertTrue(r.get("SVM-RBF").score() > 0.95, r.toString());
        assertTrue(r.get("GP-Laplace").score() > 0.95, r.toString());
        assertTrue(r.get("logistic").score() < 0.75, r.toString());
    }

    @Test
    void clusteringShootout() {
        Points blobs = ClassicMlShootout.blobs(300, 6, 0.3, 7);
        Map<String, Row> b = ClassicMlShootout.clustering(blobs, 6, 0.35, 5, 8);
        System.out.printf("blobs ARI: %s%n", line(b, "%.3f"));
        assertTrue(b.get("k-means++").score() >= b.get("k-means random").score() - 1e-9, b.toString());
        assertTrue(b.get("k-means++").score() > 0.85, b.toString());

        Points aniso = ClassicMlShootout.anisotropic(300, 9);
        Map<String, Row> a = ClassicMlShootout.clustering(aniso, 3, 0.4, 5, 10);
        System.out.printf("anisotropic ARI: %s%n", line(a, "%.3f"));
        assertTrue(a.get("GMM").score() > a.get("k-means++").score() + 0.2, a.toString());

        Points moons = ClassicMlShootout.moons(300, 0.05, 11);
        Map<String, Row> m = ClassicMlShootout.clustering(moons, 2, 0.2, 5, 12);
        System.out.printf("moons ARI: %s%n", line(m, "%.3f"));
        assertTrue(m.get("spectral").score() > 0.9, m.toString());
        assertTrue(m.get("DBSCAN").score() > 0.9, m.toString());
        assertTrue(m.get("k-means++").score() < 0.6, m.toString());

        Points rings = ClassicMlShootout.circles(300, 0.04, 0.4, 13);
        Map<String, Row> c = ClassicMlShootout.clustering(rings, 2, 0.2, 5, 14);
        System.out.printf("rings ARI: %s%n", line(c, "%.3f"));
        assertTrue(c.get("spectral").score() > 0.9, c.toString());
        assertTrue(c.get("GMM").score() < 0.3, c.toString());

        Points noisy = ClassicMlShootout.noisyBlobs(300, 40, 15);
        Map<String, Row> nz = ClassicMlShootout.clustering(noisy, 4, 0.3, 6, 16);
        System.out.printf("noisy blobs ARI: %s%n", line(nz, "%.3f"));
        assertTrue(nz.get("DBSCAN noise recall").score() > 0.7, nz.toString());
        assertTrue(nz.get("DBSCAN").score() > nz.get("k-means++").score(), nz.toString());
    }
}
