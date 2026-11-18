package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.experimental.ml.GrowingNeuralGas;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.ml.SelfOrganizingMap;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Topology-preserving prototype maps against k-means at an equal number of prototypes: a
 * Kohonen {@link SelfOrganizingMap} on a fixed grid, {@link GrowingNeuralGas} with a learned
 * graph, and {@link KMeans k-means++} (no topology). Data: two concentric rings and a 3-D
 * S-curve. Reported: the quantisation error (mean distance to the nearest prototype) on held-out
 * points, and for the two maps the topographic error (the share of points whose two nearest
 * prototypes are not neighbours in the map).
 */
public final class PrototypeMaps {

    private PrototypeMaps() {
    }

    /**
     * Points on a 3-D S-shaped sheet.
     *
     * @param n the number of points
     * @param seed the seed
     * @return {@code [n][3]}
     */
    public static double[][] sCurve(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) {
            double t = 3 * Math.PI * (r.nextDouble() - 0.5);
            x[i] = new double[] {Math.sin(t), 2 * r.nextDouble(), Math.signum(t) * (Math.cos(t) - 1)};
        }
        return x;
    }

    /**
     * Fits the three methods with {@code side * side} prototypes.
     *
     * @param train training points
     * @param test held-out points
     * @param side the SOM grid side
     * @param seed the seed
     * @return method to {quantisation error, topographic error (NaN for k-means)}
     */
    public static Map<String, double[]> compare(double[][] train, double[][] test, int side, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        int k = side * side;
        KMeans km = KMeans.fit(train, k, KMeans.Init.PLUS_PLUS, 3, 100, seed);
        out.put("k-means++", new double[] {SelfOrganizingMap.meanDistance(km.centres(), test), Double.NaN});
        SelfOrganizingMap som = SelfOrganizingMap.fit(train, side, side, 20, seed);
        out.put("SOM", new double[] {som.quantizationError(test), som.topographicError(test)});
        GrowingNeuralGas gng = GrowingNeuralGas.fit(train, k, 40 * train.length / 2, seed);
        out.put("growing neural gas", new double[] {gng.quantizationError(test), gng.topographicError(test)});
        return out;
    }

    /**
     * Rings and S-curve comparisons.
     *
     * @param seed the seed
     * @return "data / method" to {quantisation error, topographic error}
     */
    public static Map<String, double[]> run(long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        double[][] rings = ClassicMlShootout.circles(600, 0.05, 0.5, seed).x();
        double[][] ringsTest = ClassicMlShootout.circles(300, 0.05, 0.5, seed + 1).x();
        compare(rings, ringsTest, 6, seed).forEach((k, v) -> out.put("rings / " + k, v));
        double[][] s = sCurve(600, seed + 2);
        double[][] sTest = sCurve(300, seed + 3);
        compare(s, sTest, 6, seed).forEach((k, v) -> out.put("S-curve / " + k, v));
        return out;
    }
}
