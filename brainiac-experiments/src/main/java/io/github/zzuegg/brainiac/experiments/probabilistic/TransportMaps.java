package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.experimental.ml.EntropicTransport;
import io.github.zzuegg.brainiac.experimental.ml.McmcDiagnostics;
import io.github.zzuegg.brainiac.experimental.nn.MinibatchOptimalTransport;
import io.github.zzuegg.brainiac.experimental.nn.NeuralMongeMap;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Continuous optimal transport maps against discrete estimates.
 *
 * <ul>
 *   <li>{@link #gaussian}: a standard normal to a correlated, shifted 2-D Gaussian, whose Monge
 *       map {@code T(x) = m + Sigma^(1/2) x} is known in closed form. Score: the unexplained
 *       variance {@code L2-UVP = 100 E|T_hat(x) - T(x)|^2 / Var(Q)} on held-out source points.</li>
 *   <li>{@link #moonsToRings}: two moons to two concentric rings. Scores: sliced Wasserstein
 *       distance of the mapped held-out points to fresh target samples, and the transport cost
 *       {@code E|T(x) - x|^2}.</li>
 * </ul>
 *
 * <p>Methods: the {@link NeuralMongeMap} (two ICNNs, Makkuva et al. 2020), which maps new
 * points by itself; the barycentric projection of an entropic Sinkhorn coupling; and the exact
 * discrete OT assignment (Hungarian). Both discrete estimates need target samples at
 * evaluation time and are solved directly on the evaluation points.
 */
public final class TransportMaps {

    private TransportMaps() {
    }

    private static Tensor tensor(double[][] x) {
        float[] v = new float[x.length * 2];
        for (int i = 0; i < x.length; i++) {
            v[2 * i] = (float) x[i][0];
            v[2 * i + 1] = (float) x[i][1];
        }
        return Tensor.of(Shape.of(x.length, 2), v);
    }

    private static double[][] points(Tensor t) {
        double[] v = t.toDoubleArray();
        double[][] x = new double[v.length / 2][];
        for (int i = 0; i < x.length; i++) {
            x[i] = new double[] {v[2 * i], v[2 * i + 1]};
        }
        return x;
    }

    private static NeuralMongeMap trainIcnn(Supplier<double[][]> source, Supplier<double[][]> target, int steps,
            long seed) {
        NeuralMongeMap map = new NeuralMongeMap(2, 32, 2, 3e-3f, seed);
        for (int s = 0; s < steps; s++) {
            map.step(tensor(source.get()), tensor(target.get()), 4);
        }
        return map;
    }

    private static double[][] barycentric(double[][] x, double[][] y) {
        return EntropicTransport.barycentricProjection(EntropicTransport.transport(x, y, 0.02, 300), y);
    }

    private static double[][] hungarian(double[][] x, double[][] y) {
        int[] a = MinibatchOptimalTransport.assign(TransportFlows.toFloat(x), TransportFlows.toFloat(y));
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = y[a[i]].clone();
        }
        return out;
    }

    private static double[][] normal(int n, SplittableRandom r) {
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) {
            x[i] = new double[] {r.nextGaussian(), r.nextGaussian()};
        }
        return x;
    }

    /**
     * Gaussian to Gaussian against the closed-form map.
     *
     * @param steps ICNN outer iterations
     * @param seed the seed
     * @return method to {@code {L2-UVP %}}
     */
    public static Map<String, double[]> gaussian(int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] m = {1.0, -0.5};
        double[][] cov = {{2.0, 1.2}, {1.2, 1.0}};
        double s = Math.sqrt(cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
        double t = Math.sqrt(cov[0][0] + cov[1][1] + 2 * s);
        double[][] root = {{(cov[0][0] + s) / t, cov[0][1] / t}, {cov[1][0] / t, (cov[1][1] + s) / t}};
        java.util.function.UnaryOperator<double[]> exact = x -> new double[] {
            m[0] + root[0][0] * x[0] + root[0][1] * x[1], m[1] + root[1][0] * x[0] + root[1][1] * x[1]};
        Supplier<double[][]> target = () -> {
            double[][] z = normal(128, r);
            for (int i = 0; i < z.length; i++) {
                z[i] = exact.apply(z[i]);
            }
            return z;
        };
        double varQ = cov[0][0] + cov[1][1];
        NeuralMongeMap icnn = trainIcnn(() -> normal(128, r), target, steps, seed);
        int n = 512;
        double[][] x = normal(n, r);
        double[][] y = new double[n][];
        for (int i = 0; i < n; i++) {
            y[i] = exact.apply(normal(1, r)[0]);
        }
        Map<String, double[][]> maps = new LinkedHashMap<>();
        maps.put("ICNN Monge map", points(icnn.transport(tensor(x))));
        maps.put("Sinkhorn barycentric", barycentric(x, y));
        maps.put("exact discrete OT", hungarian(x, y));
        Map<String, double[]> out = new LinkedHashMap<>();
        maps.forEach((k, v) -> {
            double e = 0;
            for (int i = 0; i < n; i++) {
                double[] want = exact.apply(x[i]);
                e += (Math.pow(v[i][0] - want[0], 2) + Math.pow(v[i][1] - want[1], 2)) / n;
            }
            out.put(k, new double[] {100 * e / varQ});
        });
        return out;
    }

    /**
     * Moons to rings.
     *
     * @param steps ICNN outer iterations
     * @param seed the seed
     * @return method to {@code {sliced W to the target, transport cost}}
     */
    public static Map<String, double[]> moonsToRings(int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        NeuralMongeMap icnn = trainIcnn(() -> TransportFlows.moons(128, r), () -> TransportFlows.rings(128, r),
                steps, seed);
        int n = 512;
        double[][] x = TransportFlows.moons(n, r);
        double[][] y = TransportFlows.rings(n, r);
        double[][] fresh = TransportFlows.rings(n, r);
        Map<String, double[][]> maps = new LinkedHashMap<>();
        maps.put("ICNN Monge map", points(icnn.transport(tensor(x))));
        maps.put("Sinkhorn barycentric", barycentric(x, y));
        maps.put("exact discrete OT", hungarian(x, y));
        maps.put("identity", x);
        Map<String, double[]> out = new LinkedHashMap<>();
        maps.forEach((k, v) -> {
            double cost = 0;
            for (int i = 0; i < n; i++) {
                cost += (Math.pow(v[i][0] - x[i][0], 2) + Math.pow(v[i][1] - x[i][1], 2)) / n;
            }
            out.put(k, new double[] {McmcDiagnostics.slicedWasserstein(v, fresh, 64, seed), cost});
        });
        return out;
    }
}
