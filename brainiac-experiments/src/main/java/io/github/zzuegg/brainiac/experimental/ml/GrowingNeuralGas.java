package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Growing neural gas (Fritzke 1995, "A growing neural gas network learns topologies"): a graph
 * of prototypes that starts with two units and grows. Each sample moves its nearest unit and,
 * less, that unit's graph neighbours towards it, connects the two nearest units (competitive
 * Hebbian learning), ages and prunes old edges, and every {@code lambda} samples a new unit is
 * inserted between the unit with the largest accumulated error and its worst neighbour. The
 * edges end up tracing the data manifold, whatever its shape or dimension.
 */
public final class GrowingNeuralGas {

    private final List<double[]> units;
    private final List<int[]> edges;

    private GrowingNeuralGas(List<double[]> units, List<int[]> edges) {
        this.units = units;
        this.edges = edges;
    }

    /**
     * Grows a network.
     *
     * @param x the data {@code [n][d]}
     * @param maxUnits the most units
     * @param samples training samples drawn (with replacement)
     * @param seed the seed
     * @return the network
     */
    public static GrowingNeuralGas fit(double[][] x, int maxUnits, int samples, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double eb = 0.05;
        double en = 0.006;
        int maxAge = 50;
        int lambda = Math.max(10, samples / (2 * maxUnits));
        double alpha = 0.5;
        double decay = 0.995;
        List<double[]> w = new ArrayList<>();
        List<Double> error = new ArrayList<>();
        w.add(x[r.nextInt(x.length)].clone());
        w.add(x[r.nextInt(x.length)].clone());
        error.add(0.0);
        error.add(0.0);
        // edges as [a, b, age] with a < b
        List<int[]> edges = new ArrayList<>();
        for (int step = 1; step <= samples; step++) {
            double[] p = x[r.nextInt(x.length)];
            int[] two = SelfOrganizingMap.twoNearest(w.toArray(new double[0][]), p);
            int s1 = two[0];
            int s2 = two[1];
            for (int[] e : edges) {
                if (e[0] == s1 || e[1] == s1) {
                    e[2]++;
                }
            }
            error.set(s1, error.get(s1) + MlMatrices.squaredDistance(w.get(s1), p));
            move(w.get(s1), p, eb);
            for (int[] e : edges) {
                if (e[0] == s1) {
                    move(w.get(e[1]), p, en);
                } else if (e[1] == s1) {
                    move(w.get(e[0]), p, en);
                }
            }
            int[] existing = find(edges, s1, s2);
            if (existing != null) {
                existing[2] = 0;
            } else {
                edges.add(new int[] {Math.min(s1, s2), Math.max(s1, s2), 0});
            }
            edges.removeIf(e -> e[2] > maxAge);
            removeIsolated(w, error, edges);
            if (step % lambda == 0 && w.size() < maxUnits) {
                int q = 0;
                for (int u = 1; u < w.size(); u++) {
                    if (error.get(u) > error.get(q)) {
                        q = u;
                    }
                }
                int f = -1;
                for (int[] e : edges) {
                    int other = e[0] == q ? e[1] : e[1] == q ? e[0] : -1;
                    if (other >= 0 && (f < 0 || error.get(other) > error.get(f))) {
                        f = other;
                    }
                }
                if (f >= 0) {
                    double[] mid = new double[p.length];
                    for (int t = 0; t < mid.length; t++) {
                        mid[t] = 0.5 * (w.get(q)[t] + w.get(f)[t]);
                    }
                    w.add(mid);
                    int nu = w.size() - 1;
                    edges.remove(find(edges, q, f));
                    edges.add(new int[] {q, nu, 0});
                    edges.add(new int[] {f, nu, 0});
                    error.set(q, error.get(q) * alpha);
                    error.set(f, error.get(f) * alpha);
                    error.add(error.get(q));
                }
            }
            for (int u = 0; u < error.size(); u++) {
                error.set(u, error.get(u) * decay);
            }
        }
        return new GrowingNeuralGas(w, edges);
    }

    private static void move(double[] w, double[] p, double rate) {
        for (int t = 0; t < w.length; t++) {
            w[t] += rate * (p[t] - w[t]);
        }
    }

    private static int[] find(List<int[]> edges, int a, int b) {
        int lo = Math.min(a, b);
        int hi = Math.max(a, b);
        for (int[] e : edges) {
            if (e[0] == lo && e[1] == hi) {
                return e;
            }
        }
        return null;
    }

    private static void removeIsolated(List<double[]> w, List<Double> error, List<int[]> edges) {
        if (w.size() <= 2) {
            return;
        }
        for (int u = w.size() - 1; u >= 0; u--) {
            boolean connected = false;
            for (int[] e : edges) {
                if (e[0] == u || e[1] == u) {
                    connected = true;
                    break;
                }
            }
            if (!connected) {
                w.remove(u);
                error.remove(u);
                for (int[] e : edges) {
                    if (e[0] > u) {
                        e[0]--;
                    }
                    if (e[1] > u) {
                        e[1]--;
                    }
                }
            }
        }
    }

    /**
     * Returns the prototypes.
     *
     * @return {@code [units][d]} (copies)
     */
    public double[][] prototypes() {
        double[][] c = new double[units.size()][];
        for (int i = 0; i < c.length; i++) {
            c[i] = units.get(i).clone();
        }
        return c;
    }

    /** @return the number of edges */
    public int edgeCount() {
        return edges.size();
    }

    /**
     * The mean distance of samples to their nearest unit.
     *
     * @param x samples
     * @return the quantisation error
     */
    public double quantizationError(double[][] x) {
        return SelfOrganizingMap.meanDistance(prototypes(), x);
    }

    /**
     * The share of samples whose two nearest units are not joined by an edge.
     *
     * @param x samples
     * @return the topographic error in {@code [0, 1]}
     */
    public double topographicError(double[][] x) {
        double[][] p = prototypes();
        int bad = 0;
        for (double[] s : x) {
            int[] two = SelfOrganizingMap.twoNearest(p, s);
            bad += find(edges, two[0], two[1]) == null ? 1 : 0;
        }
        return bad / (double) x.length;
    }
}
