package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Timestamped edge streams with one planted mechanism at a time, after the probe design of Hayes
 * et al. (2025, <a href="https://arxiv.org/abs/2510.09416">arXiv:2510.09416</a>), so a temporal
 * link predictor can be checked for what it actually picks up:
 * <ul>
 *   <li>{@link Kind#RECENCY}: a source mostly returns to one of its three most recent partners
 *       (weights 0.6, 0.25, 0.15), else picks a new random destination;</li>
 *   <li>{@link Kind#PERIODIC}: each source has three partners it contacts with fixed, different
 *       periods (with a little jitter), so which partner comes next depends on the time since
 *       each was last seen;</li>
 *   <li>{@link Kind#PREFERENTIAL}: destinations are drawn proportionally to
 *       {@code (degree + 1)^1.2};</li>
 *   <li>{@link Kind#HOMOPHILY}: nodes belong to four hidden communities and 90% of edges stay
 *       inside one;</li>
 *   <li>{@link Kind#REPEATED}: every edge is drawn from a fixed pool of pairs;</li>
 *   <li>{@link Kind#MIXED}: each event uses one of recency, preferential attachment, homophily or
 *       the repeated pool, at random.</li>
 * </ul>
 * Sources are uniform (except in the periodic stream); times grow by exponential gaps of mean 1.
 */
public final class TemporalGraphs {

    /** The planted mechanism. */
    public enum Kind {
        /** Return to recent partners. */
        RECENCY,
        /** Periodic contacts. */
        PERIODIC,
        /** Rich get richer. */
        PREFERENTIAL,
        /** Community structure. */
        HOMOPHILY,
        /** A fixed pool of edges. */
        REPEATED,
        /** All but periodic, mixed per event. */
        MIXED
    }

    /**
     * One interaction.
     *
     * @param source the source node
     * @param destination the destination node
     * @param time the timestamp
     */
    public record Event(int source, int destination, double time) {
    }

    private TemporalGraphs() {
    }

    /**
     * Generates a stream sorted by time.
     *
     * @param kind the mechanism
     * @param nodes the number of nodes
     * @param events about this many events
     * @param seed the seed
     * @return the events
     */
    public static List<Event> generate(Kind kind, int nodes, int events, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        if (kind == Kind.PERIODIC) {
            return periodic(nodes, events, r);
        }
        int[] community = new int[nodes];
        for (int i = 0; i < nodes; i++) {
            community[i] = r.nextInt(4);
        }
        int[][] pool = new int[Math.max(1, nodes * 2)][];
        for (int i = 0; i < pool.length; i++) {
            int u = r.nextInt(nodes);
            pool[i] = new int[] {u, other(u, nodes, r)};
        }
        double[] degree = new double[nodes];
        List<List<Integer>> recent = new ArrayList<>();
        for (int i = 0; i < nodes; i++) {
            recent.add(new ArrayList<>());
        }
        List<Event> out = new ArrayList<>();
        double t = 0;
        Kind[] mixed = {Kind.RECENCY, Kind.PREFERENTIAL, Kind.HOMOPHILY, Kind.REPEATED};
        for (int e = 0; e < events; e++) {
            t += -Math.log(1 - r.nextDouble());
            Kind k = kind == Kind.MIXED ? mixed[r.nextInt(mixed.length)] : kind;
            int u = r.nextInt(nodes);
            int v;
            switch (k) {
                case RECENCY -> {
                    List<Integer> rec = recent.get(u);
                    double x = r.nextDouble();
                    if (rec.size() >= 3 && x < 0.85) {
                        v = rec.get(rec.size() - 1 - (x < 0.6 * 0.85 ? 0 : x < 0.85 * 0.85 ? 1 : 2));
                    } else {
                        v = other(u, nodes, r);
                    }
                }
                case PREFERENTIAL -> {
                    double total = 0;
                    for (int i = 0; i < nodes; i++) {
                        total += i == u ? 0 : Math.pow(degree[i] + 1, 1.2);
                    }
                    double x = r.nextDouble() * total;
                    v = nodes - 1;
                    for (int i = 0; i < nodes; i++) {
                        x -= i == u ? 0 : Math.pow(degree[i] + 1, 1.2);
                        if (x <= 0) {
                            v = i;
                            break;
                        }
                    }
                    if (v == u) {
                        v = other(u, nodes, r);
                    }
                }
                case HOMOPHILY -> {
                    v = other(u, nodes, r);
                    if (r.nextDouble() < 0.9) {
                        for (int tries = 0; tries < 200 && community[v] != community[u]; tries++) {
                            v = other(u, nodes, r);
                        }
                    }
                }
                default -> {
                    int[] p = pool[r.nextInt(pool.length)];
                    u = p[0];
                    v = p[1];
                }
            }
            degree[v]++;
            List<Integer> rec = recent.get(u);
            rec.remove(Integer.valueOf(v));
            rec.add(v);
            out.add(new Event(u, v, t));
        }
        return out;
    }

    private static int other(int u, int nodes, SplittableRandom r) {
        int v = r.nextInt(nodes - 1);
        return v >= u ? v + 1 : v;
    }

    private static List<Event> periodic(int nodes, int events, SplittableRandom r) {
        int[] periods = {6, 10, 15};
        List<int[]> pairs = new ArrayList<>();
        for (int u = 0; u < nodes; u++) {
            for (int p : periods) {
                pairs.add(new int[] {u, other(u, nodes, r), p});
            }
        }
        double rate = 0;
        for (int[] p : pairs) {
            rate += 1.0 / p[2];
        }
        double horizon = events / rate;
        List<Event> out = new ArrayList<>();
        for (int[] p : pairs) {
            double phase = r.nextDouble() * p[2];
            for (double t = phase; t < horizon; t += p[2]) {
                out.add(new Event(p[0], p[1], t + 0.3 * r.nextGaussian()));
            }
        }
        out.sort(Comparator.comparingDouble(Event::time));
        return out;
    }
}
