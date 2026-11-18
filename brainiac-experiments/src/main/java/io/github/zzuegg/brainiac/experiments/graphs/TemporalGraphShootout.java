package io.github.zzuegg.brainiac.experiments.graphs;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.TemporalGraphs;
import io.github.zzuegg.brainiac.experimental.graph.TemporalGraphModels;
import io.github.zzuegg.brainiac.experimental.nn.GcnLayer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Wave 10 item 10.5.12: temporal link prediction on {@link TemporalGraphs} streams, one planted
 * mechanism at a time. The first 70% of each stream trains, the rest is predicted batch by batch
 * (each batch scored from what was observed before it, then revealed). Each positive
 * {@code (u, v, t)} is ranked against random negatives and against historical negatives (past
 * partners of {@code u} other than {@code v}, topped up at random), the protocol of Poursafaei et
 * al. (2022) under which EdgeBank stops looking strong. Ties count half. Models: EdgeBank
 * (unlimited and windowed), TGN, GraphMixer, a HyperEvent-style relational encoder, and a static
 * two-layer GCN on the aggregated training graph.
 */
public final class TemporalGraphShootout {

    /**
     * Sizes.
     *
     * @param nodes nodes
     * @param events events in the stream
     * @param negatives negatives per positive
     * @param epochs training passes for learned models
     * @param batch events per batch
     */
    public record Budget(int nodes, int events, int negatives, int epochs, int batch) {
    }

    private TemporalGraphShootout() {
    }

    /**
     * Runs every model on one stream.
     *
     * @param kind the planted mechanism
     * @param budget sizes
     * @param seed the seed
     * @return per model {@code {MRR random, AUC random, MRR historical, AUC historical, ms}}
     */
    public static Map<String, double[]> compare(TemporalGraphs.Kind kind, Budget budget, long seed) {
        List<TemporalGraphs.Event> stream = TemporalGraphs.generate(kind, budget.nodes(), budget.events(), seed);
        int n = budget.nodes();
        int split = (int) (stream.size() * 0.7);
        int[] src = stream.stream().mapToInt(TemporalGraphs.Event::source).toArray();
        int[] dst = stream.stream().mapToInt(TemporalGraphs.Event::destination).toArray();
        double[] time = stream.stream().mapToDouble(TemporalGraphs.Event::time).toArray();
        int[] trS = Arrays.copyOf(src, split);
        int[] trD = Arrays.copyOf(dst, split);
        double[] trT = Arrays.copyOf(time, split);
        double testSpan = time[time.length - 1] - time[split];
        Map<String, TemporalGraphModels.LinkModel> models = new LinkedHashMap<>();
        models.put("EdgeBank", TemporalGraphModels.edgeBank(Double.POSITIVE_INFINITY));
        models.put("EdgeBank window", TemporalGraphModels.edgeBank(testSpan));
        models.put("TGN", TemporalGraphModels.tgn(n, 24, 10, seed));
        models.put("GraphMixer", TemporalGraphModels.graphMixer(n, 24, 10, seed));
        models.put("HyperEvent", TemporalGraphModels.hyperEvent(n, 24, 8, seed));
        models.put("static GCN", staticGcn(n, seed));
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, TemporalGraphModels.LinkModel> e : models.entrySet()) {
            long t0 = System.currentTimeMillis();
            TemporalGraphModels.LinkModel m = e.getValue();
            m.fit(trS, trD, trT, budget.epochs(), budget.batch(), 1e-2f, seed);
            double[] r = evaluate(m, src, dst, time, split, n, budget, seed + 5);
            out.put(e.getKey(), new double[] {r[0], r[1], r[2], r[3], System.currentTimeMillis() - t0});
        }
        return out;
    }

    private static double[] evaluate(TemporalGraphModels.LinkModel m, int[] src, int[] dst, double[] time, int split,
            int n, Budget budget, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Set<Integer>> past = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            past.add(new HashSet<>());
        }
        for (int i = 0; i < split; i++) {
            past.get(src[i]).add(dst[i]);
        }
        double[] sums = new double[4];
        int count = 0;
        int k = budget.negatives();
        for (int start = split; start < src.length; start += budget.batch()) {
            int end = Math.min(src.length, start + budget.batch());
            int q = end - start;
            int[] s = Arrays.copyOfRange(src, start, end);
            int[][] random = new int[q][k + 1];
            int[][] historical = new int[q][k + 1];
            for (int i = 0; i < q; i++) {
                int v = dst[start + i];
                random[i][0] = v;
                historical[i][0] = v;
                for (int j = 1; j <= k; j++) {
                    random[i][j] = other(v, n, r);
                }
                List<Integer> hist = new ArrayList<>(past.get(s[i]));
                hist.remove(Integer.valueOf(v));
                for (int j = 1; j <= k; j++) {
                    historical[i][j] = hist.isEmpty() ? other(v, n, r) : hist.remove(r.nextInt(hist.size()));
                }
            }
            double[][] a = m.score(s, random, time[start]);
            double[][] b = m.score(s, historical, time[start]);
            for (int i = 0; i < q; i++) {
                double[] ra = rank(a[i]);
                double[] rb = rank(b[i]);
                sums[0] += ra[0];
                sums[1] += ra[1];
                sums[2] += rb[0];
                sums[3] += rb[1];
                count++;
            }
            int[] d = Arrays.copyOfRange(dst, start, end);
            m.observe(s, d, Arrays.copyOfRange(time, start, end));
            for (int i = 0; i < q; i++) {
                past.get(s[i]).add(d[i]);
            }
        }
        for (int i = 0; i < 4; i++) {
            sums[i] /= count;
        }
        return sums;
    }

    private static int other(int v, int n, SplittableRandom r) {
        int x = r.nextInt(n - 1);
        return x >= v ? x + 1 : x;
    }

    /** {reciprocal rank, AUC} of entry 0, ties counted half. */
    private static double[] rank(double[] s) {
        double greater = 0;
        double ties = 0;
        for (int j = 1; j < s.length; j++) {
            greater += s[j] > s[0] ? 1 : 0;
            ties += s[j] == s[0] ? 1 : 0;
        }
        double negatives = s.length - 1;
        return new double[] {1 / (1 + greater + ties / 2), (negatives - greater - ties / 2) / negatives};
    }

    /** A two-layer GCN on the aggregated training graph with a dot-product decoder; it ignores time. */
    private static TemporalGraphModels.LinkModel staticGcn(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Var x = Var.param(Tensor.uniform(Shape.of(n, 16), -0.5f, 0.5f, r));
        GcnLayer g1 = new GcnLayer(16, 16, Activation.RELU, r);
        GcnLayer g2 = new GcnLayer(16, 16, Activation.IDENTITY, r);
        List<Var> params = new ArrayList<>(List.of(x));
        params.addAll(g1.parameters());
        params.addAll(g2.parameters());
        float[][] embedding = new float[1][];
        return new TemporalGraphModels.LinkModel() {
            @Override
            public void reset() {
            }

            @Override
            public double[][] score(int[] sources, int[][] candidates, double time) {
                double[][] s = new double[sources.length][];
                for (int i = 0; i < sources.length; i++) {
                    s[i] = new double[candidates[i].length];
                    for (int j = 0; j < s[i].length; j++) {
                        double dot = 0;
                        for (int c = 0; c < 16; c++) {
                            dot += embedding[0][sources[i] * 16 + c] * embedding[0][candidates[i][j] * 16 + c];
                        }
                        s[i][j] = dot;
                    }
                }
                return s;
            }

            @Override
            public void observe(int[] sources, int[] destinations, double[] times) {
            }

            @Override
            public void fit(int[] sources, int[] destinations, double[] times, int epochs, int batch, float lr,
                    long sampleSeed) {
                float[] a = new float[n * n];
                for (int i = 0; i < sources.length; i++) {
                    a[sources[i] * n + destinations[i]] = 1;
                    a[destinations[i] * n + sources[i]] = 1;
                }
                Var adj = Var.of(Tensor.of(Shape.of(n, n), a));
                Optimizer opt = Optimizer.adam(0.01f);
                SplittableRandom rs = new SplittableRandom(sampleSeed);
                int steps = Math.max(100, epochs * sources.length / batch);
                for (int step = 0; step < steps; step++) {
                    float[] pos = new float[n * n];
                    float[] neg = new float[n * n];
                    for (int i = 0; i < batch; i++) {
                        int e = rs.nextInt(sources.length);
                        pos[sources[e] * n + destinations[e]] += 1;
                        neg[sources[e] * n + rs.nextInt(n)] += 1;
                    }
                    Var z = g2.forward(List.of(g1.forward(List.of(x, adj), true), adj), true);
                    Var logits = z.matmul(z.transpose());
                    Var loss = logits.neg().softplus().mul(Tensor.of(Shape.of(n, n), pos)).sum()
                            .add(logits.softplus().mul(Tensor.of(Shape.of(n, n), neg)).sum()).mul(1f / batch);
                    loss.backward();
                    opt.step(params);
                }
                embedding[0] = Var.noGrad(() -> g2.forward(List.of(g1.forward(List.of(x, adj), false), adj), false))
                        .value().toArray();
            }
        };
    }

    /**
     * Every stream at a larger size.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        Budget budget = new Budget(100, 6000, 20, 12, 100);
        for (TemporalGraphs.Kind kind : TemporalGraphs.Kind.values()) {
            Map<String, double[]> r = compare(kind, budget, 1);
            StringBuilder s = new StringBuilder(kind + " MRR/AUC random, MRR/AUC historical, ms:");
            r.forEach((k, v) -> s.append(String.format("%n  %-16s %.3f %.3f %.3f %.3f %.0f", k, v[0], v[1], v[2], v[3],
                    v[4])));
            System.out.println(s);
        }
    }
}
