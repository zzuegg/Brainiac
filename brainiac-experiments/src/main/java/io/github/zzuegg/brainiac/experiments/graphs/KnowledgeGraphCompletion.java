package io.github.zzuegg.brainiac.experiments.graphs;

import io.github.zzuegg.brainiac.experimental.data.FamilyTreeGraph;
import io.github.zzuegg.brainiac.experimental.graph.KnowledgeGraphEmbedding;
import io.github.zzuegg.brainiac.experimental.nn.TwoTowerRetrieval;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Knowledge-graph completion on a synthetic family tree (wave 7, 7.1.13): TransE,
 * DistMult, ComplEx and RotatE ({@link KnowledgeGraphEmbedding}) against a rule-free
 * frequency baseline (rank tails by how often they are a tail of the relation) and a
 * relation-aware two-tower model ({@link TwoTowerRetrieval}, query tower on
 * {@code [head || relation]}, item tower on the tail). The metric is the filtered mean
 * reciprocal rank and Hits@10 of the tail, per relation pattern ({@link FamilyTreeGraph}:
 * symmetric sibling, antisymmetric parent, inverse child, composed grandparent).
 */
public final class KnowledgeGraphCompletion {

    private KnowledgeGraphCompletion() {
    }

    /** Filtered tail ranks from a score function. */
    private interface Scorer {
        float[] tails(int head, int relation);
    }

    private static void evaluate(String name, Scorer scorer, FamilyTreeGraph.Data data, Set<Long> known,
            Map<String, Double> out) {
        double[] rr = new double[4];
        double[] hits = new double[4];
        int[] count = new int[4];
        double rrAll = 0;
        for (int[] t : data.test()) {
            float[] s = scorer.tails(t[0], t[1]);
            int rank = 1;
            for (int e = 0; e < s.length; e++) {
                if (e != t[2] && s[e] > s[t[2]] && !known.contains(KnowledgeGraphEmbedding.key(t[0], t[1], e))) {
                    rank++;
                }
            }
            rr[t[1]] += 1.0 / rank;
            hits[t[1]] += rank <= 10 ? 1 : 0;
            count[t[1]]++;
            rrAll += 1.0 / rank;
        }
        out.put(name + " MRR", GraphTraining.round(rrAll / data.test().size()));
        for (int r = 0; r < 4; r++) {
            if (count[r] > 0) {
                out.put(name + " MRR " + FamilyTreeGraph.RELATIONS.get(r), GraphTraining.round(rr[r] / count[r]));
                out.put(name + " Hits@10 " + FamilyTreeGraph.RELATIONS.get(r), GraphTraining.round(hits[r] / count[r]));
            }
        }
    }

    /**
     * Trains and evaluates every model.
     *
     * @param models the KGE models to train
     * @param families families in the knowledge graph
     * @param steps training steps (batches of 64 triples)
     * @param seed the seed
     * @return "model MRR", "model MRR relation" and "model Hits@10 relation"
     */
    public static Map<String, Double> compare(List<KnowledgeGraphEmbedding.Model> models, int families, int steps,
            long seed) {
        FamilyTreeGraph.Data data = FamilyTreeGraph.generate(families, 0.25, seed);
        Set<Long> known = new HashSet<>();
        for (int[] t : data.all()) {
            known.add(KnowledgeGraphEmbedding.key(t[0], t[1], t[2]));
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("entities", (double) data.entities());
        out.put("test triples", (double) data.test().size());
        int n = data.entities();
        float[][] freq = new float[4][n];
        for (int[] t : data.train()) {
            freq[t[1]][t[2]]++;
        }
        evaluate("frequency", (h, r) -> freq[r], data, known, out);
        evaluate("two-tower", twoTower(data, steps, seed), data, known, out);
        for (KnowledgeGraphEmbedding.Model model : models) {
            SplittableRandom r = new SplittableRandom(seed);
            KnowledgeGraphEmbedding kge = new KnowledgeGraphEmbedding(model, n, 4, 16, 6f, r);
            Optimizer opt = Optimizer.adam(0.02f);
            for (int s = 0; s < steps; s++) {
                List<int[]> batch = new ArrayList<>();
                for (int i = 0; i < 64; i++) {
                    batch.add(data.train().get(r.nextInt(data.train().size())));
                }
                kge.loss(batch, 16, 1f, r).backward();
                opt.step(kge.parameters());
            }
            evaluate(model.name(), kge::scoreTails, data, known, out);
        }
        return out;
    }

    private static Scorer twoTower(FamilyTreeGraph.Data data, int steps, long seed) {
        int n = data.entities();
        Model query = Model.sequential().input(n + 4).dense(32, Activation.RELU).dense(16).seed(seed).build();
        Model item = Model.sequential().input(n).dense(32, Activation.RELU).dense(16).seed(seed + 1).build();
        TwoTowerRetrieval tt = TwoTowerRetrieval.of(query, item).temperature(0.1f).optimizer(Optimizer.adam(0.01f));
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int b = 32;
            float[] q = new float[b * (n + 4)];
            float[] it = new float[b * n];
            for (int i = 0; i < b; i++) {
                int[] t = data.train().get(r.nextInt(data.train().size()));
                q[i * (n + 4) + t[0]] = 1;
                q[i * (n + 4) + n + t[1]] = 1;
                it[i * n + t[2]] = 1;
            }
            tt.trainStep(Tensor.of(Shape.of(b, n + 4), q), Tensor.of(Shape.of(b, n), it));
        }
        float[] eye = new float[n * n];
        for (int i = 0; i < n; i++) {
            eye[i * n + i] = 1;
        }
        Tensor items = tt.embedItems(Tensor.of(Shape.of(n, n), eye));
        return (h, rel) -> {
            float[] q = new float[n + 4];
            q[h] = 1;
            q[n + rel] = 1;
            Tensor u = tt.embedUsers(Tensor.of(Shape.of(1, n + 4), q));
            return items.matmul(u.transpose()).reshape(n).toArray();
        };
    }
}
