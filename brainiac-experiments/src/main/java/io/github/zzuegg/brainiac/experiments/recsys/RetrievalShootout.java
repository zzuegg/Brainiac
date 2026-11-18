package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.VectorIndex;
import io.github.zzuegg.brainiac.experimental.data.SyntheticCorpus;
import io.github.zzuegg.brainiac.experimental.nn.TwoTowerRetrieval;
import io.github.zzuegg.brainiac.experimental.recsys.Bm25Index;
import io.github.zzuegg.brainiac.experimental.recsys.DenseRetriever;
import io.github.zzuegg.brainiac.experimental.recsys.LateInteractionRetriever;
import io.github.zzuegg.brainiac.experimental.recsys.RankingLosses;
import io.github.zzuegg.brainiac.experimental.recsys.RankingMetrics;
import io.github.zzuegg.brainiac.experimental.recsys.SparseLexicalEncoder;
import io.github.zzuegg.brainiac.experimental.train.LambdaMart;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Lexical, dense, late-interaction and learned-sparse retrieval, and learning to rank (W7 2.9 BM25
 * and DPR, 2.10 ColBERT, 2.11 SPLADE, 2.13 RankNet / LambdaRank / ListNet / LambdaMART) on a
 * {@link SyntheticCorpus}: queries of held-out documents in three kinds, lexical (a rare key term),
 * paraphrase (synonyms only) and multi-aspect (two terms that only the target holds together).
 * Measured per kind by recall@1, recall@10 and MRR@10 over the whole corpus; reranking by NDCG@5
 * and NDCG@10 against graded labels.
 */
public final class RetrievalShootout {

    /**
     * Retrieval metrics.
     *
     * @param recall1 recall@1
     * @param recall10 recall@10
     * @param mrr10 MRR@10
     */
    public record Metrics(double recall1, double recall10, double mrr10) {
    }

    /**
     * Transformer blocks of the neural encoders. On this small corpus, trained from scratch at
     * this budget, a bag of token embeddings with positions (0 blocks) learns the synonyms far
     * faster than one transformer block (paraphrase MRR@10 0.71 against 0.25 in a side run).
     */
    private static final int LAYERS = 0;
    private static final float LR = 0.01f;

    private final SyntheticCorpus corpus;
    private final long seed;
    private final Bm25Index bm25 = Bm25Index.bm25(1.2, 0.75);
    private final Bm25Index tfIdf = Bm25Index.tfIdf();
    private final int[][] docs;
    private final List<int[]> hardNegative = new ArrayList<>();
    private DenseRetriever dense;
    private VectorIndex denseIndex;

    /**
     * Creates the corpus and the lexical indexes.
     *
     * @param seed the seed
     */
    public RetrievalShootout(long seed) {
        this.seed = seed;
        this.corpus = SyntheticCorpus.generate(6, 10, 30, 0.6, seed);
        this.docs = corpus.documents();
        for (int[] d : docs) {
            bm25.add(d);
            tfIdf.add(d);
        }
        for (SyntheticCorpus.Query q : corpus.trainQueries()) {
            int[] top = bm25.search(q.tokens(), 2);
            hardNegative.add(docs[top[0] == q.target() ? top[1] : top[0]]);
        }
    }

    /** @return the corpus */
    public SyntheticCorpus corpus() {
        return corpus;
    }

    /**
     * Measures a scorer per query kind.
     *
     * @param scorer query tokens to one score per document
     * @return kind to metrics
     */
    public Map<SyntheticCorpus.Kind, Metrics> evaluate(Function<int[], float[]> scorer) {
        return evaluateRanked(q -> RankingMetrics.topK(scorer.apply(q), 10, null));
    }

    /**
     * Measures a ranker per query kind.
     *
     * @param ranker query tokens to the top 10 documents
     * @return kind to metrics
     */
    public Map<SyntheticCorpus.Kind, Metrics> evaluateRanked(Function<int[], int[]> ranker) {
        Map<SyntheticCorpus.Kind, Metrics> out = new EnumMap<>(SyntheticCorpus.Kind.class);
        for (SyntheticCorpus.Kind kind : SyntheticCorpus.Kind.values()) {
            List<SyntheticCorpus.Query> qs = corpus.testQueries(kind);
            double r1 = 0;
            double r10 = 0;
            double mrr = 0;
            for (SyntheticCorpus.Query q : qs) {
                int[] top = ranker.apply(q.tokens());
                r1 += RankingMetrics.hit(top, q.target(), 1);
                r10 += RankingMetrics.hit(top, q.target(), 10);
                mrr += RankingMetrics.reciprocalRank(top, q.target(), 10);
            }
            int n = Math.max(1, qs.size());
            out.put(kind, new Metrics(r1 / n, r10 / n, mrr / n));
        }
        return out;
    }

    private interface PairStep {
        void step(int[][] queries, int[][] positives, int[][] negatives);
    }

    private void train(int steps, boolean hard, PairStep step) {
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<SyntheticCorpus.Query> train = corpus.trainQueries();
        int batch = 32;
        for (int s = 0; s < steps; s++) {
            int[][] q = new int[batch][];
            int[][] p = new int[batch][];
            int[][] n = new int[batch][];
            Set<Integer> used = new HashSet<>();
            for (int b = 0; b < batch; b++) {
                int i;
                do {
                    i = r.nextInt(train.size());
                } while (!used.add(train.get(i).target()));
                q[b] = train.get(i).tokens();
                p[b] = docs[train.get(i).target()];
                n[b] = hardNegative.get(i);
            }
            step.step(q, p, hard ? n : null);
        }
    }

    /**
     * Trains (once) and returns the dense retriever.
     *
     * @param steps InfoNCE steps of 32 queries, with BM25 hard negatives
     * @return the retriever
     */
    public DenseRetriever dense(int steps) {
        if (dense == null) {
            dense = new DenseRetriever(corpus.vocabulary(), SyntheticCorpus.DOC_LENGTH, 32, 2, LAYERS, true, seed)
                    .optimizer(Optimizer.adam(LR));
            train(steps, true, dense::trainStep);
            denseIndex = dense.index(docs);
        }
        return dense;
    }

    private float[] denseScores(int[] q) {
        return dense.scores(denseIndex, q);
    }

    private float[] bowRow(int[] tokens) {
        float[] v = new float[corpus.vocabulary()];
        for (int t : tokens) {
            if (t != 0) {
                v[t] = 1;
            }
        }
        return v;
    }

    private Tensor bow(int[][] rows) {
        int v = corpus.vocabulary();
        float[] out = new float[rows.length * v];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(bowRow(rows[i]), 0, out, i * v, v);
        }
        return Tensor.of(Shape.of(rows.length, v), out);
    }

    /**
     * Lexical against dense retrieval.
     *
     * @param steps dense training steps
     * @return retriever name to per-kind metrics
     */
    public Map<String, Map<SyntheticCorpus.Kind, Metrics>> lexicalVersusDense(int steps) {
        Map<String, Map<SyntheticCorpus.Kind, Metrics>> out = new LinkedHashMap<>();
        out.put("TF-IDF", evaluate(tfIdf::scores));
        out.put("BM25", evaluate(bm25::scores));
        DenseRetriever untrained = new DenseRetriever(corpus.vocabulary(), SyntheticCorpus.DOC_LENGTH, 32, 2, LAYERS,
                true, seed);
        VectorIndex ui = untrained.index(docs);
        out.put("untrained encoder", evaluate(q -> untrained.scores(ui, q)));
        dense(steps);
        out.put("DPR", evaluate(this::denseScores));
        DenseRetriever inBatch = new DenseRetriever(corpus.vocabulary(), SyntheticCorpus.DOC_LENGTH, 32, 2, LAYERS,
                true, seed).optimizer(Optimizer.adam(LR));
        train(steps, false, inBatch::trainStep);
        VectorIndex ib = inBatch.index(docs);
        out.put("DPR (in-batch negatives only)", evaluate(q -> inBatch.scores(ib, q)));
        out.put("hybrid RRF", evaluateRanked(q -> {
            int[] a = RankingMetrics.topK(bm25.scores(q), docs.length, null);
            int[] b = RankingMetrics.topK(denseScores(q), docs.length, null);
            float[] fused = new float[docs.length];
            for (int i = 0; i < a.length; i++) {
                fused[a[i]] += 1f / (60 + i + 1);
                fused[b[i]] += 1f / (60 + i + 1);
            }
            return RankingMetrics.topK(fused, 10, null);
        }));
        int v = corpus.vocabulary();
        Model qt = Model.sequential().input(v).dense(64, Activation.RELU).dense(32).seed(seed).build();
        Model dt = Model.sequential().input(v).dense(64, Activation.RELU).dense(32).seed(seed + 1).build();
        TwoTowerRetrieval tt = TwoTowerRetrieval.of(qt, dt).temperature(0.1f);
        train(steps, false, (q, p, n) -> tt.trainStep(bow(q), bow(p)));
        Tensor docEmb = tt.embedItems(bow(docs));
        out.put("two-tower (bag of words)", evaluate(q -> {
            Tensor e = tt.embedUsers(bow(new int[][] {q}));
            return e.matmul(docEmb.transpose()).toArray();
        }));
        return out;
    }

    /**
     * ColBERT late interaction against DPR and BM25.
     *
     * @param steps training steps of each neural retriever
     * @param centroids centroids of the compressed index
     * @return retriever name to per-kind metrics; {@code bytes} maps the ColBERT variants to their
     *     bytes per document
     */
    public Map<String, Map<SyntheticCorpus.Kind, Metrics>> lateInteraction(int steps, int centroids,
            Map<String, Double> bytes) {
        Map<String, Map<SyntheticCorpus.Kind, Metrics>> out = new LinkedHashMap<>();
        out.put("BM25", evaluate(bm25::scores));
        dense(steps);
        out.put("DPR", evaluate(this::denseScores));
        bytes.put("DPR", 32 * 4.0);
        LateInteractionRetriever colbert = new LateInteractionRetriever(corpus.vocabulary(), SyntheticCorpus.DOC_LENGTH,
                32, 16, 2, LAYERS, seed).optimizer(Optimizer.adam(LR));
        train(steps, true, colbert::trainStep);
        colbert.index(docs, 0, seed);
        out.put("ColBERT fp32", evaluate(colbert::scores));
        bytes.put("ColBERT fp32", colbert.bytesPerDocument());
        out.put("ColBERT fp32 (kNN candidates)", evaluateRanked(q -> colbert.search(q, 10, 8)));
        colbert.index(docs, centroids, seed);
        out.put("ColBERT 2-bit residual", evaluate(colbert::scores));
        bytes.put("ColBERT 2-bit residual", colbert.bytesPerDocument());
        return out;
    }

    /**
     * SPLADE at several sparsity weights against BM25 and DPR.
     *
     * @param steps training steps
     * @param flopsWeights FLOPS regulariser weights to sweep
     * @param nonZeros filled with the average non-zero terms per document of each SPLADE run
     * @return retriever name to per-kind metrics
     */
    public Map<String, Map<SyntheticCorpus.Kind, Metrics>> sparse(int steps, float[] flopsWeights,
            Map<String, Double> nonZeros) {
        Map<String, Map<SyntheticCorpus.Kind, Metrics>> out = new LinkedHashMap<>();
        out.put("BM25", evaluate(bm25::scores));
        nonZeros.put("BM25", bm25.averageTermsPerDocument());
        dense(steps);
        out.put("DPR", evaluate(this::denseScores));
        for (float w : flopsWeights) {
            SparseLexicalEncoder splade = new SparseLexicalEncoder(corpus.vocabulary(), SyntheticCorpus.DOC_LENGTH, 32,
                    2, LAYERS, w, seed).optimizer(Optimizer.adam(LR));
            train(steps, true, splade::trainStep);
            Bm25Index index = splade.index(docs);
            String name = "SPLADE flops=" + w;
            out.put(name, evaluate(q -> splade.scores(index, q)));
            nonZeros.put(name, index.averageTermsPerDocument());
        }
        return out;
    }

    // ---------------------------------------------------------------- reranking (learning to rank)

    private static final int CANDIDATES = 20;

    private record Lists(double[][][] features, int[][] grades) {
    }

    private Lists lists(List<SyntheticCorpus.Query> queries) {
        Map<Integer, Integer> source = new HashMap<>();
        for (int w = 0; w < corpus.vocabulary(); w++) {
            if (corpus.paraphrase(w) != w) {
                source.put(corpus.paraphrase(w), w);
            }
        }
        double[][][] f = new double[queries.size()][CANDIDATES][];
        int[][] g = new int[queries.size()][CANDIDATES];
        for (int i = 0; i < queries.size(); i++) {
            SyntheticCorpus.Query q = queries.get(i);
            float[] lex = bm25.scores(q.tokens());
            float[] tf = tfIdf.scores(q.tokens());
            float[] den = denseScores(q.tokens());
            int[] a = RankingMetrics.topK(lex, docs.length, null);
            int[] b = RankingMetrics.topK(den, docs.length, null);
            float[] fused = new float[docs.length];
            for (int r = 0; r < a.length; r++) {
                fused[a[r]] += 1f / (61 + r);
                fused[b[r]] += 1f / (61 + r);
            }
            int[] cand = RankingMetrics.topK(fused, CANDIDATES, null);
            for (int c = 0; c < CANDIDATES; c++) {
                int d = cand[c];
                Set<Integer> words = new HashSet<>();
                for (int t : docs[d]) {
                    words.add(t);
                }
                int overlap = 0;
                int topical = 0;
                for (int t : q.tokens()) {
                    if (t != 0 && words.contains(t)) {
                        overlap++;
                    }
                    if (source.containsKey(t) && corpus.topic(d) == corpus.topic(cand[0])) {
                        topical++;
                    }
                }
                f[i][c] = new double[] {lex[d], tf[d], den[d], overlap, topical, 1.0 / (c + 1)};
                g[i][c] = corpus.grade(q, d);
            }
        }
        return new Lists(f, g);
    }

    private static double ndcg(Lists l, Function<double[], Double> scorer, int k) {
        double s = 0;
        for (int i = 0; i < l.grades().length; i++) {
            float[] sc = new float[CANDIDATES];
            for (int c = 0; c < CANDIDATES; c++) {
                sc[c] = scorer.apply(l.features()[i][c]).floatValue();
            }
            s += RankingMetrics.ndcg(sc, l.grades()[i], k);
        }
        return s / l.grades().length;
    }

    /**
     * Reranking the top 20 of the BM25 + DPR hybrid with learned scorers over six features (BM25,
     * TF-IDF, dense score, exact overlap, paraphrase terms when the candidate shares the top
     * candidate's topic, hybrid rank).
     *
     * @param steps gradient steps of the neural scorers (and the dense retriever)
     * @param trees LambdaMART and GBT trees
     * @return scorer name to {@code {NDCG@5, NDCG@10}}
     */
    public Map<String, double[]> reranking(int steps, int trees) {
        dense(steps);
        Lists train = lists(corpus.trainQueries());
        Lists test = lists(corpus.testQueries());
        int nf = train.features()[0][0].length;
        double[] mean = new double[nf];
        double[] sd = new double[nf];
        int rows = train.grades().length * CANDIDATES;
        for (double[][] q : train.features()) {
            for (double[] x : q) {
                for (int j = 0; j < nf; j++) {
                    mean[j] += x[j] / rows;
                }
            }
        }
        for (double[][] q : train.features()) {
            for (double[] x : q) {
                for (int j = 0; j < nf; j++) {
                    sd[j] += (x[j] - mean[j]) * (x[j] - mean[j]) / rows;
                }
            }
        }
        for (int j = 0; j < nf; j++) {
            sd[j] = Math.sqrt(sd[j]) + 1e-9;
        }
        Function<double[], double[]> norm = x -> {
            double[] o = new double[nf];
            for (int j = 0; j < nf; j++) {
                o[j] = (x[j] - mean[j]) / sd[j];
            }
            return o;
        };
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("hybrid order", new double[] {ndcg(test, x -> x[5], 5), ndcg(test, x -> x[5], 10)});
        double[][] flat = new double[rows][];
        double[] y = new double[rows];
        int[] grades = new int[rows];
        int[] qid = new int[rows];
        for (int i = 0; i < train.grades().length; i++) {
            for (int c = 0; c < CANDIDATES; c++) {
                flat[i * CANDIDATES + c] = train.features()[i][c];
                y[i * CANDIDATES + c] = train.grades()[i][c];
                grades[i * CANDIDATES + c] = train.grades()[i][c];
                qid[i * CANDIDATES + c] = i;
            }
        }
        GradientBoostedTrees gbt = GradientBoostedTrees.fit(flat, y, GradientBoostedTrees.Objective.REGRESSION,
                GradientBoostedTrees.Settings.defaults().withTrees(trees).withDepth(3));
        out.put("GBT pointwise", new double[] {ndcg(test, gbt::predict, 5), ndcg(test, gbt::predict, 10)});
        LambdaMart mart = LambdaMart.fit(flat, grades, qid,
                new LambdaMart.Settings(trees, 3, 0.1, 5, 32, CANDIDATES));
        out.put("LambdaMART", new double[] {ndcg(test, mart::predict, 5), ndcg(test, mart::predict, 10)});
        Map<String, Loss> losses = new LinkedHashMap<>();
        losses.put("MLP pointwise", Loss.MSE);
        losses.put("RankNet", RankingLosses.rankNet());
        losses.put("LambdaRank", RankingLosses.lambdaRank(CANDIDATES));
        losses.put("ListNet", RankingLosses.listNet());
        for (Map.Entry<String, Loss> e : losses.entrySet()) {
            Model mlp = Model.sequential().input(nf).dense(32, Activation.RELU).dense(1).seed(seed).build();
            Optimizer opt = Optimizer.adam(3e-3f);
            SplittableRandom r = new SplittableRandom(seed + 4);
            int lists = 16;
            for (int s = 0; s < steps; s++) {
                float[] x = new float[lists * CANDIDATES * nf];
                float[] g = new float[lists * CANDIDATES];
                for (int b = 0; b < lists; b++) {
                    int q = r.nextInt(train.grades().length);
                    for (int c = 0; c < CANDIDATES; c++) {
                        double[] v = norm.apply(train.features()[q][c]);
                        for (int j = 0; j < nf; j++) {
                            x[(b * CANDIDATES + c) * nf + j] = (float) v[j];
                        }
                        g[b * CANDIDATES + c] = train.grades()[q][c];
                    }
                }
                Var scores = mlp.forward(Var.of(Tensor.of(Shape.of(lists * CANDIDATES, nf), x)), true)
                        .reshape(lists, CANDIDATES);
                e.getValue().compute(scores, Tensor.of(Shape.of(lists, CANDIDATES), g)).backward();
                opt.step(mlp.parameters());
            }
            Function<double[], Double> score = f -> {
                double[] v = norm.apply(f);
                float[] fx = new float[nf];
                for (int j = 0; j < nf; j++) {
                    fx[j] = (float) v[j];
                }
                return (double) Var.noGrad(() -> mlp.forward(Var.of(Tensor.of(Shape.of(1, nf), fx)), false)).value()
                        .item();
            };
            out.put(e.getKey(), new double[] {ndcg(test, score, 5), ndcg(test, score, 10)});
        }
        return out;
    }
}
