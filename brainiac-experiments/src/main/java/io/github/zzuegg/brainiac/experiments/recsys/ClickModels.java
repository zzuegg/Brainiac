package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ClickLogs;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.nn.CrossNetwork;
import io.github.zzuegg.brainiac.experimental.nn.FactorizationMachine;
import io.github.zzuegg.brainiac.experimental.nn.FeatureTokenizer;
import io.github.zzuegg.brainiac.experimental.nn.FtTransformer;
import io.github.zzuegg.brainiac.experimental.recsys.CompactEmbedding;
import io.github.zzuegg.brainiac.experimental.recsys.CompressedInteraction;
import io.github.zzuegg.brainiac.experimental.recsys.FieldAttention;
import io.github.zzuegg.brainiac.experimental.recsys.RankMixerBlock;
import io.github.zzuegg.brainiac.experimental.recsys.WideAndDeep;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.BiFunction;

/**
 * Deep click-through-rate models (W7 2.4: Wide &amp; Deep, DeepFM) against logistic regression,
 * the factorization machine and DCN-V2 on {@link SyntheticRecsys#clicks}, where a hidden
 * interaction between two categorical fields drives clicks. Every model gets the same Adam
 * budget; the metrics are test accuracy, ROC-AUC and log-loss. NeuMF, the third model of the
 * item, ranks items and is measured in {@link CollaborativeFiltering#neuMf}.
 * <p>
 * Wave 9 adds three studies:
 * <ul>
 *   <li>{@link #interactionOrder}: AutoInt ({@link FieldAttention}), xDeepFM
 *       ({@link CompressedInteraction}) and RankMixer against FM, Wide &amp; Deep, DeepFM, DCN-V2
 *       and the FT-Transformer on {@link ClickLogs#interactionClicks} with a planted interaction
 *       of order 2, 3 or 4, so the question is which architecture recovers high-order crosses;</li>
 *   <li>{@link #rankMixerSweep}: dense and sparse-MoE {@link RankMixerBlock} rankers against an
 *       FT-Transformer of the same token width, reporting parameters, AUC and training time per
 *       batch (the scaling claim of the RankMixer paper);</li>
 *   <li>{@link #memorySweep}: the {@link CompactEmbedding} layouts on a 100k-ID Zipf field at a
 *       range of memory budgets, with AUC overall and on tail IDs.</li>
 * </ul>
 */
public final class ClickModels {

    private ClickModels() {
    }

    /**
     * @param steps gradient steps per model
     * @param seed the seed
     * @return model name to test metrics
     */
    public static Map<String, BinaryBench.Result> run(int steps, long seed) {
        int fields = 3;
        int cats = 8;
        SyntheticRecsys.Table all = SyntheticRecsys.clicks(6000, fields, cats, seed);
        SyntheticRecsys.Table train = all.rows(0, 4000);
        SyntheticRecsys.Table test = all.rows(4000, 6000);
        int cols = fields * cats;
        Map<String, BinaryBench.Result> out = new LinkedHashMap<>();
        Model logistic = Model.sequential().input(cols).dense(1).seed(seed).build();
        out.put("logistic", BinaryBench.train(logistic::forward, logistic.parameters(), train, test, steps, 0.03f,
                seed));
        FactorizationMachine fm = new FactorizationMachine(cols, 8, new SplittableRandom(seed));
        out.put("FM", BinaryBench.train(fm::forward, fm.parameters(), train, test, steps, 0.03f, seed));
        CrossNetwork cross = new CrossNetwork(cols, 2, new SplittableRandom(seed));
        Model head = Model.sequential().input(cols).dense(1).seed(seed).build();
        List<Var> dcn = new ArrayList<>(cross.parameters());
        dcn.addAll(head.parameters());
        out.put("DCN-V2", BinaryBench.train((x, t) -> head.forward(cross.forward(x, t), t), dcn, train, test, steps,
                0.01f, seed));
        WideAndDeep wd = WideAndDeep.wideDeep(fields, cats, 8, 32, seed);
        out.put("Wide&Deep", BinaryBench.train(wd::forward, wd.parameters(), train, test, steps, 0.01f, seed));
        WideAndDeep dfm = WideAndDeep.deepFm(fields, cats, 8, 32, seed);
        out.put("DeepFM", BinaryBench.train(dfm::forward, dfm.parameters(), train, test, steps, 0.01f, seed));
        return out;
    }

    private static long count(List<Var> params) {
        long s = 0;
        for (Var v : params) {
            s += v.shape().size();
        }
        return s;
    }

    /**
     * An FT-Transformer with one categorical token per field plus [CLS], one block, two heads (the
     * tokenizer needs a numeric column, so a constant zero column adds one uninformative token).
     */
    private static FtTransformer categoricalFt(int fields, int cats, int dim, SplittableRandom r) {
        int[] cards = new int[fields];
        java.util.Arrays.fill(cards, cats);
        return new FtTransformer(new FeatureTokenizer(1, cards, dim, FeatureTokenizer.Numeric.LINEAR, 0, 0, true, r),
                1, 2, 1, r);
    }

    /** Feeds one-hot rows to a categorical FT-Transformer as {@code [batch, fields]} category ids. */
    private static BiFunction<Var, Boolean, Var> fieldIds(FtTransformer ft, int fields, int cats) {
        float[] idx = new float[cats];
        for (int c = 0; c < cats; c++) {
            idx[c] = c;
        }
        Tensor index = Tensor.of(Shape.of(cats, 1), idx);
        return (x, t) -> {
            int b = x.shape().dim(0);
            Tensor ids = x.value().reshape(b, fields, cats).matmul(index).reshape(b, fields);
            return ft.forward(Var.of(Tensor.concat(1, Tensor.zeros(Shape.of(b, 1)), ids)), t);
        };
    }

    private static void put(Map<String, double[]> out, String name, BiFunction<Var, Boolean, Var> model,
            List<Var> params, SyntheticRecsys.Table train, SyntheticRecsys.Table test, int steps, float lr,
            long seed) {
        BinaryBench.Result r = BinaryBench.train(model, params, train, test, steps, lr, seed);
        out.put(name, new double[] {r.auc(), r.logLoss(), count(params)});
    }

    /**
     * Feature-interaction networks against a planted interaction of a given order (9.1.5 and the
     * RankMixer row of 9.1.6). Six one-hot fields of six categories; the click logit is dominated
     * by a random effect per combination of the first {@code order} fields.
     *
     * @param order the planted interaction order, 2 to 4
     * @param steps gradient steps per model
     * @param seed the seed
     * @return model name to {@code {test AUC, test log-loss, parameters}}
     */
    public static Map<String, double[]> interactionOrder(int order, int steps, long seed) {
        int fields = 6;
        int cats = 6;
        int cols = fields * cats;
        SyntheticRecsys.Table all = ClickLogs.interactionClicks(8000, fields, cats, order, seed);
        SyntheticRecsys.Table train = all.rows(0, 6000);
        SyntheticRecsys.Table test = all.rows(6000, 8000);
        Map<String, double[]> out = new LinkedHashMap<>();
        Model logistic = Model.sequential().input(cols).dense(1).seed(seed).build();
        put(out, "logistic", logistic::forward, logistic.parameters(), train, test, steps, 0.03f, seed);
        FactorizationMachine fm = new FactorizationMachine(cols, 8, new SplittableRandom(seed));
        put(out, "FM", fm::forward, fm.parameters(), train, test, steps, 0.03f, seed);
        WideAndDeep wd = WideAndDeep.wideDeep(fields, cats, 8, 64, seed);
        put(out, "Wide&Deep", wd::forward, wd.parameters(), train, test, steps, 0.01f, seed);
        WideAndDeep dfm = WideAndDeep.deepFm(fields, cats, 8, 64, seed);
        put(out, "DeepFM", dfm::forward, dfm.parameters(), train, test, steps, 0.01f, seed);
        CrossNetwork cross = new CrossNetwork(cols, 3, new SplittableRandom(seed));
        Model head = Model.sequential().input(cols).dense(32, Activation.RELU).dense(1).seed(seed).build();
        List<Var> dcn = new ArrayList<>(cross.parameters());
        dcn.addAll(head.parameters());
        put(out, "DCN-V2", (x, t) -> head.forward(cross.forward(x, t), t), dcn, train, test, steps, 0.01f, seed);
        SplittableRandom r = new SplittableRandom(seed);
        FtTransformer ft = categoricalFt(fields, cats, 8, r);
        put(out, "FT-Transformer", fieldIds(ft, fields, cats), ft.parameters(), train, test, steps, 0.005f, seed);
        FieldAttention autoInt = new FieldAttention(fields, cats, 8, 2, 8, 2, new SplittableRandom(seed));
        put(out, "AutoInt", autoInt::forward, autoInt.parameters(), train, test, steps, 0.01f, seed);
        CompressedInteraction xdfm = new CompressedInteraction(fields, cats, 8, new int[] {16, 16, 16}, 64,
                new SplittableRandom(seed));
        put(out, "xDeepFM (CIN)", xdfm::forward, xdfm.parameters(), train, test, steps, 0.01f, seed);
        RankMixerBlock.Ranker mixer = new RankMixerBlock.Ranker(fields, cats, 8, 3, 24, 2, 2, 0,
                new SplittableRandom(seed));
        put(out, "RankMixer", mixer::forward, mixer.parameters(), train, test, steps, 0.005f, seed);
        return out;
    }

    /**
     * Trains a layer on binary cross-entropy plus its auxiliary loss (the sparse-MoE L1 penalty),
     * batches of 64.
     */
    private static BinaryBench.Result trainLayer(Layer layer, SyntheticRecsys.Table train,
            SyntheticRecsys.Table test, int steps, float lr, long seed) {
        long start = System.currentTimeMillis();
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int cols = train.features().shape().dim(1);
        float[] x = train.features().toArray();
        float[] y = train.labels().toArray();
        int batch = 64;
        List<Var> params = layer.parameters();
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[batch * cols];
            float[] by = new float[batch];
            for (int i = 0; i < batch; i++) {
                int j = r.nextInt(train.rows());
                System.arraycopy(x, j * cols, bx, i * cols, cols);
                by[i] = y[j];
            }
            Var logit = layer.forward(Var.of(Tensor.of(Shape.of(batch, cols), bx)), true);
            Var loss = logit.softplus().sub(logit.mul(Tensor.of(Shape.of(batch, 1), by))).mean();
            Var aux = layer.auxiliaryLoss();
            (aux == null ? loss : loss.add(aux)).backward();
            opt.step(params);
        }
        long millis = System.currentTimeMillis() - start;
        float[] logits = Var.noGrad(() -> layer.forward(Var.of(test.features()), false)).value().toArray();
        return BinaryBench.score(logits, test.labels().toArray(), millis);
    }

    /**
     * RankMixer scaling (9.1.6): dense and sparse-MoE rankers (four feature tokens, one block)
     * and an FT-Transformer (one block, two heads) at each token width, on a planted third-order
     * interaction over eight fields. The HSTU baseline of the item is not included.
     *
     * @param widths token widths, each divisible by 4
     * @param steps gradient steps per model
     * @param seed the seed
     * @return {@code "<model> d=<width>"} to {@code {parameters, test AUC, ms per batch, active
     *     experts per token (NaN for dense models)}}
     */
    public static Map<String, double[]> rankMixerSweep(int[] widths, int steps, long seed) {
        int fields = 8;
        int cats = 6;
        int cols = fields * cats;
        SyntheticRecsys.Table all = ClickLogs.interactionClicks(8000, fields, cats, 3, seed);
        SyntheticRecsys.Table train = all.rows(0, 6000);
        SyntheticRecsys.Table test = all.rows(6000, 8000);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int d : widths) {
            RankMixerBlock.Ranker dense = new RankMixerBlock.Ranker(fields, cats, 8, 4, d, 1, 2, 0,
                    new SplittableRandom(seed));
            BinaryBench.Result rd = trainLayer(dense, train, test, steps, 0.005f, seed);
            out.put("RankMixer d=" + d, new double[] {count(dense.parameters()), rd.auc(),
                rd.millis() / (double) steps, Double.NaN});
            RankMixerBlock.Ranker sparse = new RankMixerBlock.Ranker(fields, cats, 8, 4, d, 1, 2, 4,
                    new SplittableRandom(seed));
            BinaryBench.Result rs = trainLayer(sparse, train, test, steps, 0.005f, seed);
            double active = 0;
            for (RankMixerBlock b : sparse.blocks()) {
                active += b.activeExperts() / sparse.blocks().size();
            }
            out.put("RankMixer-MoE d=" + d, new double[] {count(sparse.parameters()), rs.auc(),
                rs.millis() / (double) steps, active});
            SplittableRandom r = new SplittableRandom(seed);
            FtTransformer ft = categoricalFt(fields, cats, d, r);
            BinaryBench.Result rf = BinaryBench.train(fieldIds(ft, fields, cats), ft.parameters(), train, test, steps,
                    0.005f, seed);
            out.put("FT-Transformer d=" + d, new double[] {count(ft.parameters()), rf.auc(),
                rf.millis() / (double) steps, Double.NaN});
        }
        return out;
    }

    /** An ID embedding plus a context embedding, concatenated into an MLP. */
    private static final class IdModel {
        final CompactEmbedding ids;
        final Var context;
        final Model mlp;

        IdModel(CompactEmbedding ids, int contexts, int dim, long seed) {
            this.ids = ids;
            this.context = Var.param(Tensor.normal(Shape.of(contexts, dim), 0, 0.1f, new SplittableRandom(seed + 1)));
            this.mlp = Model.sequential().input(2 * dim).dense(32, Activation.RELU).dense(1).seed(seed).build();
        }

        Var logits(int[] id, int[] ctx) {
            float[] c = new float[ctx.length];
            for (int i = 0; i < c.length; i++) {
                c[i] = ctx[i];
            }
            Var e = Var.concat(1, ids.lookup(id), context.gather(0, Tensor.of(Shape.of(c.length), c)));
            return mlp.forward(e);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(ids.parameters());
            p.add(context);
            p.addAll(mlp.parameters());
            return p;
        }
    }

    private static double[] trainIds(IdModel m, ClickLogs.IdTable train, ClickLogs.IdTable test, int steps,
            int tailFrom, long seed) {
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom r = new SplittableRandom(seed);
        int batch = 256;
        List<Var> params = m.parameters();
        for (int s = 0; s < steps; s++) {
            int[] id = new int[batch];
            int[] ctx = new int[batch];
            float[] y = new float[batch];
            for (int i = 0; i < batch; i++) {
                int j = r.nextInt(train.rows());
                id[i] = train.id()[j];
                ctx[i] = train.context()[j];
                y[i] = train.label()[j];
            }
            Var logit = m.logits(id, ctx);
            logit.softplus().sub(logit.mul(Tensor.of(Shape.of(batch, 1), y))).mean().backward();
            opt.step(params);
        }
        float[] logits = Var.noGrad(() -> m.logits(test.id(), test.context())).value().toArray();
        int n = logits.length;
        double[] p = new double[n];
        boolean[] pos = new boolean[n];
        int tail = 0;
        for (int i = 0; i < n; i++) {
            p[i] = logits[i];
            pos[i] = test.label()[i] > 0.5f;
            tail += test.id()[i] >= tailFrom ? 1 : 0;
        }
        double[] tp = new double[tail];
        boolean[] tpos = new boolean[tail];
        int k = 0;
        for (int i = 0; i < n; i++) {
            if (test.id()[i] >= tailFrom) {
                tp[k] = p[i];
                tpos[k++] = pos[i];
            }
        }
        return new double[] {MlMetrics.auroc(p, pos), tail == 0 ? Double.NaN : MlMetrics.auroc(tp, tpos)};
    }

    /**
     * Compact embeddings at memory budgets (9.1.7): 80,000 clicks over a 100,000-ID Zipf field
     * (exponent 1.1) and an 8-way context, 16-wide embeddings. For each budget (a fraction of the
     * full table's floats) every compact layout is sized to fit it; the full table is the
     * reference. Tail IDs are those with ID at least 1000.
     *
     * @param budgets memory budgets as fractions of the full table, such as {@code {0.25, 0.03}}
     * @param steps gradient steps of batch 256 per model
     * @param seed the seed
     * @return {@code "<layout> @<budget>"} to {@code {bytes / full bytes, test AUC, tail AUC}}
     */
    public static Map<String, double[]> memorySweep(double[] budgets, int steps, long seed) {
        int vocab = 100_000;
        int dim = 16;
        ClickLogs.IdTable all = ClickLogs.zipfIds(80_000, vocab, 8, 1.1, seed);
        ClickLogs.IdTable train = all.rows(0, 64_000);
        ClickLogs.IdTable test = all.rows(64_000, 80_000);
        Map<String, double[]> out = new LinkedHashMap<>();
        CompactEmbedding full = CompactEmbedding.full(vocab, dim, seed);
        double fullBytes = full.bytes();
        double[] ref = trainIds(new IdModel(full, all.contexts(), dim, seed), train, test, steps, 1000, seed);
        out.put("full", new double[] {1, ref[0], ref[1]});
        for (double budget : budgets) {
            int rows = Math.max(2, (int) (budget * vocab));
            List<CompactEmbedding> tables = new ArrayList<>();
            tables.add(CompactEmbedding.hashed(vocab, dim, rows, seed));
            double disc = (double) rows * rows - 4.0 * vocab;
            int m = disc > 0 ? (int) Math.ceil((rows + Math.sqrt(disc)) / 2) : (int) Math.ceil(Math.sqrt(vocab));
            tables.add(CompactEmbedding.quotientRemainder(vocab, dim, Math.min(m, vocab), seed));
            tables.add(CompactEmbedding.frequencyThreshold(vocab, dim, Math.max(1, rows - 1), seed));
            int tailDim = budget >= 0.25 ? 4 : budget >= 0.08 ? 2 : 1;
            int frequent = (int) ((budget * vocab * dim - (double) vocab * tailDim) / (dim - tailDim));
            if (frequent >= 1) {
                tables.add(CompactEmbedding.mixedDimension(vocab, dim, frequent, tailDim, seed));
            }
            tables.add(CompactEmbedding.lowRank(vocab, dim, Math.max(1, (int) Math.round(budget * dim)), seed));
            for (CompactEmbedding t : tables) {
                double[] res = trainIds(new IdModel(t, all.contexts(), dim, seed), train, test, steps, 1000, seed);
                String kind = t.toString().substring("CompactEmbedding(".length()).split(",")[0];
                out.put(kind + " @" + budget, new double[] {t.bytes() / fullBytes, res[0], res[1]});
            }
        }
        return out;
    }
}
