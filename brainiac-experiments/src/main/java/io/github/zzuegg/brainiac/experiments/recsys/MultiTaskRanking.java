package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ClickLogs;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.nn.CrossNetwork;
import io.github.zzuegg.brainiac.experimental.nn.FineGrainedMoe;
import io.github.zzuegg.brainiac.experimental.recsys.EntireSpaceModel;
import io.github.zzuegg.brainiac.experimental.recsys.MultiGateExperts;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Multi-task ranking (W9 1.1 and 1.2):
 * <ul>
 *   <li>{@link #correlationSweep}: shared-bottom, MMoE and PLE ({@link MultiGateExperts}) against
 *       a DCN-V2 shared body with two heads, two single-task models and a hard-routed
 *       {@link FineGrainedMoe} with the task one-hot appended to its input (task-indexed routing),
 *       on {@link ClickLogs#multiTask} at a given task correlation, by test AUC per task;</li>
 *   <li>{@link #entireSpace}: ESMM ({@link EntireSpaceModel}) against click-space CVR training
 *       (shared-bottom and MMoE bodies) and inverse-propensity-weighted click-space training, on
 *       {@link ClickLogs#conversions}, by CVR AUC over all test impressions (against the latent
 *       "would convert" label) and the calibration error against the true CVR.</li>
 * </ul>
 * Every model gets the same Adam budget with batches of 128.
 */
public final class MultiTaskRanking {

    private MultiTaskRanking() {
    }

    /**
     * Test AUC per task of every model at one correlation.
     *
     * @param rho the task correlation
     * @param steps Adam steps
     * @param seed the seed
     * @return model name to {@code {AUC task 1, AUC task 2}}
     */
    public static Map<String, double[]> correlationSweep(double rho, int steps, long seed) {
        int fields = 4;
        int cats = 8;
        ClickLogs.MultiTaskTable all = ClickLogs.multiTask(8000, fields, cats, rho, seed);
        ClickLogs.MultiTaskTable train = all.rows(0, 6000);
        ClickLogs.MultiTaskTable test = all.rows(6000, 8000);
        int in = fields * cats;
        Map<String, double[]> out = new LinkedHashMap<>();
        for (MultiGateExperts.Kind kind : MultiGateExperts.Kind.values()) {
            MultiGateExperts m = new MultiGateExperts(kind, in, 2, kind == MultiGateExperts.Kind.MMOE ? 4 : 2, 32,
                    16, 2, new SplittableRandom(seed));
            out.put(kind.toString(), train(m::forward, m.parameters(), train, test, steps, seed));
        }
        SplittableRandom r = new SplittableRandom(seed);
        CrossNetwork cross = new CrossNetwork(in, 2, r);
        Var head = Var.param(Tensor.normal(Shape.of(in, 2), 0, (float) Math.sqrt(1.0 / in), r));
        Var headBias = Var.param(Tensor.zeros(Shape.of(2)));
        List<Var> dcn = new ArrayList<>(cross.parameters());
        dcn.add(head);
        dcn.add(headBias);
        out.put("DCN-V2 shared, two heads", train(x -> cross.forward(x).matmul(head).add(headBias), dcn, train, test,
                steps, seed));
        double[] single = new double[2];
        for (int t = 0; t < 2; t++) {
            MultiGateExperts m = new MultiGateExperts(MultiGateExperts.Kind.SHARED_BOTTOM, in, 2, 1, 32, 16, 1,
                    new SplittableRandom(seed + t));
            single[t] = train(m::forward, m.parameters(), train, test, steps, seed, t)[t];
        }
        out.put("single-task models", single);
        FineGrainedMoe moe = FineGrainedMoe.of(32, 4, 0, 32, 1, r);
        Var proj = Var.param(Tensor.normal(Shape.of(in + 2, 32), 0, (float) Math.sqrt(2.0 / (in + 34)), r));
        Var out1 = Var.param(Tensor.normal(Shape.of(32, 1), 0, 0.2f, r));
        Var bias1 = Var.param(Tensor.zeros(Shape.of(1)));
        List<Var> moeParams = new ArrayList<>(moe.parameters());
        moeParams.addAll(List.of(proj, out1, bias1));
        out.put("FineGrainedMoe (task-indexed routing)", train(x -> {
            int b = x.shape().dim(0);
            List<Var> logits = new ArrayList<>();
            for (int t = 0; t < 2; t++) {
                float[] onehot = new float[b * 2];
                for (int i = 0; i < b; i++) {
                    onehot[i * 2 + t] = 1;
                }
                Var xt = Var.concat(1, x, Var.of(Tensor.of(Shape.of(b, 2), onehot)));
                logits.add(moe.forward(xt.matmul(proj).relu()).relu().matmul(out1).add(bias1));
            }
            return Var.concat(1, logits.toArray(Var[]::new));
        }, moeParams, train, test, steps, seed));
        return out;
    }

    private static double[] train(UnaryOperator<Var> model, List<Var> params, ClickLogs.MultiTaskTable train,
            ClickLogs.MultiTaskTable test, int steps, long seed) {
        return train(model, params, train, test, steps, seed, -1);
    }

    /** Trains on both tasks ({@code onlyTask < 0}) or one, and returns the test AUC per task. */
    private static double[] train(UnaryOperator<Var> model, List<Var> params, ClickLogs.MultiTaskTable train,
            ClickLogs.MultiTaskTable test, int steps, long seed, int onlyTask) {
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 11);
        int cols = train.features().shape().dim(1);
        float[] x = train.features().toArray();
        float[] y = train.labels().toArray();
        int batch = 128;
        float[] mask = {onlyTask == 1 ? 0 : 1, onlyTask == 0 ? 0 : 1};
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[batch * cols];
            float[] by = new float[batch * 2];
            float[] bm = new float[batch * 2];
            for (int i = 0; i < batch; i++) {
                int j = r.nextInt(train.rows());
                System.arraycopy(x, j * cols, bx, i * cols, cols);
                by[2 * i] = y[2 * j];
                by[2 * i + 1] = y[2 * j + 1];
                bm[2 * i] = mask[0];
                bm[2 * i + 1] = mask[1];
            }
            Var logit = model.apply(Var.of(Tensor.of(Shape.of(batch, cols), bx)));
            logit.softplus().sub(logit.mul(Tensor.of(Shape.of(batch, 2), by))).mul(Tensor.of(Shape.of(batch, 2), bm))
                    .mean().backward();
            opt.step(params);
        }
        float[] logits = Var.noGrad(() -> model.apply(Var.of(test.features()))).value().toArray();
        float[] ty = test.labels().toArray();
        double[] auc = new double[2];
        for (int t = 0; t < 2; t++) {
            double[] p = new double[test.rows()];
            boolean[] pos = new boolean[test.rows()];
            for (int i = 0; i < p.length; i++) {
                p[i] = logits[2 * i + t];
                pos[i] = ty[2 * i + t] > 0.5f;
            }
            auc[t] = MlMetrics.auroc(p, pos);
        }
        return auc;
    }

    /**
     * ESMM against click-space CVR training.
     *
     * @param steps Adam steps
     * @param seed the seed
     * @return model name to {@code {CVR AUC on all impressions, CVR calibration error, CTR AUC}}
     */
    public static Map<String, double[]> entireSpace(int steps, long seed) {
        int fields = 4;
        int cats = 8;
        ClickLogs.ConversionLog all = ClickLogs.conversions(20000, fields, cats, seed);
        ClickLogs.ConversionLog train = all.rows(0, 15000);
        ClickLogs.ConversionLog test = all.rows(15000, 20000);
        int in = fields * cats;
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("click-space CVR (shared bottom)", conversions(MultiGateExperts.Kind.SHARED_BOTTOM, Mode.CLICK_SPACE,
                train, test, in, steps, seed));
        out.put("click-space CVR (MMoE)", conversions(MultiGateExperts.Kind.MMOE, Mode.CLICK_SPACE, train, test, in,
                steps, seed));
        out.put("click-space CVR + IPW (shared bottom)", conversions(MultiGateExperts.Kind.SHARED_BOTTOM, Mode.IPW,
                train, test, in, steps, seed));
        out.put("ESMM (shared bottom)", conversions(MultiGateExperts.Kind.SHARED_BOTTOM, Mode.ESMM, train, test, in,
                steps, seed));
        out.put("ESMM (MMoE)", conversions(MultiGateExperts.Kind.MMOE, Mode.ESMM, train, test, in, steps, seed));
        return out;
    }

    private enum Mode {
        CLICK_SPACE, IPW, ESMM
    }

    private static double[] conversions(MultiGateExperts.Kind kind, Mode mode, ClickLogs.ConversionLog train,
            ClickLogs.ConversionLog test, int in, int steps, long seed) {
        EntireSpaceModel m = new EntireSpaceModel(new MultiGateExperts(kind, in, 2, 4, 32, 16, 1,
                new SplittableRandom(seed)));
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 5);
        float[] x = train.features().toArray();
        int batch = 128;
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[batch * in];
            float[] click = new float[batch];
            float[] conv = new float[batch];
            for (int i = 0; i < batch; i++) {
                int j = r.nextInt(train.rows());
                System.arraycopy(x, j * in, bx, i * in, in);
                click[i] = train.click()[j];
                conv[i] = train.convert()[j];
            }
            Shape col = Shape.of(batch, 1);
            Var logits = m.forward(Var.of(Tensor.of(Shape.of(batch, in), bx)));
            Var loss;
            if (mode == Mode.ESMM) {
                float[] both = new float[batch];
                for (int i = 0; i < batch; i++) {
                    both[i] = click[i] * conv[i];
                }
                loss = EntireSpaceModel.loss(logits, Tensor.of(col, click), Tensor.of(col, both));
            } else {
                float[] w = new float[batch];
                float[] ctr = logits.value().slice(1, 0, 1).toArray();
                for (int i = 0; i < batch; i++) {
                    w[i] = mode == Mode.IPW ? (float) (1 / Math.max(0.05, 1 / (1 + Math.exp(-ctr[i])))) : 1;
                }
                loss = EntireSpaceModel.clickSpaceLoss(logits, Tensor.of(col, click), Tensor.of(col, conv),
                        Tensor.of(col, w));
            }
            loss.backward();
            opt.step(m.parameters());
        }
        float[] logits = Var.noGrad(() -> m.forward(Var.of(test.features()))).value().toArray();
        int n = test.rows();
        double[] cvr = new double[n];
        double[] ctr = new double[n];
        boolean[] converts = new boolean[n];
        boolean[] clicks = new boolean[n];
        for (int i = 0; i < n; i++) {
            ctr[i] = logits[2 * i];
            cvr[i] = 1 / (1 + Math.exp(-logits[2 * i + 1]));
            converts[i] = test.convert()[i] > 0.5f;
            clicks[i] = test.click()[i] > 0.5f;
        }
        return new double[] {MlMetrics.auroc(cvr, converts), calibration(cvr, test.trueCvr()),
            MlMetrics.auroc(ctr, clicks)};
    }

    /** Expected calibration error against the true probabilities, over 10 equal-width bins. */
    private static double calibration(double[] p, double[] truth) {
        double[] sp = new double[10];
        double[] st = new double[10];
        int[] c = new int[10];
        for (int i = 0; i < p.length; i++) {
            int b = Math.min(9, (int) (p[i] * 10));
            sp[b] += p[i];
            st[b] += truth[i];
            c[b]++;
        }
        double e = 0;
        for (int b = 0; b < 10; b++) {
            e += Math.abs(sp[b] - st[b]) / p.length;
        }
        return e;
    }
}
