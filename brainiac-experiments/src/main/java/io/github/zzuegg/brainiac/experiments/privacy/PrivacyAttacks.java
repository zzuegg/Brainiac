package io.github.zzuegg.brainiac.experiments.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.data.ClientSplits;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.privacy.GradientInversionAttack;
import io.github.zzuegg.brainiac.experimental.privacy.LikelihoodRatioAttack;
import io.github.zzuegg.brainiac.experimental.privacy.TeacherEnsembleAggregation;
import io.github.zzuegg.brainiac.experimental.train.DpSgd;
import io.github.zzuegg.brainiac.experimental.train.GradientCompressor;
import io.github.zzuegg.brainiac.experimental.train.MachineUnlearning;
import io.github.zzuegg.brainiac.experimental.train.QuantizedGradients;
import io.github.zzuegg.brainiac.experimental.train.RdpAccountant;
import io.github.zzuegg.brainiac.experimental.train.SparseGradientCompression;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Wave 8 items 8.6.5 to 8.6.7: privacy attacks and the private-teacher alternative.
 *
 * <ul>
 *   <li>{@link #inversion}: {@link GradientInversionAttack} (DLG and cosine + TV) against a
 *       tanh MLP at initialisation on {@link ToyTasks#digits} or {@link ShapeImages}, for batch
 *       sizes and defences: none, DP noise (per-example clipping plus Gaussian noise),
 *       top-k sparsification ({@link SparseGradientCompression}), QSGD
 *       ({@link QuantizedGradients}), and secure aggregation (the server sees only the mean
 *       of ten clients' gradients). Reported with each defence's accuracy cost when training
 *       with it.</li>
 *   <li>{@link #lira}: {@link LikelihoodRatioAttack} on small MLPs trained on random halves
 *       of a 500-digit pool, against the loss-threshold attack, for non-private training,
 *       DP-SGD and unlearning (a model trained on a fixed half, then NPO- or
 *       gradient-ascent-unlearned on part of it; the forget set is compared with unseen pool
 *       examples).</li>
 *   <li>{@link #pate}: {@link TeacherEnsembleAggregation} teachers on disjoint parts of the
 *       private digits label public queries for a student; against DP-SGD at the same
 *       {@code ε} and non-private training.</li>
 * </ul>
 */
public final class PrivacyAttacks {

    /** A gradient-inversion defence. */
    public enum Defence {
        /** the raw gradient */
        NONE,
        /** per-example clipping to 1 and Gaussian noise of standard deviation {@code σ = 0.5} */
        DP_NOISE,
        /** the largest 1% of entries per tensor */
        TOP_K_1,
        /** the largest 10% of entries per tensor */
        TOP_K_10,
        /** QSGD with 4 levels */
        QSGD,
        /** only the mean over ten clients is visible */
        SECURE_AGGREGATION
    }

    /**
     * One inversion run.
     *
     * @param attack DLG or cosine+TV
     * @param defence the defence
     * @param batch the batch size
     * @param psnr mean reconstruction PSNR (dB) after matching
     * @param labelAccuracy the share of labels recovered
     * @param defendedAccuracy test accuracy of training with the defence
     */
    public record Inversion(String attack, Defence defence, int batch, double psnr, double labelAccuracy,
            double defendedAccuracy) {
    }

    /**
     * One membership-inference row.
     *
     * @param model the attacked training
     * @param lira online LiRA
     * @param offline offline LiRA
     * @param loss the loss-threshold baseline
     * @param testAccuracy the target models' mean test accuracy
     */
    public record Membership(String model, LikelihoodRatioAttack.Quality lira, LikelihoodRatioAttack.Quality offline,
            LikelihoodRatioAttack.Quality loss, double testAccuracy) {
    }

    /**
     * One PATE or baseline row.
     *
     * @param method the method
     * @param epsilon the spent {@code ε}
     * @param accuracy the student's test accuracy
     * @param answered queries answered (labelled examples the student got)
     * @param labelAccuracy the share of the student's labels that are correct
     */
    public record Pate(String method, double epsilon, double accuracy, int answered, double labelAccuracy) {
    }

    static final double DELTA = 1e-5;

    private PrivacyAttacks() {
    }

    // ---------------------------------------------------------------- gradient inversion

    /**
     * Gradient inversion against a defence.
     *
     * @param shapes {@link ShapeImages} if true, digits otherwise
     * @param cosine cosine + TV if true, DLG otherwise
     * @param defence the defence
     * @param batch the batch size
     * @param iterations attack steps
     * @param seed the seed
     * @return the outcome, averaged over the batch
     */
    public static Inversion inversion(boolean shapes, boolean cosine, Defence defence, int batch, int iterations,
            long seed) {
        ClientSplits.Labelled[] d = inversionData(shapes, seed);
        int classes = d[0].classes();
        SimpleMlp net = SimpleMlp.tanh(seed, 64, 32, classes);
        UnaryOperator<Var> logits = net::forward;
        List<Var> params = net.parameters();
        int[] rows = new int[batch];
        SplittableRandom r = new SplittableRandom(seed + 5);
        for (int i = 0; i < batch; i++) {
            rows[i] = r.nextInt(d[0].size());
        }
        ClientSplits.Labelled target = d[0].subset(rows);
        Tensor[] observed = defend(defence, logits, params, target, d[0], r);
        GradientInversionAttack attack = cosine
                ? GradientInversionAttack.cosineTv(logits, params, classes, 8, 0.02f).iterations(iterations, 0.05f)
                : GradientInversionAttack.dlg(logits, params, classes).iterations(iterations, 0.05f);
        GradientInversionAttack.Reconstruction rec = attack.attack(observed, batch, 64, 3, seed + 7);
        return new Inversion(cosine ? "cosine+TV" : "DLG", defence, batch,
                GradientInversionAttack.matchedPsnr(rec.inputs(), target.inputs()),
                GradientInversionAttack.labelAccuracy(rec.labels(), target.y(), classes),
                defendedTraining(d, defence, seed));
    }

    private static ClientSplits.Labelled[] inversionData(boolean shapes, long seed) {
        if (shapes) {
            ShapeImages images = new ShapeImages(8, 0.05f);
            return new ClientSplits.Labelled[] {ClientSplits.fromImages(images.generate(600, seed + 1), 6),
                    ClientSplits.fromImages(images.generate(300, seed + 2), 6)};
        }
        ToyTasks.Data tr = ToyTasks.digits(10, 60, 8, 0.05, seed, seed + 1);
        ToyTasks.Data te = ToyTasks.digits(10, 30, 8, 0.05, seed, seed + 2);
        return new ClientSplits.Labelled[] {new ClientSplits.Labelled(tr.x(), tr.y(), 10),
                new ClientSplits.Labelled(te.x(), te.y(), 10)};
    }

    /** The gradient the server observes under a defence. */
    private static Tensor[] defend(Defence defence, UnaryOperator<Var> logits, List<Var> params,
            ClientSplits.Labelled batch, ClientSplits.Labelled pool, SplittableRandom r) {
        Tensor[] g = GradientInversionAttack.clientGradient(logits, params, batch.inputs(), batch.ids());
        return switch (defence) {
            case NONE -> g;
            case DP_NOISE -> dpGradient(logits, params, batch, 1.0, 0.5, r);
            case TOP_K_1 -> compress(SparseGradientCompression.topK(0.01, false), params, g);
            case TOP_K_10 -> compress(SparseGradientCompression.topK(0.1, false), params, g);
            case QSGD -> compress(QuantizedGradients.qsgd(4, 128, r.nextLong()), params, g);
            case SECURE_AGGREGATION -> {
                Tensor[] sum = g.clone();
                for (int c = 1; c < 10; c++) {
                    int[] rows = new int[batch.size()];
                    for (int i = 0; i < rows.length; i++) {
                        rows[i] = r.nextInt(pool.size());
                    }
                    ClientSplits.Labelled other = pool.subset(rows);
                    Tensor[] go = GradientInversionAttack.clientGradient(logits, params, other.inputs(), other.ids());
                    for (int p = 0; p < sum.length; p++) {
                        sum[p] = sum[p].add(go[p]);
                    }
                }
                for (int p = 0; p < sum.length; p++) {
                    sum[p] = sum[p].div(10);
                }
                yield sum;
            }
        };
    }

    private static Tensor[] compress(GradientCompressor c, List<Var> params, Tensor[] g) {
        Tensor[] out = new Tensor[g.length];
        for (int p = 0; p < g.length; p++) {
            out[p] = c.aggregate(params.get(p), List.of(g[p]));
        }
        return out;
    }

    /** Per-example clipped gradients plus Gaussian noise, averaged: one DP-SGD step's gradient. */
    private static Tensor[] dpGradient(UnaryOperator<Var> logits, List<Var> params, ClientSplits.Labelled batch,
            double clip, double sigma, SplittableRandom r) {
        Tensor[] sum = null;
        for (int i = 0; i < batch.size(); i++) {
            ClientSplits.Labelled one = batch.subset(new int[] {i});
            Tensor[] g = GradientInversionAttack.clientGradient(logits, params, one.inputs(), one.ids());
            double norm = 0;
            for (Tensor t : g) {
                norm += t.square().sum().item();
            }
            float scale = (float) Math.min(1, clip / Math.max(Math.sqrt(norm), 1e-12));
            if (sum == null) {
                sum = new Tensor[g.length];
            }
            for (int p = 0; p < g.length; p++) {
                sum[p] = sum[p] == null ? g[p].mul(scale) : sum[p].add(g[p].mul(scale));
            }
        }
        for (int p = 0; p < sum.length; p++) {
            Tensor noise = Tensor.normal(sum[p].shape(), 0, (float) (sigma * clip), r);
            sum[p] = sum[p].add(noise).div(batch.size());
        }
        return sum;
    }

    private static final java.util.Map<String, Double> DEFENDED = new java.util.concurrent.ConcurrentHashMap<>();

    /** Test accuracy of 300 SGD steps (batch 8) whose gradients pass through the defence (memoised). */
    private static double defendedTraining(ClientSplits.Labelled[] d, Defence defence, long seed) {
        String key = d[0].classes() + "/" + defence + "/" + seed;
        Double known = DEFENDED.get(key);
        if (known != null) {
            return known;
        }
        double acc = trainDefended(d, defence, seed);
        DEFENDED.put(key, acc);
        return acc;
    }

    private static double trainDefended(ClientSplits.Labelled[] d, Defence defence, long seed) {
        SimpleMlp net = SimpleMlp.tanh(seed + 11, 64, 32, d[0].classes());
        UnaryOperator<Var> logits = net::forward;
        List<Var> params = net.parameters();
        SplittableRandom r = new SplittableRandom(seed + 12);
        for (int s = 0; s < 300; s++) {
            int[] rows = new int[8];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = r.nextInt(d[0].size());
            }
            ClientSplits.Labelled b = d[0].subset(rows);
            Tensor[] g = defence == Defence.SECURE_AGGREGATION ? defend(Defence.NONE, logits, params, b, d[0], r)
                    : defend(defence, logits, params, b, d[0], r);
            for (int p = 0; p < g.length; p++) {
                params.get(p).assign(params.get(p).value().sub(g[p].mul(0.3f)));
            }
        }
        return ToyTasks.accuracy(net.predict(d[1].inputs()), d[1].y());
    }

    // ---------------------------------------------------------------- LiRA

    /** How the attacked models are trained. */
    public enum Training {
        /** Adam, full batch */
        PLAIN,
        /** DP-SGD */
        DP_SGD
    }

    /**
     * LiRA against the loss attack on models trained on random halves of a pool.
     *
     * @param training how every model is trained
     * @param epsilon the DP-SGD budget (ignored for plain training)
     * @param pool the pool size
     * @param shadows the number of models (each in turn the target)
     * @param seed the seed
     * @return the membership row
     */
    public static Membership lira(Training training, double epsilon, int pool, int shadows, long seed) {
        Dataset[] d = liraData(pool, seed);
        double[] testAcc = new double[1];
        LikelihoodRatioAttack.ShadowTrainer trainer = (members, s) -> {
            Model m = train(training, epsilon, subset(d[0], members), s);
            testAcc[0] += MachineUnlearning.accuracy(m, d[1]) / shadows;
            return trueLabelProbabilities(m, d[0]);
        };
        LikelihoodRatioAttack a = LikelihoodRatioAttack.fit(pool, shadows, trainer, true, seed);
        String name = training == Training.PLAIN ? "non-private" : "DP-SGD eps " + epsilon;
        return new Membership(name, a.evaluate(true), a.evaluate(false), a.evaluateLossThreshold(), testAcc[0]);
    }

    /**
     * LiRA and the loss attack against unlearning: a target trained on a fixed half of the
     * pool unlearns part of it; the forget set is scored against pool examples the target never
     * saw (an ideal unlearner makes them indistinguishable, AUC 0.5).
     *
     * @param method the unlearning method, or {@code null} for the original model
     * @param pool the pool size
     * @param shadows shadow models
     * @param forget the size of the forget set
     * @param seed the seed
     * @return AUC and TPR of LiRA and the loss attack on forget versus unseen examples
     */
    public static Membership liraUnlearning(MachineUnlearning.Method method, int pool, int shadows, int forget,
            long seed) {
        Dataset[] d = liraData(pool, seed);
        LikelihoodRatioAttack a = LikelihoodRatioAttack.fit(pool, shadows,
                (members, s) -> trueLabelProbabilities(train(Training.PLAIN, 0, subset(d[0], members), s), d[0]), true,
                seed);
        SplittableRandom r = new SplittableRandom(seed + 31);
        boolean[] members = new boolean[pool];
        for (int i = 0; i < pool; i++) {
            members[i] = r.nextBoolean();
        }
        Model target = train(Training.PLAIN, 0, subset(d[0], members), seed + 99);
        boolean[] forgetMask = new boolean[pool];
        boolean[] retainMask = members.clone();
        int f = 0;
        for (int i = 0; i < pool && f < forget; i++) {
            if (members[i]) {
                forgetMask[i] = true;
                retainMask[i] = false;
                f++;
            }
        }
        if (method != null) {
            MachineUnlearning.of(target, Loss.CROSS_ENTROPY).method(method).optimizer(() -> Optimizer.adam(0.003f))
                    .steps(100).batchSize(16).seed(seed).run(subset(d[0], forgetMask), subset(d[0], retainMask));
        }
        double[] p = trueLabelProbabilities(target, d[0]);
        double[] lira = a.score(p, true);
        double[] offline = a.score(p, false);
        double[] loss = LikelihoodRatioAttack.logitScale(p);
        List<Integer> evalRows = new ArrayList<>();
        for (int i = 0; i < pool; i++) {
            if (forgetMask[i] || !members[i]) {
                evalRows.add(i);
            }
        }
        double[][] s = new double[3][evalRows.size()];
        boolean[] label = new boolean[evalRows.size()];
        for (int k = 0; k < label.length; k++) {
            int i = evalRows.get(k);
            s[0][k] = lira[i];
            s[1][k] = offline[i];
            s[2][k] = loss[i];
            label[k] = forgetMask[i];
        }
        return new Membership(method == null ? "original (no unlearning)" : method.toString(),
                LikelihoodRatioAttack.quality(s[0], label), LikelihoodRatioAttack.quality(s[1], label),
                LikelihoodRatioAttack.quality(s[2], label), MachineUnlearning.accuracy(target, d[1]));
    }

    /** A noisy digits pool (so that models overfit) and a test set. */
    private static Dataset[] liraData(int pool, long seed) {
        ToyTasks.Data tr = ToyTasks.digits(10, pool / 10, 8, 0.2, seed, seed + 1);
        ToyTasks.Data te = ToyTasks.digits(10, 30, 8, 0.2, seed, seed + 2);
        return new Dataset[] {dataset(tr), dataset(te)};
    }

    static Dataset dataset(ToyTasks.Data d) {
        float[][] y = new float[d.size()][d.classes()];
        for (int i = 0; i < y.length; i++) {
            y[i][d.y()[i]] = 1;
        }
        return Dataset.of(d.x(), y);
    }

    static Dataset subset(Dataset d, boolean[] keep) {
        List<float[]> x = new ArrayList<>();
        List<float[]> y = new ArrayList<>();
        for (int i = 0; i < d.size(); i++) {
            if (keep[i]) {
                x.add(d.featuresOf(i));
                y.add(d.targetsOf(i));
            }
        }
        return Dataset.of(x.toArray(new float[0][]), y.toArray(new float[0][]));
    }

    static Model model(long seed) {
        return Model.sequential().input(64).dense(64, Activation.TANH).dense(10, Activation.SOFTMAX).seed(seed).build();
    }

    /** Trains a fresh model on the data: 150 full-batch Adam steps, or DP-SGD at the budget. */
    static Model train(Training training, double epsilon, Dataset data, long seed) {
        Model m = model(seed);
        if (training == Training.DP_SGD) {
            int steps = 100;
            double q = 0.1;
            double sigma = RdpAccountant.noiseMultiplierFor(epsilon, q, steps, DELTA);
            DpSgd.of(m, Loss.CROSS_ENTROPY).clipNorm(1).noiseMultiplier(sigma).learningRate(2).samplingRate(q)
                    .seed(seed).train(data, steps);
            return m;
        }
        float[][] x = new float[data.size()][];
        float[][] y = new float[data.size()][];
        for (int i = 0; i < x.length; i++) {
            x[i] = data.featuresOf(i);
            y[i] = data.targetsOf(i);
        }
        Tensor tx = Tensor.of(x);
        Tensor ty = Tensor.of(y);
        Optimizer adam = Optimizer.adam(0.01f);
        for (int s = 0; s < 150; s++) {
            Loss.CROSS_ENTROPY.compute(m.forward(Var.of(tx), true), ty).backward();
            adam.step(m.parameters());
        }
        return m;
    }

    static double[] trueLabelProbabilities(Model m, Dataset data) {
        float[][] x = new float[data.size()][];
        for (int i = 0; i < x.length; i++) {
            x[i] = data.featuresOf(i);
        }
        float[][] p = m.predict(x);
        double[] out = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            float[] t = data.targetsOf(i);
            int label = 0;
            for (int c = 1; c < t.length; c++) {
                label = t[c] > t[label] ? c : label;
            }
            out[i] = p[i][label];
        }
        return out;
    }

    // ---------------------------------------------------------------- PATE

    /**
     * PATE against DP-SGD and non-private training on the same private data.
     *
     * @param teachers the number of teachers (disjoint shares of the private data)
     * @param privateSize private labelled examples
     * @param queries public examples the student asks about
     * @param sigma1 the threshold-check noise
     * @param sigma2 the answer noise
     * @param threshold the confidence threshold
     * @param seed the seed
     * @return non-private, GNMax (all queries), Confident-GNMax, DP-SGD at Confident-GNMax's ε
     */
    public static List<Pate> pate(int teachers, int privateSize, int queries, double sigma1, double sigma2,
            double threshold, long seed) {
        ToyTasks.Data priv = ToyTasks.digits(10, privateSize / 10, 8, 0.05, seed, seed + 1);
        ToyTasks.Data pub = ToyTasks.digits(10, queries / 10, 8, 0.05, seed, seed + 2);
        ToyTasks.Data test = ToyTasks.digits(10, 30, 8, 0.05, seed, seed + 3);
        List<Pate> out = new ArrayList<>();
        out.add(new Pate("non-private", Double.POSITIVE_INFINITY, studentAccuracy(priv.x(), priv.y(), test, seed),
                priv.size(), 1));
        // teachers
        int[][] votes = new int[pub.size()][teachers];
        int share = priv.size() / teachers;
        for (int t = 0; t < teachers; t++) {
            ToyTasks.Data part = priv.slice(t * share, (t + 1) * share);
            SimpleMlp teacher = SimpleMlp.tanh(seed + 100 + t, 64, 32, 10);
            fit(teacher, part.x(), part.y(), 150, seed + t);
            float[] l = teacher.predict(Tensor.of(pub.x())).toArray();
            for (int i = 0; i < pub.size(); i++) {
                int best = 0;
                for (int c = 1; c < 10; c++) {
                    best = l[i * 10 + c] > l[i * 10 + best] ? c : best;
                }
                votes[i][t] = best;
            }
        }
        for (boolean confident : new boolean[] {false, true}) {
            TeacherEnsembleAggregation agg = TeacherEnsembleAggregation.of(10, seed + 7);
            List<float[]> sx = new ArrayList<>();
            List<Integer> sy = new ArrayList<>();
            int labelRight = 0;
            for (int i = 0; i < pub.size(); i++) {
                double[] counts = TeacherEnsembleAggregation.votes(votes[i], 10);
                int label = confident ? agg.confidentGnMax(counts, threshold, sigma1, sigma2)
                        : agg.gnMax(counts, sigma2);
                if (label >= 0) {
                    sx.add(pub.x()[i]);
                    sy.add(label);
                    labelRight += label == pub.y()[i] ? 1 : 0;
                }
            }
            float[][] x = sx.toArray(new float[0][]);
            int[] y = sy.stream().mapToInt(Integer::intValue).toArray();
            double eps = agg.epsilon(DELTA);
            out.add(new Pate(confident ? "Confident-GNMax" : "GNMax", eps,
                    y.length == 0 ? 0 : studentAccuracy(x, y, test, seed), agg.answered(),
                    y.length == 0 ? 0 : (double) labelRight / y.length));
            if (confident) {
                // DP-SGD baseline at the same budget
                int steps = 200;
                double q = 0.05;
                double sigma = RdpAccountant.noiseMultiplierFor(eps, q, steps, DELTA);
                Model m = model(seed);
                DpSgd dp = DpSgd.of(m, Loss.CROSS_ENTROPY).clipNorm(1).noiseMultiplier(sigma).learningRate(2)
                        .samplingRate(q).seed(seed).train(dataset(priv), steps);
                out.add(new Pate("DP-SGD", dp.epsilon(DELTA), MachineUnlearning.accuracy(m, dataset(test)),
                        priv.size(), 1));
            }
        }
        return out;
    }

    private static double studentAccuracy(float[][] x, int[] y, ToyTasks.Data test, long seed) {
        SimpleMlp student = SimpleMlp.tanh(seed + 55, 64, 32, 10);
        fit(student, x, y, 200, seed);
        return ToyTasks.accuracy(student.predict(test.inputs()), test.y());
    }

    private static void fit(SimpleMlp net, float[][] x, int[] y, int steps, long seed) {
        Tensor tx = Tensor.of(x);
        float[] ids = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            ids[i] = y[i];
        }
        Tensor ty = Tensor.of(Shape.of(y.length), ids);
        Optimizer adam = Optimizer.adam(0.01f);
        for (int s = 0; s < steps; s++) {
            Functional.crossEntropy(net.forward(Var.of(tx)), ty).backward();
            adam.step(net.parameters());
        }
    }
}
