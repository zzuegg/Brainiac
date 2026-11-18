package io.github.zzuegg.brainiac.experiments.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DataPruner;
import io.github.zzuegg.brainiac.experimental.data.InRunShapley;
import io.github.zzuegg.brainiac.experimental.interp.Trak;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.TrainingDataAttribution;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Influence functions and TracIn ({@link TrainingDataAttribution}) on the flipped-label toy
 * digits of {@link ShapleyPruning} (the same data and seeds), against EL2N and In-Run Data
 * Shapley.
 *
 * <p><b>Flipped-label detection</b> ({@link #detection}): the share of flipped labels among
 * the 10% of examples each score flags, from one SGD run of the MLP (TracIn checkpoints at
 * every epoch; the influence function on the last layer of the final network).
 * {@link ShapleyPruning#run} gives the same number for EL2N, In-Run Shapley and random.
 *
 * <p><b>Leave-k-out</b> ({@link #leaveOut}): for a linear softmax classifier trained by SGD
 * (convex, so the influence function's assumptions hold), random groups of 60 examples are
 * removed and the run repeated with the same batches minus those examples; each method's
 * summed per-example prediction of the validation-loss change is correlated (Pearson) with
 * the actual change.
 *
 * <p><b>Linear datamodeling score</b> ({@link #datamodeling}, wave 9 item 9.4.9): per
 * validation target, the Spearman correlation over random half-size training subsets
 * between a method's summed attributions and the target's margin in a model retrained on
 * that subset (same batches minus the left-out rows). Methods: {@link Trak} with one and
 * with several models, the influence function (exact Hessian; for the MLP on its last
 * layer), TracIn over the epoch checkpoints, and the plain gradient dot product of the
 * final model.
 */
public final class TrainingAttributionExperiment {

    /** The attribution scores. */
    public enum Method {
        /** Influence function, self-influence (detection) or validation influence (leave-k-out). */
        INFLUENCE,
        /** TracIn, self-influence (detection) or validation influence (leave-k-out). */
        TRACIN,
        /** In-Run Data Shapley (first order). */
        IN_RUN_SHAPLEY,
        /** EL2N, the mean error norm early in training. */
        EL2N
    }

    private record Labelled(Tensor x, Tensor y, boolean[] flipped) {
    }

    private static final int BATCH = 32;
    private static final float LR = 0.2f;
    private static final int EPOCHS = 8;

    private TrainingAttributionExperiment() {
    }

    /** As {@code ShapleyPruning.tensors}, so the flipped labels are the same. */
    private static Labelled tensors(ToyTasks.Data d, double flip, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = d.size();
        int f = d.features();
        float[] x = new float[n * f];
        float[] y = new float[n];
        boolean[] flipped = new boolean[n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(d.x()[i], 0, x, i * f, f);
            int label = d.y()[i];
            if (r.nextDouble() < flip) {
                label = (label + 1 + r.nextInt(d.classes() - 1)) % d.classes();
                flipped[i] = true;
            }
            y[i] = label;
        }
        return new Labelled(Tensor.of(Shape.of(n, f), x), Tensor.of(Shape.of(n), y), flipped);
    }

    private static Labelled train(long seed, double flip) {
        return tensors(ToyTasks.digits(10, 60, 8, 0.08, seed, seed + 100), flip, seed + 300);
    }

    private static Labelled valid(long seed) {
        return tensors(ToyTasks.digits(10, 20, 8, 0.08, seed, seed + 400), 0, seed);
    }

    /** The batches of every epoch, fixed by the seed. */
    private static int[][][] schedule(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[][][] out = new int[EPOCHS][n / BATCH][];
        for (int e = 0; e < EPOCHS; e++) {
            int[] order = r.ints(0, n).distinct().limit(n).toArray();
            for (int b = 0; b < n / BATCH; b++) {
                out[e][b] = Arrays.copyOfRange(order, b * BATCH, (b + 1) * BATCH);
            }
        }
        return out;
    }

    /** Hooks into a run: called before every step and after every epoch. */
    private interface Observer {
        default void beforeStep(SimpleMlp net, int[] rows, Tensor x, Tensor y) {
        }

        default void afterEpoch(SimpleMlp net) {
        }
    }

    private static SimpleMlp run(Labelled data, int[] sizes, long seed, boolean[] removed, Observer observer) {
        SimpleMlp net = SimpleMlp.relu(seed, sizes);
        Optimizer opt = Optimizer.sgd(LR);
        for (int[][] epoch : schedule(data.x().shape().dim(0), seed)) {
            for (int[] batch : epoch) {
                int[] rows = removed == null ? batch
                        : Arrays.stream(batch).filter(i -> !removed[i]).toArray();
                if (rows.length == 0) {
                    continue;
                }
                Tensor x = DataPruner.select(data.x(), rows);
                Tensor y = DataPruner.select(data.y(), rows);
                observer.beforeStep(net, rows, x, y);
                // the loss is summed over the kept rows and divided by the full batch size,
                // so removing an example takes away exactly its own share of the step
                Functional.crossEntropy(net.forward(Var.of(x)), y).mul(rows.length / (float) BATCH).backward();
                opt.step(net.parameters());
            }
            observer.afterEpoch(net);
        }
        return net;
    }

    private static TrainingDataAttribution attribution(SimpleMlp net, List<Var> params) {
        return new TrainingDataAttribution(params, (x, y) -> Functional.crossEntropy(net.forward(Var.of(x)), y));
    }

    /**
     * The flipped-label precision of one score: the share of flipped labels among the 10% of
     * examples flagged.
     *
     * @param method the score (the influence-based ones use self-influence)
     * @param flip the share of labels flipped
     * @param seed the seed
     * @return the precision; random flagging gives about {@code flip}
     */
    public static double detection(Method method, double flip, long seed) {
        Labelled train = train(seed, flip);
        int n = train.x().shape().dim(0);
        double[] suspicion;
        switch (method) {
            case INFLUENCE -> {
                SimpleMlp net = run(train, new int[] {64, 64, 10}, seed + 1000, null, new Observer() { });
                suspicion = attribution(net, List.of(net.weight(1), net.bias(1)))
                        .influence(train.x(), train.y(), train.x(), train.y(), 0.01).self();
            }
            case TRACIN -> {
                TrainingDataAttribution.TracIn[] tracIn = {null};
                run(train, new int[] {64, 64, 10}, seed + 1000, null, new Observer() {
                    @Override
                    public void afterEpoch(SimpleMlp net) {
                        if (tracIn[0] == null) {
                            tracIn[0] = new TrainingDataAttribution.TracIn(attribution(net, net.parameters()),
                                    train.x(), train.y(), null, null);
                        }
                        tracIn[0].checkpoint(LR, 1.0 / BATCH);
                    }
                });
                suspicion = tracIn[0].scores().self();
            }
            default -> {
                return ShapleyPruning.run(method == Method.EL2N ? ShapleyPruning.Method.EL2N
                        : ShapleyPruning.Method.IN_RUN_SHAPLEY, flip, flip, seed).flippedRemoved();
            }
        }
        double[] score = suspicion;
        int[] flagged = java.util.stream.IntStream.range(0, n).boxed()
                .sorted((i, j) -> Double.compare(score[j], score[i])).limit(Math.round(flip * n))
                .mapToInt(Integer::intValue).toArray();
        int hits = 0;
        for (int i : flagged) {
            hits += train.flipped()[i] ? 1 : 0;
        }
        return hits / (double) flagged.length;
    }

    /**
     * Leave-k-out: the Pearson correlation between each method's predicted and the actual
     * change of the validation loss when groups of examples are removed.
     *
     * @param method the method
     * @param groups how many random groups
     * @param size the group size
     * @param seed the seed
     * @return the correlation
     */
    public static double leaveOut(Method method, int groups, int size, long seed) {
        Labelled train = train(seed, 0.1);
        Labelled valid = valid(seed);
        int n = train.x().shape().dim(0);
        int[] linear = {64, 10};
        long runSeed = seed + 2000;
        double[] perExample = new double[n];
        switch (method) {
            case INFLUENCE -> {
                SimpleMlp net = run(train, linear, runSeed, null, new Observer() { });
                perExample = attribution(net, net.parameters())
                        .influence(train.x(), train.y(), valid.x(), valid.y(), 1e-3).removal();
            }
            case TRACIN -> {
                TrainingDataAttribution.TracIn[] tracIn = {null};
                run(train, linear, runSeed, null, new Observer() {
                    @Override
                    public void afterEpoch(SimpleMlp net) {
                        if (tracIn[0] == null) {
                            tracIn[0] = new TrainingDataAttribution.TracIn(attribution(net, net.parameters()),
                                    train.x(), train.y(), valid.x(), valid.y());
                        }
                        tracIn[0].checkpoint(LR, 1.0 / BATCH);
                    }
                });
                perExample = tracIn[0].scores().removal();
            }
            case IN_RUN_SHAPLEY -> {
                InRunShapley shapley = new InRunShapley(n);
                run(train, linear, runSeed, null, new Observer() {
                    @Override
                    public void beforeStep(SimpleMlp net, int[] rows, Tensor x, Tensor y) {
                        shapley.record(net, x, y, rows, valid.x(), valid.y(), LR, false);
                    }
                });
                perExample = shapley.scores();
            }
            case EL2N -> {
                DataPruner pruner = new DataPruner(n);
                run(train, linear, runSeed, null, new Observer() {
                    @Override
                    public void afterEpoch(SimpleMlp net) {
                        pruner.record(net.predict(train.x()), train.y());
                    }
                });
                double[] el2n = pruner.el2n();
                for (int i = 0; i < n; i++) {
                    perExample[i] = -el2n[i];   // hard examples are taken to hurt
                }
            }
        }
        double base = validLoss(run(train, linear, runSeed, null, new Observer() { }), valid);
        SplittableRandom r = new SplittableRandom(seed + 7);
        double[] predicted = new double[groups];
        double[] actual = new double[groups];
        for (int g = 0; g < groups; g++) {
            int[] members = r.ints(0, n).distinct().limit(size).toArray();
            boolean[] removed = new boolean[n];
            for (int i : members) {
                removed[i] = true;
                predicted[g] += perExample[i];
            }
            actual[g] = validLoss(run(train, linear, runSeed, removed, new Observer() { }), valid) - base;
        }
        return pearson(predicted, actual);
    }

    /**
     * Runs the linear datamodeling comparison.
     *
     * @param mlp whether to use the 64-64-10 MLP (otherwise the linear 64-10 classifier)
     * @param trakModels models in the TRAK ensemble
     * @param subsets retrained subset models
     * @param targets validation targets
     * @param seed the seed
     * @return per method its linear datamodeling score (higher is better, random is 0)
     */
    public static Map<String, Double> datamodeling(boolean mlp, int trakModels, int subsets, int targets, long seed) {
        Labelled train = train(seed, 0);
        Labelled valid = valid(seed);
        int n = train.x().shape().dim(0);
        int[] sizes = mlp ? new int[] {64, 64, 10} : new int[] {64, 10};
        Tensor tx = valid.x().slice(0, 0, targets);
        Tensor ty = valid.y().slice(0, 0, targets);
        long runSeed = seed + 3000;
        Trak trak1 = new Trak(Math.min(256, n / 2), seed + 11);
        Trak trakM = new Trak(Math.min(256, n / 2), seed + 12);
        double[][] tracIn = new double[targets][n];
        double[][] dot = null;
        double[][] influence = null;
        for (int k = 0; k < trakModels; k++) {
            boolean first = k == 0;
            SimpleMlp net = run(train, sizes, runSeed + 17L * k, null, first ? new Observer() {
                @Override
                public void afterEpoch(SimpleMlp current) {
                    accumulate(tracIn, marginProducts(current, current.parameters(), train, tx, ty), LR);
                }
            } : new Observer() { });
            TrainingDataAttribution att = attribution(net, net.parameters());
            double[][] gTrain = att.perExampleGradients(train.x(), train.y());
            double[] pTrain = correctProbability(net, train.x(), train.y());
            double[][] phi = marginGradients(gTrain, pTrain);
            double[][] phiT = marginGradients(att.perExampleGradients(tx, ty), correctProbability(net, tx, ty));
            double[] q = new double[n];
            for (int j = 0; j < n; j++) {
                q[j] = 1 - pTrain[j];
            }
            trakM.addModel(phi, phiT, q);
            if (first) {
                trak1.addModel(phi, phiT, q);
                dot = marginProducts(net, net.parameters(), train, tx, ty);
                List<Var> head = mlp ? List.of(net.weight(1), net.bias(1)) : net.parameters();
                influence = influence(net, head, train, tx, ty);
            }
        }
        SplittableRandom r = new SplittableRandom(seed + 9);
        boolean[][] member = new boolean[subsets][n];
        double[][] outputs = new double[subsets][];
        for (int s = 0; s < subsets; s++) {
            boolean[] removed = new boolean[n];
            for (int j = 0; j < n; j++) {
                member[s][j] = r.nextBoolean();
                removed[j] = !member[s][j];
            }
            outputs[s] = margins(run(train, sizes, runSeed, removed, new Observer() { }), tx, ty);
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("TRAK (1 model)", Trak.linearDatamodelingScore(trak1.scores(), member, outputs));
        out.put("TRAK (" + trakModels + " models)", Trak.linearDatamodelingScore(trakM.scores(), member, outputs));
        out.put(mlp ? "influence (last layer)" : "influence", Trak.linearDatamodelingScore(influence, member, outputs));
        out.put("TracIn (epoch checkpoints)", Trak.linearDatamodelingScore(tracIn, member, outputs));
        out.put("gradient dot product", Trak.linearDatamodelingScore(dot, member, outputs));
        return out;
    }

    /** {@code p(correct class)} per row. */
    private static double[] correctProbability(SimpleMlp net, Tensor x, Tensor y) {
        float[] logits = Var.noGrad(() -> net.forward(Var.of(x))).value().toArray();
        float[] t = y.toArray();
        int c = logits.length / t.length;
        double[] p = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < c; k++) {
                max = Math.max(max, logits[i * c + k]);
            }
            double z = 0;
            for (int k = 0; k < c; k++) {
                z += Math.exp(logits[i * c + k] - max);
            }
            p[i] = Math.exp(logits[i * c + (int) t[i]] - max) / z;
        }
        return p;
    }

    /** The margin {@code log p / (1 - p)} per row. */
    private static double[] margins(SimpleMlp net, Tensor x, Tensor y) {
        double[] p = correctProbability(net, x, y);
        double[] m = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            double q = Math.min(Math.max(p[i], 1e-9), 1 - 1e-9);
            m[i] = Math.log(q / (1 - q));
        }
        return m;
    }

    /** Margin gradients from loss gradients: {@code grad f = -grad L / (1 - p)}. */
    private static double[][] marginGradients(double[][] lossGradients, double[] p) {
        double[][] out = new double[lossGradients.length][];
        for (int i = 0; i < out.length; i++) {
            double s = -1 / Math.max(1 - p[i], 1e-6);
            out[i] = new double[lossGradients[i].length];
            for (int j = 0; j < out[i].length; j++) {
                out[i][j] = s * lossGradients[i][j];
            }
        }
        return out;
    }

    /** {@code -grad f_t . grad L_j}: the first-order margin change of target t from a step on j. */
    private static double[][] marginProducts(SimpleMlp net, List<Var> params, Labelled train, Tensor tx, Tensor ty) {
        TrainingDataAttribution att = attribution(net, params);
        double[][] gTrain = att.perExampleGradients(train.x(), train.y());
        double[][] fT = marginGradients(att.perExampleGradients(tx, ty), correctProbability(net, tx, ty));
        double[][] out = new double[fT.length][gTrain.length];
        for (int t = 0; t < fT.length; t++) {
            for (int j = 0; j < gTrain.length; j++) {
                out[t][j] = -dot(fT[t], gTrain[j]);
            }
        }
        return out;
    }

    /** {@code -grad f_t H^-1 grad L_j} with the damped Hessian of the mean training loss. */
    private static double[][] influence(SimpleMlp net, List<Var> params, Labelled train, Tensor tx, Tensor ty) {
        TrainingDataAttribution att = attribution(net, params);
        double[] inv = Linalg.inverse(att.hessian(train.x(), train.y(), 1e-3)).toDoubleArray();
        double[][] gTrain = att.perExampleGradients(train.x(), train.y());
        double[][] fT = marginGradients(att.perExampleGradients(tx, ty), correctProbability(net, tx, ty));
        int p = fT[0].length;
        double[][] out = new double[fT.length][gTrain.length];
        for (int t = 0; t < fT.length; t++) {
            double[] h = new double[p];
            for (int i = 0; i < p; i++) {
                double s = 0;
                for (int j = 0; j < p; j++) {
                    s += inv[i * p + j] * fT[t][j];
                }
                h[i] = s;
            }
            for (int j = 0; j < gTrain.length; j++) {
                out[t][j] = -dot(h, gTrain[j]);
            }
        }
        return out;
    }

    private static void accumulate(double[][] into, double[][] add, double scale) {
        for (int i = 0; i < into.length; i++) {
            for (int j = 0; j < into[i].length; j++) {
                into[i][j] += scale * add[i][j];
            }
        }
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static double validLoss(SimpleMlp net, Labelled valid) {
        return Functional.crossEntropy(net.forward(Var.of(valid.x())), valid.y()).value().item();
    }

    private static double pearson(double[] a, double[] b) {
        double ma = Arrays.stream(a).average().orElse(0);
        double mb = Arrays.stream(b).average().orElse(0);
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < a.length; i++) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        return sab / Math.sqrt(Math.max(saa * sbb, 1e-300));
    }
}
