package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.nn.LoraAdapterDense;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Exact unlearning by SISA (Bourtoule et al., IEEE S&amp;P 2021,
 * <a href="https://arxiv.org/abs/1912.03817">arXiv:1912.03817</a>): the training set is
 * split into disjoint shards, one model per shard, each trained on its shard's slices one
 * after another with a checkpoint after every slice; predictions average the shard models'
 * outputs. Deleting a point restores its shard from the checkpoint before the point's slice
 * and retrains from there without it, so the result is exactly a model never trained on it.
 * {@link #cost} counts example-epochs of training, the compute measure of retraining.
 *
 * <p>{@link S3t} is the parameter-efficient variant (Chowdhury et al., ICLR 2025,
 * <a href="https://arxiv.org/abs/2406.16257">arXiv:2406.16257</a>).
 */
public final class ShardedUnlearning {

    private final Supplier<Model> factory;
    private final Dataset data;
    private final int epochsPerSlice;
    private final long seed;
    private final Model[] models;
    private final int[][][] slices;
    private final List<List<List<Tensor>>> checkpoints;
    private final boolean[] deleted;
    private final int[] shardOf;
    private final int[] sliceOf;
    private long cost;
    private long trainingCost;

    private ShardedUnlearning(Supplier<Model> factory, Dataset data, int shards, int sliceCount, int epochsPerSlice,
            long seed) {
        this.factory = Objects.requireNonNull(factory);
        this.data = Objects.requireNonNull(data);
        this.epochsPerSlice = epochsPerSlice;
        this.seed = seed;
        this.models = new Model[shards];
        this.slices = new int[shards][sliceCount][];
        this.checkpoints = new ArrayList<>();
        this.deleted = new boolean[data.size()];
        this.shardOf = new int[data.size()];
        this.sliceOf = new int[data.size()];
        int[] order = permutation(data.size(), new SplittableRandom(seed));
        List<List<List<Integer>>> rows = new ArrayList<>();
        for (int s = 0; s < shards; s++) {
            List<List<Integer>> sl = new ArrayList<>();
            for (int r = 0; r < sliceCount; r++) {
                sl.add(new ArrayList<>());
            }
            rows.add(sl);
        }
        for (int k = 0; k < order.length; k++) {
            int s = k % shards;
            int r = (k / shards) % sliceCount;
            rows.get(s).get(r).add(order[k]);
            shardOf[order[k]] = s;
            sliceOf[order[k]] = r;
        }
        for (int s = 0; s < shards; s++) {
            for (int r = 0; r < sliceCount; r++) {
                slices[s][r] = rows.get(s).get(r).stream().mapToInt(Integer::intValue).toArray();
            }
        }
    }

    /**
     * Trains SISA.
     *
     * @param factory creates an untrained model (softmax output)
     * @param data the training data
     * @param shards the number of shards
     * @param slices slices per shard
     * @param epochsPerSlice epochs of each incremental stage
     * @param seed the seed
     * @return the trained ensemble
     */
    public static ShardedUnlearning sisa(Supplier<Model> factory, Dataset data, int shards, int slices,
            int epochsPerSlice, long seed) {
        ShardedUnlearning u = new ShardedUnlearning(factory, data, shards, slices, epochsPerSlice, seed);
        for (int s = 0; s < shards; s++) {
            u.models[s] = factory.get();
            List<List<Tensor>> cps = new ArrayList<>();
            cps.add(snapshot(u.models[s]));
            for (int r = 0; r < slices; r++) {
                u.stage(s, r);
                cps.add(snapshot(u.models[s]));
            }
            u.checkpoints.add(cps);
        }
        u.trainingCost = u.cost;
        u.cost = 0;
        return u;
    }

    private void stage(int s, int r) {
        List<Integer> rows = new ArrayList<>();
        for (int k = 0; k <= r; k++) {
            for (int i : slices[s][k]) {
                if (!deleted[i]) {
                    rows.add(i);
                }
            }
        }
        if (rows.isEmpty()) {
            return;
        }
        Dataset subset = subset(data, rows);
        Trainer.of(models[s]).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.01f)).batchSize(32)
                .epochs(epochsPerSlice).seed(seed + s * 31L + r).fit(subset);
        cost += (long) epochsPerSlice * rows.size();
    }

    /**
     * Deletes a training example exactly.
     *
     * @param row the example's index in the training data
     */
    public void delete(int row) {
        if (deleted[row]) {
            return;
        }
        deleted[row] = true;
        int s = shardOf[row];
        int r = sliceOf[row];
        load(models[s], checkpoints.get(s).get(r));
        for (int k = r; k < slices[s].length; k++) {
            stage(s, k);
            checkpoints.get(s).set(k + 1, snapshot(models[s]));
        }
    }

    /** @return example-epochs spent on retraining after deletions */
    public long cost() {
        return cost;
    }

    /** @return example-epochs of the initial training */
    public long trainingCost() {
        return trainingCost;
    }

    /** @return the shard models */
    public List<Model> models() {
        return List.of(models);
    }

    /**
     * The averaged class probabilities.
     *
     * @param x inputs
     * @return {@code [n, classes]}
     */
    public Tensor predict(Tensor x) {
        Tensor sum = null;
        for (Model m : models) {
            Tensor p = m.predict(x);
            sum = sum == null ? p : sum.add(p);
        }
        return sum.mul(1f / models.length);
    }

    /**
     * Accuracy of the averaged prediction.
     *
     * @param test the data
     * @return the share classified correctly
     */
    public double accuracy(Dataset test) {
        return accuracy(predict(test.features()), test);
    }

    static double accuracy(Tensor probabilities, Dataset test) {
        int[] pred = probabilities.argmax(1).toIntArray();
        int[] truth = test.targets().argmax(1).toIntArray();
        int ok = 0;
        for (int i = 0; i < pred.length; i++) {
            ok += pred[i] == truth[i] ? 1 : 0;
        }
        return ok / (double) pred.length;
    }

    static Dataset subset(Dataset data, List<Integer> rows) {
        float[][] x = new float[rows.size()][];
        float[][] y = new float[rows.size()][];
        for (int i = 0; i < x.length; i++) {
            x[i] = data.featuresOf(rows.get(i));
            y[i] = data.targetsOf(rows.get(i));
        }
        return Dataset.of(x, y);
    }

    static List<Tensor> snapshot(Model m) {
        List<Tensor> out = new ArrayList<>();
        for (Var p : m.parameters()) {
            out.add(p.value());
        }
        return out;
    }

    static void load(Model m, List<Tensor> values) {
        List<Var> p = m.parameters();
        for (int i = 0; i < p.size(); i++) {
            p.get(i).assign(values.get(i));
        }
    }

    static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    /**
     * Sequence-aware sharded sliced training (S3T). A frozen pretrained two-layer base
     * ({@link Dense} hidden layer, {@link Dense} softmax output) gets one pair of
     * {@link LoraAdapterDense} adapters per slice position, trained one after another, each
     * on its own data slice with the earlier adapters active and frozen. Deleting a point
     * of slice {@code s} switches off the adapter trained on {@code s} and every later one,
     * which is exact and costs no training. Several sequences (cyclic orders of the slices)
     * are trained so that after deletions some sequence keeps a long untouched prefix; the
     * model predicts with the sequence whose usable prefix is longest.
     */
    public static final class S3t {

        private final Dense hidden;
        private final Dense output;
        private final int[][] order;
        private final LoraAdapterDense[][][] adapters;
        private final int[] usable;
        private final int[] sliceOf;

        private S3t(Dense hidden, Dense output, int[][] order, LoraAdapterDense[][][] adapters, int[] sliceOf) {
            this.hidden = hidden;
            this.output = output;
            this.order = order;
            this.adapters = adapters;
            this.sliceOf = sliceOf;
            this.usable = new int[order.length];
            java.util.Arrays.fill(usable, order[0].length);
        }

        /**
         * Trains S3T.
         *
         * @param base a pretrained model of two dense layers, frozen
         * @param data the deletable training data
         * @param slices the number of slices (adapter positions)
         * @param sequences the number of slice orders trained
         * @param rank the adapters' rank
         * @param epochs epochs per adapter
         * @param seed the seed
         * @return the trained model
         */
        public static S3t train(Model base, Dataset data, int slices, int sequences, int rank, int epochs,
                long seed) {
            Dense hidden = (Dense) base.layers().get(0);
            Dense output = (Dense) base.layers().get(1);
            SplittableRandom r = new SplittableRandom(seed);
            int[] perm = permutation(data.size(), r);
            int[] sliceOf = new int[data.size()];
            List<List<Integer>> rows = new ArrayList<>();
            for (int s = 0; s < slices; s++) {
                rows.add(new ArrayList<>());
            }
            for (int k = 0; k < perm.length; k++) {
                sliceOf[perm[k]] = k % slices;
                rows.get(k % slices).add(perm[k]);
            }
            int[][] order = new int[sequences][slices];
            for (int p = 0; p < sequences; p++) {
                for (int l = 0; l < slices; l++) {
                    order[p][l] = (l + p * Math.max(1, slices / sequences)) % slices;
                }
            }
            LoraAdapterDense[][][] adapters = new LoraAdapterDense[sequences][slices][2];
            S3t model = new S3t(hidden, output, order, adapters, sliceOf);
            for (int p = 0; p < sequences; p++) {
                for (int l = 0; l < slices; l++) {
                    adapters[p][l][0] = LoraAdapterDense.lora(hidden, rank, 2f * rank, r);
                    adapters[p][l][1] = LoraAdapterDense.lora(output, rank, 2f * rank, r);
                    Dataset slice = subset(data, rows.get(order[p][l]));
                    List<Var> params = new ArrayList<>(adapters[p][l][0].parameters());
                    params.addAll(adapters[p][l][1].parameters());
                    Optimizer opt = Optimizer.adam(0.01f);
                    for (int e = 0; e < epochs; e++) {
                        int[] idx = permutation(slice.size(), r);
                        for (int b = 0; b < slice.size(); b += 32) {
                            List<Integer> batch = new ArrayList<>();
                            for (int i = b; i < Math.min(slice.size(), b + 32); i++) {
                                batch.add(idx[i]);
                            }
                            Dataset bd = subset(slice, batch);
                            Loss.CROSS_ENTROPY.compute(model.forward(p, l + 1, Var.of(bd.features()), true),
                                    bd.targets()).backward();
                            opt.step(params);
                        }
                    }
                }
            }
            return model;
        }

        private Var forward(int p, int depth, Var x, boolean training) {
            Var w0 = Var.of(hidden.weight().value());
            Var w1 = Var.of(output.weight().value());
            for (int l = 0; l < depth; l++) {
                w0 = w0.add(delta(adapters[p][l][0]));
                w1 = w1.add(delta(adapters[p][l][1]));
            }
            Var h = hidden.activation().forward(x.matmul(w0).add(hidden.bias().value()), training);
            return output.activation().forward(h.matmul(w1).add(output.bias().value()), training);
        }

        private static Var delta(LoraAdapterDense a) {
            return a.weight().sub(a.frozen());
        }

        /**
         * Deletes a training example by switching off its slice's adapter and all later
         * ones in every sequence.
         *
         * @param row the example's index in the training data
         */
        public void delete(int row) {
            int s = sliceOf[row];
            for (int p = 0; p < order.length; p++) {
                for (int l = 0; l < order[p].length; l++) {
                    if (order[p][l] == s) {
                        usable[p] = Math.min(usable[p], l);
                    }
                }
            }
        }

        /** @return the longest usable prefix over the sequences (0: only the base is left) */
        public int depth() {
            int best = 0;
            for (int u : usable) {
                best = Math.max(best, u);
            }
            return best;
        }

        /**
         * Class probabilities with the best surviving sequence.
         *
         * @param x inputs
         * @return {@code [n, classes]}
         */
        public Tensor predict(Tensor x) {
            int best = 0;
            for (int p = 1; p < usable.length; p++) {
                best = usable[p] > usable[best] ? p : best;
            }
            return forward(best, usable[best], Var.of(x), false).value();
        }

        /**
         * Accuracy.
         *
         * @param test the data
         * @return the share classified correctly
         */
        public double accuracy(Dataset test) {
            return ShardedUnlearning.accuracy(predict(test.features()), test);
        }
    }
}
