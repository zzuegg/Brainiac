package io.github.zzuegg.brainiac.experimental.parallel;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.DataParallel;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * ZeRO-sharded data parallelism (stages 1, 2 and 3, the last also known as fully sharded data
 * parallelism, FSDP) over {@code N} simulated ranks, wrapping any core {@link Optimizer}.
 *
 * <p>Plain data parallelism ({@link DataParallel}) keeps a full copy of the parameters, the
 * gradients and the optimizer state on every rank. ZeRO partitions the flattened parameter
 * vector into {@code N} equal shards, rank {@code r} owning shard {@code r}:
 * <ul>
 *   <li>stage 1 shards the optimizer state: the gradients are reduce-scattered (the first half
 *       of an all-reduce), each rank updates only its shard with its own optimizer, and the
 *       updated shards are all-gathered (the second half), so the traffic is that of plain data
 *       parallelism, {@code 2 Psi};</li>
 *   <li>stage 2 also shards the gradients: each rank keeps only the summed gradient of its
 *       shard, freeing the full gradient buffer bucket by bucket (the same {@code 2 Psi});</li>
 *   <li>stage 3 also shards the parameters: each layer is all-gathered just in time for the
 *       forward pass, again for the backward pass, and freed afterwards, so a rank holds its
 *       shard plus the largest layer (plus one prefetched layer); {@code 3 Psi} bytes.</li>
 * </ul>
 * For an elementwise optimizer such as AdamW every stage computes exactly the step of
 * {@link DataParallel}. The optimizer's per-parameter state size is given in bytes per
 * parameter (8 for fp32 Adam, about 1 for 4-bit states), so the memory accounting of the paper,
 * {@code (2 + 2 + K) Psi} for mixed precision, here {@code (4 + 4 + K) Psi} in fp32, can be
 * reproduced per stage.
 *
 * <p>Papers: Rajbhandari et al., "ZeRO", <a href="https://arxiv.org/abs/1910.02054">arXiv:1910.02054</a>;
 * Zhao et al., "PyTorch FSDP", <a href="https://arxiv.org/abs/2304.11277">arXiv:2304.11277</a>.
 */
public final class ShardedDataParallel {

    /**
     * Per-rank memory by category, in bytes.
     *
     * @param parameters parameter bytes held (stage 3: the shard plus the gathered layers)
     * @param gradients gradient bytes held
     * @param optimizerStates optimizer state bytes held
     */
    public record Memory(long parameters, long gradients, long optimizerStates) {

        /** @return the total */
        public long total() {
            return parameters + gradients + optimizerStates;
        }
    }

    private final int ranks;
    private final int stage;
    private final List<Var> parameters;
    private final DataParallel.Loss loss;
    private final List<Optimizer> optimizers = new ArrayList<>();
    private final List<Var> shards = new ArrayList<>();
    private final List<List<Var>> replicas = new ArrayList<>();
    private final double stateBytesPerParameter;
    private final Collectives collectives;
    private final int total;
    private final int padded;
    private final int largestLayer;

    /**
     * Creates the trainer; each rank gets its own optimizer from the factory.
     *
     * @param ranks {@code N}
     * @param stage 1, 2 or 3
     * @param parameters the model's parameters, updated in place each step
     * @param loss the loss of a share of the batch (the core data-parallel interface)
     * @param optimizer makes one optimizer per rank
     * @param stateBytesPerParameter optimizer state bytes per parameter, for the accounting
     */
    public ShardedDataParallel(int ranks, int stage, List<Var> parameters, DataParallel.Loss loss,
            Supplier<Optimizer> optimizer, double stateBytesPerParameter) {
        if (stage < 1 || stage > 3) {
            throw new IllegalArgumentException("stage must be 1, 2 or 3, got " + stage);
        }
        this.ranks = ranks;
        this.stage = stage;
        this.parameters = List.copyOf(parameters);
        this.loss = Objects.requireNonNull(loss, "loss");
        this.stateBytesPerParameter = stateBytesPerParameter;
        this.collectives = new Collectives(ranks);
        int n = 0;
        int largest = 0;
        for (Var p : this.parameters) {
            n += Math.toIntExact(p.value().size());
            largest = Math.max(largest, Math.toIntExact(p.value().size()));
        }
        this.total = n;
        this.padded = (n + ranks - 1) / ranks * ranks;
        this.largestLayer = largest;
        float[] flat = flatten(values());
        int chunk = padded / ranks;
        for (int r = 0; r < ranks; r++) {
            optimizers.add(optimizer.get());
            float[] mine = new float[chunk];
            System.arraycopy(flat, r * chunk, mine, 0, chunk);
            shards.add(Var.param(Tensor.of(Shape.of(chunk), mine)));
            List<Var> copy = new ArrayList<>();
            for (Var p : this.parameters) {
                copy.add(Var.param(p.value().copy()));
            }
            replicas.add(copy);
        }
    }

    /** @return the process group, whose counters hold the traffic so far */
    public Collectives collectives() {
        return collectives;
    }

    /** @return the number of parameters {@code Psi} */
    public int parameterCount() {
        return total;
    }

    private List<Tensor> values() {
        List<Tensor> v = new ArrayList<>();
        for (Var p : parameters) {
            v.add(p.value());
        }
        return v;
    }

    private float[] flatten(List<Tensor> tensors) {
        float[] out = new float[padded];
        int at = 0;
        for (Tensor t : tensors) {
            float[] a = t.toArray();
            System.arraycopy(a, 0, out, at, a.length);
            at += a.length;
        }
        return out;
    }

    private void unflatten(float[] flat, List<Var> into) {
        int at = 0;
        for (Var p : into) {
            int n = Math.toIntExact(p.value().size());
            float[] a = new float[n];
            System.arraycopy(flat, at, a, 0, n);
            p.assign(Tensor.of(p.shape(), a));
            at += n;
        }
    }

    /**
     * One step on a batch, split over the ranks by rows.
     *
     * @param inputs {@code [batch, ...]}
     * @param targets {@code [batch, ...]}
     * @return the batch's mean loss
     */
    public float step(Tensor inputs, Tensor targets) {
        int rows = inputs.shape().dim(0);
        int chunk = padded / ranks;
        List<float[]> out = Collectives.run(ranks, r -> {
            int from = (int) ((long) rows * r / ranks);
            int to = (int) ((long) rows * (r + 1) / ranks);
            List<Var> replica = replicas.get(r);
            if (stage == 3) {
                // just-in-time gather for the forward pass (layer by layer in a real system;
                // the bytes are the same)
                unflatten(collectives.allGather(r, shards.get(r).value().toArray()), replica);
            }
            for (Var p : replica) {
                p.zeroGrad();
            }
            float value = 0;
            List<Tensor> grads = new ArrayList<>();
            if (to > from) {
                Var l = loss.loss(replica, inputs.slice(0, from, to), targets.slice(0, from, to));
                float weight = (float) (to - from) / rows;
                l.mul(weight).backward();
                value = l.value().item() * weight;
            }
            for (Var p : replica) {
                grads.add(p.grad() == null ? Tensor.zeros(p.shape()) : p.grad());
            }
            if (stage == 3) {
                // the backward pass gathers each layer again
                collectives.allGather(r, shards.get(r).value().toArray());
            }
            // every stage needs only its shard of the summed gradient: a reduce-scatter (stage 1
            // still holds the full gradient buffer until then, stages 2 and 3 free it bucket by bucket)
            float[] mine = collectives.reduceScatter(r, flatten(grads));
            Var shard = shards.get(r);
            shard.setGrad(Tensor.of(Shape.of(chunk), mine));
            optimizers.get(r).step(List.of(shard));
            if (stage < 3) {
                unflatten(collectives.allGather(r, shard.value().toArray()), replica);
            }
            return new float[] {value};
        });
        // the host's copy of the parameters, from the shards (not counted as traffic)
        float[] flat = new float[padded];
        for (int r = 0; r < ranks; r++) {
            System.arraycopy(shards.get(r).value().toArray(), 0, flat, r * chunk, chunk);
        }
        unflatten(flat, parameters);
        float sum = 0;
        for (float[] o : out) {
            sum += o[0];
        }
        return sum;
    }

    /**
     * The bytes one rank holds, by category (fp32 parameters and gradients).
     *
     * @return the accounting for this trainer's stage
     */
    public Memory memoryPerRank() {
        return memory(total, ranks, stage, stateBytesPerParameter, largestLayer);
    }

    /**
     * The per-rank memory of ZeRO stage {@code stage} (0 is plain data parallelism).
     *
     * @param parameters {@code Psi}
     * @param ranks {@code N}
     * @param stage 0 to 3
     * @param stateBytesPerParameter optimizer state bytes per parameter
     * @param largestLayer parameters of the largest layer, gathered in full under stage 3
     * @return the accounting
     */
    public static Memory memory(long parameters, int ranks, int stage, double stateBytesPerParameter,
            long largestLayer) {
        long shard = (parameters + ranks - 1) / ranks;
        long p = stage >= 3 ? 4 * (shard + 2 * largestLayer) : 4 * parameters;
        long g = stage >= 2 ? 4 * (shard + (stage >= 3 ? largestLayer : 0)) : 4 * parameters;
        long s = (long) Math.ceil(stateBytesPerParameter * (stage >= 1 ? shard : parameters));
        return new Memory(p, g, s);
    }
}
