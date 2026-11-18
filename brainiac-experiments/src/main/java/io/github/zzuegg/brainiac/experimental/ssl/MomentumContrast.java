package io.github.zzuegg.brainiac.experimental.ssl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * MoCo, momentum contrast (He et al., CVPR 2020; v3: Chen et al., ICCV 2021). SimCLR needs a
 * large batch for many negatives; MoCo keeps a FIFO <em>queue</em> of keys from past batches as
 * negatives, encoded by a slowly updated momentum copy of the encoder (use
 * {@code experimental.train.MomentumTeacher}) so the queued keys stay consistent. The v2 loss
 * is InfoNCE of each query against its own key (positive) and the whole queue (negatives),
 * with temperature 0.2; the batch's keys are then {@linkplain #enqueue enqueued}, oldest out.
 *
 * <p>MoCo v3 ({@link #symmetricV3}) drops the queue: in-batch keys are the negatives, the
 * query branch gets a predictor, and the loss is symmetrised over the two views and scaled by
 * {@code 2 tau}.
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1911.05722">arXiv:1911.05722</a>,
 * <a href="https://arxiv.org/abs/2104.02057">arXiv:2104.02057</a>.
 */
public final class MomentumContrast {

    private final float[] queue;
    private final int dim;
    private final int capacity;
    private final float temperature;
    private int head;
    private int filled;

    /**
     * Creates a queue filled with random unit keys (as in the reference code).
     *
     * @param dim the key width
     * @param capacity the queue length, 65536 in the paper; a few hundred here
     * @param temperature the InfoNCE temperature, 0.2 for v2
     * @param seed the seed of the initial keys
     */
    public MomentumContrast(int dim, int capacity, float temperature, long seed) {
        this.dim = dim;
        this.capacity = capacity;
        this.temperature = temperature;
        this.queue = new float[capacity * dim];
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < capacity; i++) {
            double s = 0;
            for (int j = 0; j < dim; j++) {
                queue[i * dim + j] = (float) r.nextGaussian();
                s += queue[i * dim + j] * queue[i * dim + j];
            }
            for (int j = 0; j < dim; j++) {
                queue[i * dim + j] /= (float) Math.sqrt(s);
            }
        }
    }

    /** @return the queued keys {@code [capacity, dim]}, unit rows */
    public Tensor queue() {
        return Tensor.of(Shape.of(capacity, dim), queue.clone());
    }

    /** @return how many real keys have been enqueued, capped at the capacity */
    public int filled() {
        return filled;
    }

    /**
     * MoCo v2's InfoNCE: logits {@code [q.k+, q.queue] / tau}, the positive at index 0.
     *
     * @param queries the query encoder's outputs {@code [n, dim]} (normalised here)
     * @param keys the momentum encoder's outputs for the other view {@code [n, dim]} (held fixed)
     * @return the mean cross-entropy
     */
    public Var loss(Var queries, Tensor keys) {
        int n = queries.shape().dim(0);
        Var q = SelfSupervisedLosses.normalize(queries);
        Tensor k = SelfSupervisedLosses.normalize(Var.of(keys)).value();
        Var positive = q.mul(k).sum(-1, true);
        Var negative = q.matmul(Var.of(queue().transpose()));
        Var logits = Var.concat(1, positive, negative).div(temperature);
        return Functional.crossEntropy(logits, Tensor.zeros(Shape.of(n)));
    }

    /**
     * Pushes a batch of keys into the queue, replacing the oldest.
     *
     * @param keys {@code [n, dim]} (normalised here)
     */
    public void enqueue(Tensor keys) {
        float[] k = SelfSupervisedLosses.normalize(Var.of(keys)).value().toArray();
        int n = keys.shape().dim(0);
        for (int i = 0; i < n; i++) {
            System.arraycopy(k, i * dim, queue, head * dim, dim);
            head = (head + 1) % capacity;
            filled = Math.min(capacity, filled + 1);
        }
    }

    /**
     * MoCo v3's symmetrised in-batch contrastive loss:
     * {@code 2 tau (ctr(q1, k2) + ctr(q2, k1))}, where {@code ctr} is InfoNCE with the other
     * rows of the batch as negatives.
     *
     * @param q1 predictor outputs of view 1 {@code [n, d]}
     * @param q2 predictor outputs of view 2
     * @param k1 momentum keys of view 1 (held fixed)
     * @param k2 momentum keys of view 2 (held fixed)
     * @param temperature the temperature, 0.2 in the paper
     * @return the loss
     */
    public static Var symmetricV3(Var q1, Var q2, Tensor k1, Tensor k2, float temperature) {
        return contrast(q1, k2, temperature).add(contrast(q2, k1, temperature)).mul(2 * temperature);
    }

    private static Var contrast(Var q, Tensor k, float temperature) {
        int n = q.shape().dim(0);
        Var qn = SelfSupervisedLosses.normalize(q);
        Tensor kn = SelfSupervisedLosses.normalize(Var.of(k)).value();
        Var logits = qn.matmul(Var.of(kn.transpose())).div(temperature);
        return Functional.crossEntropy(logits, Tensor.arange(n));
    }
}
