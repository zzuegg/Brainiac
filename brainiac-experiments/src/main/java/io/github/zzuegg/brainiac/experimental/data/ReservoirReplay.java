package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * An experience-replay memory for continual learning (Rolnick et al. 2019,
 * <a href="https://arxiv.org/abs/1811.11682">arXiv:1811.11682</a>; Chaudhry et al. 2019,
 * <a href="https://arxiv.org/abs/1902.10486">arXiv:1902.10486</a>): a fixed number of past
 * samples kept by <em>reservoir sampling</em>, so every sample seen so far is in memory with
 * equal probability whatever the stream's length. Mixing a few stored samples into every
 * new batch is a strong, simple defence against forgetting.
 */
public final class ReservoirReplay {

    private final int capacity;
    private final float[][] features;
    private final float[][] targets;
    private final SplittableRandom random;
    private long seen;
    private int size;

    /**
     * @param capacity the most samples kept
     * @param seed the random seed
     */
    public ReservoirReplay(int capacity, long seed) {
        this.capacity = capacity;
        this.features = new float[capacity][];
        this.targets = new float[capacity][];
        this.random = new SplittableRandom(seed);
    }

    /** @return the samples held */
    public int size() {
        return size;
    }

    /** @return the samples offered so far */
    public long seen() {
        return seen;
    }

    /**
     * Offers a batch to the memory.
     *
     * @param x {@code [batch, features]}
     * @param y {@code [batch, targets]} or {@code [batch]}
     */
    public void add(Tensor x, Tensor y) {
        int n = x.shape().dim(0);
        int f = x.shape().dim(1);
        int t = Math.toIntExact(y.size() / n);
        float[] xs = x.toArray();
        float[] ys = y.toArray();
        for (int i = 0; i < n; i++) {
            seen++;
            int slot = size < capacity ? size++ : (int) random.nextLong(seen);
            if (slot < capacity) {
                features[slot] = java.util.Arrays.copyOfRange(xs, i * f, (i + 1) * f);
                targets[slot] = java.util.Arrays.copyOfRange(ys, i * t, (i + 1) * t);
            }
        }
    }

    /**
     * Draws stored samples, with replacement.
     *
     * @param count how many
     * @return {@code [x, y]} with {@code y} shaped {@code [count, targets]}
     */
    public Tensor[] sample(int count) {
        int f = features[0].length;
        int t = targets[0].length;
        float[] xs = new float[count * f];
        float[] ys = new float[count * t];
        for (int i = 0; i < count; i++) {
            int k = random.nextInt(size);
            System.arraycopy(features[k], 0, xs, i * f, f);
            System.arraycopy(targets[k], 0, ys, i * t, t);
        }
        return new Tensor[] {Tensor.of(Shape.of(count, f), xs), Tensor.of(Shape.of(count, t), ys)};
    }

    /**
     * A new batch with {@code count} replayed samples appended (nothing appended while the
     * memory is empty). Targets keep the new batch's layout ({@code [batch]} ids or
     * {@code [batch, targets]}).
     *
     * @param x the new inputs
     * @param y the new targets
     * @param count replayed samples to add
     * @return {@code [x', y']}
     */
    public Tensor[] mix(Tensor x, Tensor y, int count) {
        if (size == 0 || count == 0) {
            return new Tensor[] {x, y};
        }
        Tensor[] old = sample(count);
        Tensor oldY = y.rank() == 1 ? old[1].reshape(count) : old[1];
        return new Tensor[] {Tensor.concat(0, x, old[0]), Tensor.concat(0, y, oldY)};
    }

    /** @return the memory as a core {@link Dataset}, for example to rehearse with a Trainer */
    public Dataset toDataset() {
        return Dataset.of(java.util.Arrays.copyOf(features, size), java.util.Arrays.copyOf(targets, size));
    }
}
