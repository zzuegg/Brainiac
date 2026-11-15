package io.github.zzuegg.brainiac.rl;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * A fixed-size memory of past transitions; the oldest is overwritten when full. Sampling
 * random batches from it breaks the correlation between consecutive steps.
 */
public class ReplayBuffer {

    private final int capacity;
    private final int observationSize;
    private final int actionSize;
    final float[] observations;
    final float[] actions;
    final float[] rewards;
    final float[] next;
    final float[] notDone;
    final float[] discounts;
    private int size;
    private int position;

    /**
     * @param capacity how many transitions it holds
     * @param observationSize values per observation
     * @param actionSize values per action (1 for discrete actions)
     */
    public ReplayBuffer(int capacity, int observationSize, int actionSize) {
        if (capacity < 1) {
            throw new IllegalArgumentException("capacity must be at least 1");
        }
        this.capacity = capacity;
        this.observationSize = observationSize;
        this.actionSize = actionSize;
        observations = new float[capacity * observationSize];
        actions = new float[capacity * actionSize];
        rewards = new float[capacity];
        next = new float[capacity * observationSize];
        notDone = new float[capacity];
        discounts = new float[capacity];
    }

    /**
     * Stores a transition.
     *
     * @param t the transition
     * @return the slot it was stored in
     */
    public int add(Transition t) {
        int i = position;
        System.arraycopy(t.observation(), 0, observations, i * observationSize, observationSize);
        System.arraycopy(t.action(), 0, actions, i * actionSize, actionSize);
        rewards[i] = t.reward();
        System.arraycopy(t.nextObservation(), 0, next, i * observationSize, observationSize);
        notDone[i] = t.terminated() ? 0 : 1;
        discounts[i] = t.discount();
        position = (position + 1) % capacity;
        size = Math.min(size + 1, capacity);
        return i;
    }

    /** @return the number of stored transitions */
    public int size() {
        return size;
    }

    /** @return the most transitions it holds */
    public int capacity() {
        return capacity;
    }

    /**
     * Draws a batch uniformly at random (with replacement).
     *
     * @param batchSize the batch size
     * @param random the random source
     * @param device where to put the tensors
     * @return the batch
     */
    public Batch sample(int batchSize, SplittableRandom random, Device device) {
        if (size == 0) {
            throw new IllegalStateException("the buffer is empty");
        }
        int[] idx = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            idx[i] = random.nextInt(size);
        }
        float[] w = new float[batchSize];
        java.util.Arrays.fill(w, 1);
        return gather(idx, w, device);
    }

    Batch gather(int[] idx, float[] weights, Device device) {
        int b = idx.length;
        float[] o = new float[b * observationSize];
        float[] a = new float[b * actionSize];
        float[] r = new float[b];
        float[] n = new float[b * observationSize];
        float[] d = new float[b];
        float[] g = new float[b];
        for (int i = 0; i < b; i++) {
            int j = idx[i];
            System.arraycopy(observations, j * observationSize, o, i * observationSize, observationSize);
            System.arraycopy(actions, j * actionSize, a, i * actionSize, actionSize);
            r[i] = rewards[j];
            System.arraycopy(next, j * observationSize, n, i * observationSize, observationSize);
            d[i] = notDone[j];
            g[i] = discounts[j];
        }
        return new Batch(Tensor.of(Shape.of(b, observationSize), o).to(device),
                Tensor.of(Shape.of(b, actionSize), a).to(device), Tensor.of(Shape.of(b, 1), r).to(device),
                Tensor.of(Shape.of(b, observationSize), n).to(device), Tensor.of(Shape.of(b, 1), d).to(device),
                Tensor.of(Shape.of(b, 1), g).to(device), Tensor.of(Shape.of(b, 1), weights).to(device), idx);
    }

    /**
     * A sampled batch; every tensor has one row per transition.
     *
     * @param observations {@code [batch, observationSize]}
     * @param actions {@code [batch, actionSize]}
     * @param rewards {@code [batch, 1]}
     * @param nextObservations {@code [batch, observationSize]}
     * @param notDone {@code [batch, 1]}: 0 where the next state is terminal, else 1
     * @param discounts {@code [batch, 1]}
     * @param weights {@code [batch, 1]} importance weights (all 1 for uniform sampling)
     * @param indices the slots the transitions came from
     */
    public record Batch(Tensor observations, Tensor actions, Tensor rewards, Tensor nextObservations,
            Tensor notDone, Tensor discounts, Tensor weights, int[] indices) {
    }
}
