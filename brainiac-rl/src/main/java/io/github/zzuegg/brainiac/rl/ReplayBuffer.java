package io.github.zzuegg.brainiac.rl;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * A fixed-size memory of past transitions; the oldest is overwritten when full. Sampling
 * random batches from it breaks the correlation between consecutive steps. Its storage grows
 * as it fills, so a large capacity costs memory only once it is used.
 */
public class ReplayBuffer {

    private final int capacity;
    private final int observationSize;
    private final int actionSize;
    private float[] observations;
    private float[] actions;
    private float[] rewards;
    private float[] next;
    private float[] notDone;
    private float[] discounts;
    private int allocated;
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
        grow(Math.min(capacity, 1024));
    }

    private void grow(int slots) {
        observations = java.util.Arrays.copyOf(observations == null ? new float[0] : observations, Math.multiplyExact(slots, observationSize));
        actions = java.util.Arrays.copyOf(actions == null ? new float[0] : actions, Math.multiplyExact(slots, actionSize));
        rewards = java.util.Arrays.copyOf(rewards == null ? new float[0] : rewards, slots);
        next = java.util.Arrays.copyOf(next == null ? new float[0] : next, Math.multiplyExact(slots, observationSize));
        notDone = java.util.Arrays.copyOf(notDone == null ? new float[0] : notDone, slots);
        discounts = java.util.Arrays.copyOf(discounts == null ? new float[0] : discounts, slots);
        allocated = slots;
    }

    /**
     * Stores a transition.
     *
     * @param t the transition
     * @return the slot it was stored in
     */
    public int add(Transition t) {
        int i = position;
        if (i >= allocated) {
            grow((int) Math.min(capacity, 2L * allocated));
        }
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
     * Gives each stored transition a new reward, as from a learned reward model (PEBBLE) or a
     * goal changed in hindsight (HIRO, HER).
     *
     * @param reward the new reward of a transition from its observation, action and next
     *     observation (fresh copies)
     */
    public void relabel(RewardFunction reward) {
        for (int i = 0; i < size; i++) {
            Transition t = transition(i);
            rewards[i] = reward.reward(t.observation(), t.action(), t.nextObservation());
        }
    }

    /**
     * Gives one stored transition a new reward.
     *
     * @param slot the slot, below {@link #size()}
     * @param reward the reward
     */
    public void setReward(int slot, float reward) {
        check(slot);
        rewards[slot] = reward;
    }

    /**
     * Returns a copy of a stored transition.
     *
     * @param slot the slot, below {@link #size()}
     * @return the transition
     */
    public Transition transition(int slot) {
        check(slot);
        return new Transition(
                java.util.Arrays.copyOfRange(observations, slot * observationSize, (slot + 1) * observationSize),
                java.util.Arrays.copyOfRange(actions, slot * actionSize, (slot + 1) * actionSize), rewards[slot],
                java.util.Arrays.copyOfRange(next, slot * observationSize, (slot + 1) * observationSize),
                notDone[slot] == 0, discounts[slot]);
    }

    private void check(int slot) {
        if (slot < 0 || slot >= size) {
            throw new IndexOutOfBoundsException("slot " + slot + " of " + size);
        }
    }

    /** A reward computed from a transition's parts. */
    @FunctionalInterface
    public interface RewardFunction {

        /**
         * @param observation the observation
         * @param action the action
         * @param nextObservation the next observation
         * @return the reward
         */
        float reward(float[] observation, float[] action, float[] nextObservation);
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

    /**
     * Collects the given slots into a batch, for custom sampling schemes.
     *
     * @param idx the slots, each below {@link #size()}
     * @param weights one importance weight per slot
     * @param device where to put the tensors
     * @return the batch
     */
    public Batch gather(int[] idx, float[] weights, Device device) {
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
