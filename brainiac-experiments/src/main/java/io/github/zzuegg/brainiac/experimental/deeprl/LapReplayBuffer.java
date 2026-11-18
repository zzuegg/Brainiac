package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Loss-adjusted prioritised replay (LAP; Fujimoto et al., "An Equivalence between Loss
 * Functions and Non-Uniform Sampling in Experience Replay", NeurIPS 2020,
 * <a href="https://arxiv.org/abs/2007.06049">arXiv:2007.06049</a>), the replay of
 * {@link Td7} and {@link MrQ}. Transitions are drawn with probability proportional to
 * {@code max(|delta|, 1)^alpha}: the clamp at 1 means small errors are sampled uniformly, and
 * paired with a Huber loss of threshold 1 the non-uniform sampling needs no importance
 * weights (the two are equivalent to a reweighted mean-squared error).
 *
 * <p>Transitions are stored in order, so {@link #gather(int[], int, Device)} can also return the
 * step {@code k} after each sampled start with a mask that is 0 once the episode has ended:
 * MR.Q's unrolled model losses and multi-step returns read sequences this way. With
 * {@code prioritised = false} sampling is uniform.
 */
public final class LapReplayBuffer {

    private final int capacity;
    private final int obsSize;
    private final int actSize;
    private final float[] obs;
    private final float[] act;
    private final float[] reward;
    private final float[] next;
    private final float[] notDone;
    private final boolean[] episodeEnd;
    private final double alpha;
    private final boolean prioritised;
    private final double[] tree;
    private final int leaves;
    private double maxPriority = 1;
    private int size;
    private int position;
    private double absRewardSum;

    /**
     * Creates the buffer.
     *
     * @param capacity transitions kept
     * @param observationSize values per observation
     * @param actionSize values per action (one-hot size for discrete actions)
     * @param alpha the priority exponent (0.4 in TD7 and MR.Q)
     * @param prioritised whether to sample by LAP priority (else uniformly)
     */
    public LapReplayBuffer(int capacity, int observationSize, int actionSize, double alpha, boolean prioritised) {
        this.capacity = capacity;
        this.obsSize = observationSize;
        this.actSize = actionSize;
        this.obs = new float[capacity * observationSize];
        this.act = new float[capacity * actionSize];
        this.reward = new float[capacity];
        this.next = new float[capacity * observationSize];
        this.notDone = new float[capacity];
        this.episodeEnd = new boolean[capacity];
        this.alpha = alpha;
        this.prioritised = prioritised;
        int l = 1;
        while (l < capacity) {
            l <<= 1;
        }
        this.leaves = l;
        this.tree = new double[2 * l];
    }

    /**
     * Stores a transition.
     *
     * @param observation the observation
     * @param action the action
     * @param r the reward
     * @param nextObservation the next observation
     * @param terminated whether the next state is terminal
     * @param done whether the episode ended here (terminated or truncated)
     */
    public void add(float[] observation, float[] action, float r, float[] nextObservation, boolean terminated,
            boolean done) {
        int i = position;
        if (size == capacity) {
            absRewardSum -= Math.abs(reward[i]);
        }
        System.arraycopy(observation, 0, obs, i * obsSize, obsSize);
        System.arraycopy(action, 0, act, i * actSize, actSize);
        reward[i] = r;
        absRewardSum += Math.abs(r);
        System.arraycopy(nextObservation, 0, next, i * obsSize, obsSize);
        notDone[i] = terminated ? 0 : 1;
        episodeEnd[i] = done;
        set(i, Math.pow(maxPriority, alpha));
        position = (position + 1) % capacity;
        size = Math.min(size + 1, capacity);
    }

    private void set(int i, double p) {
        int node = i + leaves;
        double delta = p - tree[node];
        while (node >= 1) {
            tree[node] += delta;
            node >>= 1;
        }
    }

    /** @return the transitions stored */
    public int size() {
        return size;
    }

    /** @return the mean absolute reward stored (MR.Q's reward scale) */
    public double meanAbsReward() {
        return size == 0 ? 1 : absRewardSum / size;
    }

    /**
     * Draws start indices whose next {@code horizon - 1} slots are already written.
     *
     * @param batchSize the batch size
     * @param horizon steps that must follow each start (1 for single transitions)
     * @param random the random source
     * @return the slots
     */
    public int[] sample(int batchSize, int horizon, SplittableRandom random) {
        if (size < horizon) {
            throw new IllegalStateException("not enough transitions");
        }
        int[] idx = new int[batchSize];
        for (int b = 0; b < batchSize; b++) {
            int i;
            int tries = 0;
            do {
                i = prioritised && tree[1] > 0 ? find(random.nextDouble() * tree[1]) : random.nextInt(size);
                tries++;
            } while (!validStart(i, horizon) && tries < 100);
            if (!validStart(i, horizon)) {
                i = Math.floorMod(position - horizon, capacity);
            }
            idx[b] = i;
        }
        return idx;
    }

    private boolean validStart(int i, int horizon) {
        if (i >= size) {
            return false;
        }
        // the slots i .. i + horizon - 1 must not cross the write head
        int age = Math.floorMod(position - 1 - i, capacity);
        return age >= horizon - 1 && age < size;
    }

    private int find(double mass) {
        int node = 1;
        while (node < leaves) {
            int left = 2 * node;
            if (mass <= tree[left] || tree[left + 1] == 0) {
                node = left;
            } else {
                mass -= tree[left];
                node = left + 1;
            }
        }
        return Math.min(node - leaves, size - 1);
    }

    /**
     * Sets LAP priorities {@code max(|error|, 1)^alpha} after learning from a batch.
     *
     * @param indices the sampled slots
     * @param errors their absolute TD errors
     */
    public void updatePriorities(int[] indices, float[] errors) {
        for (int i = 0; i < indices.length; i++) {
            double p = Math.max(Math.abs(errors[i]), 1.0);
            maxPriority = Math.max(maxPriority, p);
            set(indices[i], Math.pow(p, alpha));
        }
    }

    /**
     * The transitions {@code k} steps after each start.
     *
     * @param starts sampled slots
     * @param k the offset
     * @param device where to put the tensors
     * @return the batch
     */
    public Batch gather(int[] starts, int k, Device device) {
        int b = starts.length;
        float[] o = new float[b * obsSize];
        float[] a = new float[b * actSize];
        float[] r = new float[b];
        float[] n = new float[b * obsSize];
        float[] nd = new float[b];
        float[] m = new float[b];
        for (int i = 0; i < b; i++) {
            int j = (starts[i] + k) % capacity;
            boolean alive = true;
            for (int s = 0; s < k; s++) {
                if (episodeEnd[(starts[i] + s) % capacity]) {
                    alive = false;
                    break;
                }
            }
            m[i] = alive ? 1 : 0;
            System.arraycopy(obs, j * obsSize, o, i * obsSize, obsSize);
            System.arraycopy(act, j * actSize, a, i * actSize, actSize);
            r[i] = reward[j];
            System.arraycopy(next, j * obsSize, n, i * obsSize, obsSize);
            nd[i] = notDone[j];
        }
        return new Batch(Tensor.of(Shape.of(b, obsSize), o).to(device), Tensor.of(Shape.of(b, actSize), a).to(device),
                Tensor.of(Shape.of(b, 1), r).to(device), Tensor.of(Shape.of(b, obsSize), n).to(device),
                Tensor.of(Shape.of(b, 1), nd).to(device), Tensor.of(Shape.of(b, 1), m).to(device), r, nd, m);
    }

    /**
     * One step of sampled sequences.
     *
     * @param observations {@code [B, obs]}
     * @param actions {@code [B, act]}
     * @param rewards {@code [B, 1]}
     * @param nextObservations {@code [B, obs]}
     * @param notDone {@code [B, 1]}, 0 where the next state is terminal
     * @param mask {@code [B, 1]}, 0 where the episode ended before this step
     * @param rawRewards the rewards as an array
     * @param rawNotDone the not-done flags as an array
     * @param rawMask the mask as an array
     */
    public record Batch(Tensor observations, Tensor actions, Tensor rewards, Tensor nextObservations, Tensor notDone,
            Tensor mask, float[] rawRewards, float[] rawNotDone, float[] rawMask) {
    }
}
