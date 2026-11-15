package io.github.zzuegg.brainiac.rl;

import io.github.zzuegg.brainiac.tensor.Device;

import java.util.SplittableRandom;

/**
 * Prioritized experience replay (Schaul et al., 2016): transitions with larger errors are
 * sampled more often, with probability proportional to {@code priority^alpha}, and
 * importance weights {@code (N * P(i))^-beta} (scaled to at most 1) correct the bias.
 * New transitions get the highest priority seen so far, so each is replayed at least once
 * soon.
 */
public final class PrioritizedReplayBuffer extends ReplayBuffer {

    private final double alpha;
    private final double[] tree;
    private final int leaves;
    private double maxPriority = 1;
    private double beta = 0.4;

    /**
     * @param capacity how many transitions it holds
     * @param observationSize values per observation
     * @param actionSize values per action
     * @param alpha how strongly priorities matter: 0 is uniform, 1 fully proportional
     */
    public PrioritizedReplayBuffer(int capacity, int observationSize, int actionSize, double alpha) {
        super(capacity, observationSize, actionSize);
        this.alpha = alpha;
        int n = 1;
        while (n < capacity) {
            n <<= 1;
        }
        leaves = n;
        tree = new double[2 * n];
    }

    /**
     * @param beta how fully importance weights correct the sampling bias, from 0 (not at
     *     all) to 1 (fully); usually raised towards 1 during training
     */
    public void beta(double beta) {
        this.beta = beta;
    }

    private void set(int i, double p) {
        int node = i + leaves;
        double delta = p - tree[node];
        while (node >= 1) {
            tree[node] += delta;
            node >>= 1;
        }
    }

    @Override
    public int add(Transition t) {
        int i = super.add(t);
        set(i, Math.pow(maxPriority, alpha));
        return i;
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
        return Math.min(node - leaves, size() - 1);
    }

    @Override
    public Batch sample(int batchSize, SplittableRandom random, Device device) {
        if (size() == 0) {
            throw new IllegalStateException("the buffer is empty");
        }
        double total = tree[1];
        int[] idx = new int[batchSize];
        float[] w = new float[batchSize];
        double segment = total / batchSize;
        double maxWeight = 0;
        for (int i = 0; i < batchSize; i++) {
            idx[i] = find(segment * (i + random.nextDouble()));
            double p = tree[idx[i] + leaves] / total;
            double weight = Math.pow(size() * Math.max(p, 1e-12), -beta);
            w[i] = (float) weight;
            maxWeight = Math.max(maxWeight, weight);
        }
        for (int i = 0; i < batchSize; i++) {
            w[i] /= (float) maxWeight;
        }
        return gather(idx, w, device);
    }

    /**
     * Sets new priorities after learning from a batch.
     *
     * @param indices the batch's slots
     * @param errors the absolute errors of each transition
     */
    public void updatePriorities(int[] indices, float[] errors) {
        for (int i = 0; i < indices.length; i++) {
            double p = Math.abs(errors[i]) + 1e-6;
            maxPriority = Math.max(maxPriority, p);
            set(indices[i], Math.pow(p, alpha));
        }
    }

    /**
     * @param index a slot
     * @return its priority (before the {@code alpha} power)
     */
    public double priority(int index) {
        return Math.pow(tree[index + leaves], 1 / alpha);
    }
}
