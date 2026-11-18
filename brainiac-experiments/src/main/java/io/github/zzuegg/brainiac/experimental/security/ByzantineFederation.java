package io.github.zzuegg.brainiac.experimental.security;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.privacy.FederatedClient;
import io.github.zzuegg.brainiac.experimental.privacy.ServerOptimizer;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A federated round loop with malicious clients and a pluggable server {@link Aggregator}.
 * Every round all clients receive the global weights and run the unchanged local step of
 * {@link FederatedClient#train(int)}; the malicious clients' updates are then replaced by a
 * {@link ModelPoisoning} attack, the updates go through the aggregator instead of
 * {@code FederatedAlgorithms}' fixed weighted mean, and the aggregate reaches the
 * {@link ServerOptimizer} (FedAvg, or FedAvgM for server momentum).
 *
 * <p>Clients report their mean local loss with their update (for
 * {@link LearnedAggregationWeights}); malicious clients report the smallest honest loss, the
 * most trusted value they can claim.
 */
public final class ByzantineFederation {

    private final List<Var> global;
    private final List<FederatedClient> clients;
    private final boolean[] byzantine;
    private final Aggregator aggregator;
    private ModelPoisoning.Attack attack = ModelPoisoning.Attack.NONE;
    private double scale = 1;
    private boolean omniscient = true;
    private ServerOptimizer server = ServerOptimizer.fedAvg();
    private final SplittableRandom random;

    private ByzantineFederation(List<Var> global, List<FederatedClient> clients, boolean[] byzantine,
            Aggregator aggregator, long seed) {
        this.global = List.copyOf(global);
        this.clients = List.copyOf(clients);
        this.byzantine = byzantine.clone();
        this.aggregator = Objects.requireNonNull(aggregator);
        this.random = new SplittableRandom(seed);
        if (byzantine.length != clients.size()) {
            throw new IllegalArgumentException("one flag per client");
        }
    }

    /**
     * Creates the loop.
     *
     * @param global the global parameters
     * @param clients all clients, honest and malicious
     * @param byzantine which clients are malicious
     * @param aggregator the server's rule
     * @param seed the seed of the attacks' noise
     * @return the loop
     */
    public static ByzantineFederation of(List<Var> global, List<FederatedClient> clients, boolean[] byzantine,
            Aggregator aggregator, long seed) {
        return new ByzantineFederation(global, clients, byzantine, aggregator, seed);
    }

    /**
     * @param a the malicious clients' attack
     * @param s its strength
     * @param knowsHonest whether the attacker sees the honest updates (omniscient)
     * @return this
     */
    public ByzantineFederation attack(ModelPoisoning.Attack a, double s, boolean knowsHonest) {
        this.attack = Objects.requireNonNull(a);
        this.scale = s;
        this.omniscient = knowsHonest;
        return this;
    }

    /**
     * @param s the server step applied to the aggregate
     * @return this
     */
    public ByzantineFederation server(ServerOptimizer s) {
        this.server = Objects.requireNonNull(s);
        return this;
    }

    /**
     * One round with full participation.
     *
     * @param localSteps local SGD steps per client
     * @return the mean honest local loss
     */
    public float round(int localSteps) {
        Tensor[] start = new Tensor[global.size()];
        for (int p = 0; p < start.length; p++) {
            start[p] = global.get(p).value();
        }
        int n = clients.size();
        float[][] updates = new float[n][];
        double[] losses = new double[n];
        List<float[]> honest = new ArrayList<>();
        List<float[]> own = new ArrayList<>();
        double minHonest = Double.POSITIVE_INFINITY;
        double sum = 0;
        for (int k = 0; k < n; k++) {
            FederatedClient c = clients.get(k);
            c.load(List.of(start));
            losses[k] = c.train(localSteps);
            Tensor[] delta = new Tensor[start.length];
            for (int p = 0; p < start.length; p++) {
                delta[p] = c.parameters().get(p).value().sub(start[p]);
            }
            updates[k] = flatten(delta);
            if (byzantine[k]) {
                own.add(updates[k]);
            } else {
                honest.add(updates[k]);
                minHonest = Math.min(minHonest, losses[k]);
                sum += losses[k];
            }
        }
        float[][] crafted = ModelPoisoning.craft(attack, honest.toArray(new float[0][]), own.toArray(new float[0][]),
                n, omniscient, scale, random);
        int b = 0;
        for (int k = 0; k < n; k++) {
            if (byzantine[k]) {
                updates[k] = crafted[b++];
                losses[k] = minHonest;
            }
        }
        float[] agg = aggregator.aggregate(updates, losses);
        server.apply(global, unflatten(agg, global));
        return (float) (sum / Math.max(1, honest.size()));
    }

    /**
     * Flattens per-parameter tensors into one vector.
     *
     * @param parts the tensors
     * @return their concatenated values
     */
    public static float[] flatten(Tensor[] parts) {
        int total = 0;
        float[][] arrays = new float[parts.length][];
        for (int p = 0; p < parts.length; p++) {
            arrays[p] = parts[p].toArray();
            total += arrays[p].length;
        }
        float[] out = new float[total];
        int o = 0;
        for (float[] a : arrays) {
            System.arraycopy(a, 0, out, o, a.length);
            o += a.length;
        }
        return out;
    }

    /**
     * Splits a flat vector into tensors shaped like the parameters.
     *
     * @param flat the vector
     * @param like the parameters giving the shapes
     * @return one tensor per parameter
     */
    public static Tensor[] unflatten(float[] flat, List<Var> like) {
        Tensor[] out = new Tensor[like.size()];
        int o = 0;
        for (int p = 0; p < out.length; p++) {
            int size = Math.toIntExact(like.get(p).value().size());
            float[] v = new float[size];
            System.arraycopy(flat, o, v, 0, size);
            o += size;
            out[p] = Tensor.of(like.get(p).shape(), v);
        }
        return out;
    }
}
