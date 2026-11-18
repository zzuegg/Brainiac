package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The iterative federated clustering algorithm (IFCA; Ghosh, Chung, Yin and Ramchandran,
 * NeurIPS 2020, <a href="https://arxiv.org/abs/2006.04088">arXiv:2006.04088</a>). The server keeps
 * {@code k} global models; every selected client evaluates each on a few of its own minibatches,
 * joins the one with the lowest loss, trains it locally, and the server averages the updates per
 * cluster. Clients whose data follow different concepts (for example permuted labels) end up in
 * different clusters instead of fighting over one model.
 */
public final class ClusteredFederation {

    private final List<List<Var>> models;
    private final List<FederatedClient> clients;
    private final SplittableRandom random;
    private final int[] membership;
    private final int probes;

    private ClusteredFederation(List<List<Var>> models, List<FederatedClient> clients, int probes, long seed) {
        this.models = List.copyOf(models);
        this.clients = List.copyOf(clients);
        this.probes = probes;
        this.random = new SplittableRandom(seed);
        this.membership = new int[clients.size()];
        java.util.Arrays.fill(membership, -1);
    }

    /**
     * IFCA.
     *
     * @param models the {@code k} global models (differently initialised), each in the clients'
     *     parameter order
     * @param clients the clients
     * @param probes minibatches each client uses to pick its cluster
     * @param seed the seed of client selection
     * @return the algorithm
     */
    public static ClusteredFederation of(List<List<Var>> models, List<FederatedClient> clients, int probes,
            long seed) {
        return new ClusteredFederation(models, clients, probes, seed);
    }

    /**
     * One round.
     *
     * @param localSteps local SGD steps
     * @param participation the share of clients selected
     * @return the mean local loss
     */
    public float round(int localSteps, double participation) {
        List<Integer> selected = FederatedAlgorithms.select(clients.size(), participation, random);
        int k = models.size();
        List<List<Tensor[]>> updates = new ArrayList<>();
        List<List<Double>> weights = new ArrayList<>();
        for (int c = 0; c < k; c++) {
            updates.add(new ArrayList<>());
            weights.add(new ArrayList<>());
        }
        double loss = 0;
        for (int idx : selected) {
            FederatedClient client = clients.get(idx);
            int best = assign(idx);
            Tensor[] start = FederatedAlgorithms.values(models.get(best));
            client.load(List.of(start));
            loss += client.train(localSteps);
            Tensor[] u = new Tensor[start.length];
            for (int p = 0; p < start.length; p++) {
                u[p] = client.parameters().get(p).value().sub(start[p]);
            }
            updates.get(best).add(u);
            weights.get(best).add((double) client.samples());
        }
        for (int c = 0; c < k; c++) {
            if (updates.get(c).isEmpty()) {
                continue;
            }
            Tensor[][] u = updates.get(c).toArray(new Tensor[0][]);
            double[] w = weights.get(c).stream().mapToDouble(Double::doubleValue).toArray();
            Tensor[] delta = FedNova.average(u, w);
            List<Var> m = models.get(c);
            for (int p = 0; p < m.size(); p++) {
                m.get(p).assign(m.get(p).value().add(delta[p]));
            }
        }
        return (float) (loss / selected.size());
    }

    /**
     * Picks a client's cluster: the model with the lowest loss on a few of its minibatches.
     *
     * @param client the client index
     * @return the cluster
     */
    public int assign(int client) {
        FederatedClient c = clients.get(client);
        int best = 0;
        double bestLoss = Double.POSITIVE_INFINITY;
        for (int m = 0; m < models.size(); m++) {
            c.load(List.of(FederatedAlgorithms.values(models.get(m))));
            double l = 0;
            for (int p = 0; p < probes; p++) {
                l += Var.noGrad(() -> c.batchLoss().get()).value().item();
            }
            if (l < bestLoss) {
                bestLoss = l;
                best = m;
            }
        }
        membership[client] = best;
        return best;
    }

    /** @return each client's last cluster, {@code -1} if never selected */
    public int[] membership() {
        return membership.clone();
    }
}
