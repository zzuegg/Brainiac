package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * FedBN (Li, Jiang, Zhang, Kamp and Dou, ICLR 2021,
 * <a href="https://arxiv.org/abs/2102.07623">arXiv:2102.07623</a>): federated averaging in which
 * the normalisation parameters stay on the clients. Each client keeps its own values of the
 * parameters marked local (batch-norm scales and shifts, or any per-client input normalisation);
 * only the shared parameters are sent, averaged and broadcast. Under feature shift (each client
 * sees differently scaled or offset inputs) the local normalisation absorbs the shift and the
 * shared weights learn the common task.
 */
public final class FedBn {

    private final List<Var> global;
    private final List<FederatedClient> clients;
    private final boolean[] local;
    private final SplittableRandom random;
    private final List<List<Tensor>> localValues = new ArrayList<>();

    private FedBn(List<Var> global, List<FederatedClient> clients, boolean[] local, long seed) {
        this.global = List.copyOf(global);
        this.clients = List.copyOf(clients);
        this.local = local.clone();
        this.random = new SplittableRandom(seed);
        for (FederatedClient c : clients) {
            List<Tensor> v = new ArrayList<>();
            for (int p = 0; p < global.size(); p++) {
                v.add(global.get(p).value());
            }
            localValues.add(v);
            c.load(v);
        }
    }

    /**
     * FedBN.
     *
     * @param global the global parameters
     * @param clients the clients
     * @param local which parameters (in order) stay local
     * @param seed the seed of client selection
     * @return the algorithm
     */
    public static FedBn of(List<Var> global, List<FederatedClient> clients, boolean[] local, long seed) {
        return new FedBn(global, clients, local, seed);
    }

    /**
     * Loads a client's personal model: the global shared parameters with its own local ones.
     *
     * @param client the client index
     */
    public void personalize(int client) {
        List<Tensor> v = new ArrayList<>();
        for (int p = 0; p < global.size(); p++) {
            v.add(local[p] ? localValues.get(client).get(p) : global.get(p).value());
        }
        clients.get(client).load(v);
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
        Tensor[] start = FederatedAlgorithms.values(global);
        Tensor[][] updates = new Tensor[selected.size()][];
        double[] weights = new double[selected.size()];
        double loss = 0;
        for (int s = 0; s < selected.size(); s++) {
            int k = selected.get(s);
            FederatedClient c = clients.get(k);
            personalize(k);
            loss += c.train(localSteps);
            updates[s] = new Tensor[start.length];
            weights[s] = c.samples();
            for (int p = 0; p < start.length; p++) {
                Tensor now = c.parameters().get(p).value();
                if (local[p]) {
                    localValues.get(k).set(p, now);
                    updates[s][p] = Tensor.zeros(now.shape());
                } else {
                    updates[s][p] = now.sub(start[p]);
                }
            }
        }
        Tensor[] delta = FedNova.average(updates, weights);
        for (int p = 0; p < global.size(); p++) {
            if (!local[p]) {
                global.get(p).assign(start[p].add(delta[p]));
            }
        }
        return (float) (loss / selected.size());
    }
}
