package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Federated Averaging (McMahan et al. 2017, <a href="https://arxiv.org/abs/1602.05629">arXiv:1602.05629</a>),
 * simulated in-process: every round a random share of the clients receives the global
 * weights, trains locally for a few steps on data that never leaves it, and the server
 * replaces the global weights by the clients' average weighted by their data sizes.
 */
public final class FederatedAveraging {

    private final List<Var> global;
    private final List<LocalWorker> clients;
    private final SplittableRandom random;
    private long transfers;

    /**
     * @param global the global model's parameters
     * @param clients the clients
     * @param seed the seed of client selection
     */
    public FederatedAveraging(List<Var> global, List<LocalWorker> clients, long seed) {
        this.global = List.copyOf(global);
        this.clients = List.copyOf(clients);
        this.random = new SplittableRandom(seed);
    }

    /** @return how many model copies were sent or received so far */
    public long transfers() {
        return transfers;
    }

    /**
     * One communication round.
     *
     * @param localSteps steps each selected client trains
     * @param participation the share of clients selected, in {@code (0, 1]}
     * @return the mean local loss of the selected clients
     */
    public float round(int localSteps, double participation) {
        List<LocalWorker> selected = new ArrayList<>();
        for (LocalWorker c : clients) {
            if (random.nextDouble() < participation) {
                selected.add(c);
            }
        }
        if (selected.isEmpty()) {
            selected.add(clients.get(random.nextInt(clients.size())));
        }
        double loss = 0;
        double total = 0;
        Tensor[] sum = new Tensor[global.size()];
        for (LocalWorker c : selected) {
            broadcast(global, c.parameters());
            transfers += 2;
            loss += c.train(localSteps);
            for (int i = 0; i < sum.length; i++) {
                Tensor w = c.parameters().get(i).value().mul(c.samples());
                sum[i] = sum[i] == null ? w : sum[i].add(w);
            }
            total += c.samples();
        }
        for (int i = 0; i < sum.length; i++) {
            global.get(i).assign(sum[i].div((float) total));
        }
        return (float) (loss / selected.size());
    }

    /**
     * Copies parameter values from one model to another.
     *
     * @param from the source parameters
     * @param to the target parameters, same shapes and order
     */
    static void broadcast(List<Var> from, List<Var> to) {
        for (int i = 0; i < from.size(); i++) {
            to.get(i).assign(from.get(i).value());
        }
    }
}
