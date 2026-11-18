package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * FedDyn (Acar, Zhao, Matas, Mattina, Whatmough and Saligrama, ICLR 2021,
 * <a href="https://arxiv.org/abs/2111.04263">arXiv:2111.04263</a>): every client minimises
 * {@code L_k(θ) - <h_k, θ> + (α/2) ||θ - θ^t-1||^2}, a dynamic regulariser whose linear term
 * {@code h_k} (the client's last gradient correction, updated as
 * {@code h_k ← h_k - α (θ_k - θ^t-1)}) makes the stationary points of the local objectives agree
 * with the global one, so client drift vanishes even with many local steps. The server keeps
 * {@code h ← h - (α/m) Σ_active (θ_k - θ^t-1)} over all {@code m} clients and sets
 * {@code θ^t = mean(θ_k) - h / α}.
 */
public final class FedDyn {

    private final List<Var> global;
    private final List<FederatedClient> clients;
    private final float alpha;
    private final SplittableRandom random;
    private final Tensor[][] clientState;
    private final Tensor[] serverState;

    private FedDyn(List<Var> global, List<FederatedClient> clients, float alpha, long seed) {
        this.global = List.copyOf(global);
        this.clients = List.copyOf(clients);
        this.alpha = alpha;
        this.random = new SplittableRandom(seed);
        this.clientState = new Tensor[clients.size()][];
        this.serverState = FederatedAlgorithms.zeros(this.global);
    }

    /**
     * FedDyn.
     *
     * @param global the global parameters
     * @param clients the clients
     * @param alpha the regulariser strength {@code α}
     * @param seed the seed of client selection
     * @return the algorithm
     */
    public static FedDyn of(List<Var> global, List<FederatedClient> clients, float alpha, long seed) {
        return new FedDyn(global, clients, alpha, seed);
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
        Tensor[] sum = null;
        Tensor[] driftSum = FederatedAlgorithms.zeros(global);
        double loss = 0;
        float[] l = new float[1];
        for (int k : selected) {
            FederatedClient c = clients.get(k);
            if (clientState[k] == null) {
                clientState[k] = FederatedAlgorithms.zeros(global);
            }
            c.load(List.of(start));
            List<Var> params = c.parameters();
            float lr = c.learningRate();
            double local = 0;
            for (int s = 0; s < localSteps; s++) {
                Tensor[] g = c.gradients(l);
                local += l[0];
                for (int p = 0; p < g.length; p++) {
                    Tensor w = params.get(p).value();
                    Tensor step = g[p].sub(clientState[k][p]).add(w.sub(start[p]).mul(alpha));
                    params.get(p).assign(w.sub(step.mul(lr)));
                }
            }
            loss += local / Math.max(1, localSteps);
            Tensor[] theta = FederatedAlgorithms.values(params);
            for (int p = 0; p < theta.length; p++) {
                Tensor drift = theta[p].sub(start[p]);
                clientState[k][p] = clientState[k][p].sub(drift.mul(alpha));
                driftSum[p] = driftSum[p].add(drift);
            }
            if (sum == null) {
                sum = theta.clone();
            } else {
                for (int p = 0; p < theta.length; p++) {
                    sum[p] = sum[p].add(theta[p]);
                }
            }
        }
        float m = clients.size();
        for (int p = 0; p < global.size(); p++) {
            serverState[p] = serverState[p].sub(driftSum[p].mul(alpha / m));
            Tensor mean = sum[p].mul(1f / selected.size());
            global.get(p).assign(mean.sub(serverState[p].mul(1f / alpha)));
        }
        return (float) (loss / selected.size());
    }
}
