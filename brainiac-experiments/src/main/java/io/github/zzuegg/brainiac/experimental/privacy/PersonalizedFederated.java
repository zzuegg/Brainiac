package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.Maml;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Personalised federated learning, for clients whose data really differ, so that one global
 * model is the wrong goal. Every method keeps a global model trained by federated averaging
 * and derives a personal model per client:
 *
 * <ul>
 *   <li>{@link #ditto}: each client also trains a personal model {@code v_i} on
 *       {@code f_i(v) + λ/2 ||v - w||²}, regularised towards the global model (Li, Hu, Beirami
 *       and Smith, ICML 2021, <a href="https://arxiv.org/abs/2012.04221">arXiv:2012.04221</a>).</li>
 *   <li>{@link #fedRep}: the first parameters (the representation) are shared and averaged;
 *       the rest (the head) stay on the client. Locally the head is fitted first with the body
 *       frozen, then the body with the head frozen (Collins et al., ICML 2021,
 *       <a href="https://arxiv.org/abs/2102.07078">arXiv:2102.07078</a>).</li>
 *   <li>{@link #perFedAvg}: the global model is meta-learned so that one local SGD step adapts
 *       it well: every local step is a first-order MAML step run by {@link Maml} (Fallah,
 *       Mokhtari and Ozdaglar, NeurIPS 2020, <a href="https://arxiv.org/abs/2002.07948">arXiv:2002.07948</a>).</li>
 * </ul>
 *
 * <p>{@link #personalize} loads a client's personal model into its parameters for evaluation.
 */
public final class PersonalizedFederated {

    /** The personalisation method. */
    public enum Method {
        /** personal models regularised towards the global one */
        DITTO,
        /** shared body, local heads */
        FED_REP,
        /** meta-learned initialisation */
        PER_FED_AVG
    }

    private final Method method;
    private final List<Var> global;
    private final List<FederatedClient> clients;
    private final SplittableRandom random;
    private final float lambda;
    private final int sharedCount;
    private final int headSteps;
    private final float alpha;
    private final float beta;
    private final Tensor[][] personal;

    private PersonalizedFederated(Method method, List<Var> global, List<FederatedClient> clients, float lambda,
            int sharedCount, int headSteps, float alpha, float beta, long seed) {
        this.method = method;
        this.global = List.copyOf(global);
        this.clients = List.copyOf(clients);
        this.lambda = lambda;
        this.sharedCount = sharedCount;
        this.headSteps = headSteps;
        this.alpha = alpha;
        this.beta = beta;
        this.random = new SplittableRandom(seed);
        this.personal = new Tensor[clients.size()][];
    }

    /**
     * Ditto.
     *
     * @param global the global parameters
     * @param clients the clients
     * @param lambda the pull of the personal model towards the global one
     * @param seed the seed of client selection
     * @return the algorithm
     */
    public static PersonalizedFederated ditto(List<Var> global, List<FederatedClient> clients, float lambda,
            long seed) {
        return new PersonalizedFederated(Method.DITTO, global, clients, lambda, global.size(), 0, 0, 0, seed);
    }

    /**
     * FedRep.
     *
     * @param global the global parameters
     * @param clients the clients
     * @param sharedCount how many leading parameters form the shared representation
     * @param headSteps local head steps before the body steps
     * @param seed the seed of client selection
     * @return the algorithm
     */
    public static PersonalizedFederated fedRep(List<Var> global, List<FederatedClient> clients, int sharedCount,
            int headSteps, long seed) {
        return new PersonalizedFederated(Method.FED_REP, global, clients, 0, sharedCount, headSteps, 0, 0, seed);
    }

    /**
     * Per-FedAvg (first order).
     *
     * @param global the global parameters
     * @param clients the clients
     * @param alpha the inner (personalisation) step size
     * @param beta the outer step size
     * @param seed the seed of client selection
     * @return the algorithm
     */
    public static PersonalizedFederated perFedAvg(List<Var> global, List<FederatedClient> clients, float alpha,
            float beta, long seed) {
        return new PersonalizedFederated(Method.PER_FED_AVG, global, clients, 0, global.size(), 0, alpha, beta, seed);
    }

    /** @return the method */
    public Method method() {
        return method;
    }

    /**
     * One communication round.
     *
     * @param localSteps local steps per selected client
     * @param participation the share of clients selected
     * @return the number of clients selected
     */
    public int round(int localSteps, double participation) {
        List<Integer> selected = FederatedAlgorithms.select(clients.size(), participation, random);
        Tensor[] start = FederatedAlgorithms.values(global);
        Tensor[][] updates = new Tensor[selected.size()][];
        double[] weights = new double[selected.size()];
        for (int s = 0; s < selected.size(); s++) {
            int k = selected.get(s);
            FederatedClient c = clients.get(k);
            weights[s] = c.samples();
            switch (method) {
                case DITTO -> {
                    c.load(List.of(start));
                    c.train(localSteps);
                    updates[s] = delta(c, start);
                    if (personal[k] == null) {
                        personal[k] = start.clone();
                    }
                    c.load(List.of(personal[k]));
                    proxTrain(c, start, localSteps);
                    personal[k] = FederatedAlgorithms.values(c.parameters());
                }
                case FED_REP -> {
                    c.load(List.of(withHead(start, k)));
                    trainRange(c, sharedCount, start.length, headSteps);
                    trainRange(c, 0, sharedCount, localSteps);
                    personal[k] = FederatedAlgorithms.values(c.parameters());
                    updates[s] = delta(c, start);
                    for (int p = sharedCount; p < start.length; p++) {
                        updates[s][p] = Tensor.zeros(start[p].shape());
                    }
                }
                case PER_FED_AVG -> {
                    c.load(List.of(start));
                    Maml maml = new Maml(c.parameters(), Optimizer.sgd(beta), alpha, 1, false);
                    Maml.Task task = task(c);
                    for (int t = 0; t < localSteps; t++) {
                        maml.metaStep(List.of(task));
                    }
                    updates[s] = delta(c, start);
                }
            }
        }
        Tensor[] avg = FedNova.average(updates, weights);
        for (int p = 0; p < start.length; p++) {
            global.get(p).assign(start[p].add(avg[p]));
        }
        return selected.size();
    }

    /**
     * Loads a client's personal model into its parameters. Clients that never took part
     * personalise from the current global model first (Ditto and FedRep with the given number
     * of local steps; Per-FedAvg always adapts the current global model by one step).
     *
     * @param k the client
     * @param steps local steps for clients without personal state
     */
    public void personalize(int k, int steps) {
        FederatedClient c = clients.get(k);
        Tensor[] start = FederatedAlgorithms.values(global);
        switch (method) {
            case DITTO -> {
                if (personal[k] == null) {
                    c.load(List.of(start));
                    proxTrain(c, start, steps);
                    personal[k] = FederatedAlgorithms.values(c.parameters());
                }
                c.load(List.of(personal[k]));
            }
            case FED_REP -> {
                c.load(List.of(withHead(start, k)));
                if (personal[k] == null) {
                    trainRange(c, sharedCount, start.length, Math.max(steps, headSteps));
                    personal[k] = FederatedAlgorithms.values(c.parameters());
                }
            }
            case PER_FED_AVG -> {
                c.load(List.of(start));
                new Maml(c.parameters(), Optimizer.sgd(beta), alpha, 1, false).adapt(task(c), 1);
            }
        }
    }

    private Tensor[] withHead(Tensor[] start, int k) {
        Tensor[] w = start.clone();
        if (personal[k] != null) {
            for (int p = sharedCount; p < w.length; p++) {
                w[p] = personal[k][p];
            }
        }
        return w;
    }

    private static Maml.Task task(FederatedClient c) {
        return new Maml.Task() {
            @Override
            public Var supportLoss() {
                return c.batchLoss().get();
            }

            @Override
            public Var queryLoss() {
                return c.batchLoss().get();
            }
        };
    }

    private void proxTrain(FederatedClient c, Tensor[] anchor, int steps) {
        List<Var> params = c.parameters();
        for (int t = 0; t < steps; t++) {
            Tensor[] g = c.gradients(null);
            for (int p = 0; p < g.length; p++) {
                Tensor w = params.get(p).value();
                params.get(p).assign(w.sub(g[p].add(w.sub(anchor[p]).mul(lambda)).mul(c.learningRate())));
            }
        }
    }

    private static void trainRange(FederatedClient c, int from, int to, int steps) {
        List<Var> params = c.parameters();
        for (int t = 0; t < steps; t++) {
            Tensor[] g = c.gradients(null);
            for (int p = from; p < to; p++) {
                Var v = params.get(p);
                v.assign(v.value().sub(g[p].mul(c.learningRate())));
            }
        }
    }

    private static Tensor[] delta(FederatedClient c, Tensor[] start) {
        List<Tensor> now = new ArrayList<>();
        for (Var p : c.parameters()) {
            now.add(p.value());
        }
        Tensor[] d = new Tensor[start.length];
        for (int p = 0; p < d.length; p++) {
            d[p] = now.get(p).sub(start[p]);
        }
        return d;
    }
}
