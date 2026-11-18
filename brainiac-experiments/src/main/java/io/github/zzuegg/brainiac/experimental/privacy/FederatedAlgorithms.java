package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.IntUnaryOperator;

/**
 * Federated optimisation that corrects client drift, simulated in-process like
 * {@code FederatedAveraging}: every round a random share of the clients receives the global
 * weights, runs local SGD on its own data, and the server aggregates the updates
 * {@code Δ_i = w_i - w}.
 *
 * <ul>
 *   <li>{@link #fedAvg}: plain local SGD and weighted averaging (McMahan et al. 2017).</li>
 *   <li>{@link #fedProx}: every local gradient gets the proximal term {@code μ (w_i - w)},
 *       the gradient of {@code μ/2 ||w_i - w||²}, which limits how far clients drift towards
 *       their own optimum (Li et al., MLSys 2020, <a href="https://arxiv.org/abs/1812.06127">arXiv:1812.06127</a>).</li>
 *   <li>{@link #scaffold}: server and clients keep control variates {@code c} and {@code c_i},
 *       estimates of the global and the local update direction; every local gradient is
 *       corrected by {@code c - c_i}, so each client follows the global direction and drift
 *       vanishes whatever the heterogeneity (Karimireddy et al., ICML 2020,
 *       <a href="https://arxiv.org/abs/1910.06378">arXiv:1910.06378</a>; option II update
 *       {@code c_i+ = c_i - c + (w - w_i) / (K η)}).</li>
 * </ul>
 *
 * <p>The aggregate goes to a {@link ServerOptimizer} (FedAvg by default; FedAdam, FedYogi,
 * FedAvgM) and may be normalised by step counts with {@link FedNova} when clients do
 * different amounts of local work.
 */
public final class FederatedAlgorithms {

    /** The local update rule. */
    public enum Rule {
        /** plain local SGD */
        SGD,
        /** SGD with a proximal term */
        PROX,
        /** SGD corrected by control variates */
        SCAFFOLD
    }

    /**
     * The server's aggregation step: turns the clients' updates of one round into the single
     * update handed to the {@link ServerOptimizer}. Plain FedAvg ({@link #FED_AVG}) is the
     * sample-weighted mean; other aggregators (ensemble distillation, robust statistics) plug in
     * with {@link FederatedAlgorithms#aggregator}. Without one, the algorithm keeps its built-in
     * FedAvg or FedNova averaging, so existing runs are bit-identical.
     */
    @FunctionalInterface
    public interface Aggregator {

        /** Sample-weighted averaging, {@link FedNova#average}. */
        Aggregator FED_AVG = (start, updates, weights, steps) -> FedNova.average(updates, weights);

        /** FedNova's step-normalised averaging, {@link FedNova#aggregate}. */
        Aggregator FED_NOVA = (start, updates, weights, steps) -> FedNova.aggregate(updates, weights, steps);

        /**
         * Aggregates one round.
         *
         * @param start the global parameter values the clients started from
         * @param updates per selected client, {@code w_i - start} per parameter
         * @param weights per selected client, its number of samples
         * @param steps per selected client, its local steps
         * @return the aggregate update, one tensor per parameter
         */
        Tensor[] aggregate(Tensor[] start, Tensor[][] updates, double[] weights, int[] steps);
    }

    private final List<Var> global;
    private final List<FederatedClient> clients;
    private final Rule rule;
    private final float mu;
    private final SplittableRandom random;
    private ServerOptimizer server = ServerOptimizer.fedAvg();
    private Aggregator aggregator;
    private boolean normalized;
    private Tensor[] serverControl;
    private Tensor[][] clientControl;
    private long transfers;

    private FederatedAlgorithms(List<Var> global, List<FederatedClient> clients, Rule rule, float mu, long seed) {
        this.global = List.copyOf(global);
        this.clients = List.copyOf(clients);
        this.rule = Objects.requireNonNull(rule);
        this.mu = mu;
        this.random = new SplittableRandom(seed);
        if (rule == Rule.SCAFFOLD) {
            serverControl = zeros(this.global);
            clientControl = new Tensor[clients.size()][];
        }
    }

    /**
     * FedAvg.
     *
     * @param global the global parameters
     * @param clients the clients
     * @param seed the seed of client selection
     * @return the algorithm
     */
    public static FederatedAlgorithms fedAvg(List<Var> global, List<FederatedClient> clients, long seed) {
        return new FederatedAlgorithms(global, clients, Rule.SGD, 0, seed);
    }

    /**
     * FedProx.
     *
     * @param global the global parameters
     * @param clients the clients
     * @param mu the proximal strength
     * @param seed the seed of client selection
     * @return the algorithm
     */
    public static FederatedAlgorithms fedProx(List<Var> global, List<FederatedClient> clients, float mu, long seed) {
        return new FederatedAlgorithms(global, clients, Rule.PROX, mu, seed);
    }

    /**
     * SCAFFOLD.
     *
     * @param global the global parameters
     * @param clients the clients
     * @param seed the seed of client selection
     * @return the algorithm
     */
    public static FederatedAlgorithms scaffold(List<Var> global, List<FederatedClient> clients, long seed) {
        return new FederatedAlgorithms(global, clients, Rule.SCAFFOLD, 0, seed);
    }

    /**
     * @param s the server step applied to the aggregate
     * @return this
     */
    public FederatedAlgorithms server(ServerOptimizer s) {
        this.server = Objects.requireNonNull(s);
        return this;
    }

    /**
     * @param a the aggregation step, which replaces the built-in averaging (and
     *     {@link #normalized}); {@code null} restores it
     * @return this
     */
    public FederatedAlgorithms aggregator(Aggregator a) {
        this.aggregator = a;
        return this;
    }

    /**
     * @param on whether to aggregate with {@link FedNova}'s step-count normalisation
     * @return this
     */
    public FederatedAlgorithms normalized(boolean on) {
        this.normalized = on;
        return this;
    }

    /** @return model copies sent or received so far (SCAFFOLD's control variates count double) */
    public long transfers() {
        return transfers;
    }

    /**
     * One round with the same number of local steps for every client.
     *
     * @param localSteps local SGD steps
     * @param participation the share of clients selected
     * @return the mean local loss
     */
    public float round(int localSteps, double participation) {
        return round(c -> localSteps, participation);
    }

    /**
     * One round.
     *
     * @param localSteps the local steps of each client, by client index
     * @param participation the share of clients selected, in {@code (0, 1]}
     * @return the mean local loss of the selected clients
     */
    public float round(IntUnaryOperator localSteps, double participation) {
        List<Integer> selected = select(clients.size(), participation, random);
        Tensor[] start = values(global);
        Tensor[][] updates = new Tensor[selected.size()][];
        Tensor[][] controlDelta = rule == Rule.SCAFFOLD ? new Tensor[selected.size()][] : null;
        double[] weights = new double[selected.size()];
        int[] steps = new int[selected.size()];
        double loss = 0;
        for (int s = 0; s < selected.size(); s++) {
            int k = selected.get(s);
            FederatedClient c = clients.get(k);
            steps[s] = Math.max(1, localSteps.applyAsInt(k));
            weights[s] = c.samples();
            c.load(List.of(start));
            Tensor[] correction = null;
            if (rule == Rule.SCAFFOLD) {
                if (clientControl[k] == null) {
                    clientControl[k] = zeros(global);
                }
                correction = new Tensor[start.length];
                for (int p = 0; p < start.length; p++) {
                    correction[p] = serverControl[p].sub(clientControl[k][p]);
                }
            }
            loss += localTrain(c, start, steps[s], correction);
            transfers += rule == Rule.SCAFFOLD ? 4 : 2;
            updates[s] = new Tensor[start.length];
            for (int p = 0; p < start.length; p++) {
                updates[s][p] = c.parameters().get(p).value().sub(start[p]);
            }
            if (rule == Rule.SCAFFOLD) {
                controlDelta[s] = new Tensor[start.length];
                float inv = 1f / (steps[s] * c.learningRate());
                for (int p = 0; p < start.length; p++) {
                    Tensor next = clientControl[k][p].sub(serverControl[p]).sub(updates[s][p].mul(inv));
                    controlDelta[s][p] = next.sub(clientControl[k][p]);
                    clientControl[k][p] = next;
                }
            }
        }
        Tensor[] delta = aggregator != null ? aggregator.aggregate(start, updates, weights, steps)
                : normalized ? FedNova.aggregate(updates, weights, steps) : FedNova.average(updates, weights);
        server.apply(global, delta);
        if (rule == Rule.SCAFFOLD) {
            float share = (float) selected.size() / clients.size();
            for (int p = 0; p < start.length; p++) {
                Tensor sum = null;
                for (Tensor[] d : controlDelta) {
                    sum = sum == null ? d[p] : sum.add(d[p]);
                }
                serverControl[p] = serverControl[p].add(sum.mul(share / selected.size()));
            }
        }
        return (float) (loss / selected.size());
    }

    private float localTrain(FederatedClient c, Tensor[] start, int steps, Tensor[] correction) {
        float[] l = new float[1];
        double sum = 0;
        float lr = c.learningRate();
        List<Var> params = c.parameters();
        for (int t = 0; t < steps; t++) {
            Tensor[] g = c.gradients(l);
            sum += l[0];
            for (int p = 0; p < g.length; p++) {
                Tensor w = params.get(p).value();
                Tensor step = g[p];
                if (rule == Rule.PROX) {
                    step = step.add(w.sub(start[p]).mul(mu));
                } else if (correction != null) {
                    step = step.add(correction[p]);
                }
                params.get(p).assign(w.sub(step.mul(lr)));
            }
        }
        return (float) (sum / steps);
    }

    /**
     * Draws the clients of a round: each independently with the given probability, at
     * least one.
     *
     * @param n the number of clients
     * @param participation the probability
     * @param random the random source
     * @return the selected client indices
     */
    static List<Integer> select(int n, double participation, SplittableRandom random) {
        List<Integer> selected = new ArrayList<>();
        for (int k = 0; k < n; k++) {
            if (random.nextDouble() < participation) {
                selected.add(k);
            }
        }
        if (selected.isEmpty()) {
            selected.add(random.nextInt(n));
        }
        return selected;
    }

    static Tensor[] values(List<Var> params) {
        Tensor[] out = new Tensor[params.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = params.get(i).value();
        }
        return out;
    }

    static Tensor[] zeros(List<Var> params) {
        Tensor[] out = new Tensor[params.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = Tensor.zeros(params.get(i).shape());
        }
        return out;
    }
}
