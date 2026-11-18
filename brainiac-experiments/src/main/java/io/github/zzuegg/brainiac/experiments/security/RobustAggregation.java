package io.github.zzuegg.brainiac.experiments.security;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ClientSplits;
import io.github.zzuegg.brainiac.experimental.data.PoisonedSets;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.privacy.FederatedAlgorithms;
import io.github.zzuegg.brainiac.experimental.privacy.FederatedClient;
import io.github.zzuegg.brainiac.experimental.privacy.ServerOptimizer;
import io.github.zzuegg.brainiac.experimental.security.Aggregator;
import io.github.zzuegg.brainiac.experimental.security.Backdoors;
import io.github.zzuegg.brainiac.experimental.security.ByzantineFederation;
import io.github.zzuegg.brainiac.experimental.security.LearnedAggregationWeights;
import io.github.zzuegg.brainiac.experimental.security.ModelPoisoning;
import io.github.zzuegg.brainiac.experimental.security.PreAggregation;
import io.github.zzuegg.brainiac.experimental.security.RobustAggregators;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 11 items 11.1.1 to 11.1.3: Byzantine-robust federated learning on {@link ToyTasks}
 * digits (8x8, 10 classes) split over 20 clients by {@link PoisonedSets#federation} (IID or
 * Dirichlet label skew), with a tanh MLP 64-32-10, full participation and five local SGD
 * steps per round. A share of the clients is malicious and sends {@link ModelPoisoning}
 * updates; the server aggregates with FedAvg, FedAvgM, the rules of
 * {@link RobustAggregators}, the pre-aggregation of {@link PreAggregation}, or
 * {@link LearnedAggregationWeights}. Model replacement plants the {@link Backdoors} patch
 * trigger from the malicious clients' poisoned data, and its success rate is reported.
 *
 * <p>Baselines without attackers: FedAvg, FedProx and FedNova of {@link FederatedAlgorithms},
 * and the honest-only FedAvg oracle that drops the malicious clients.
 */
public final class RobustAggregation {

    /** The server's defence. */
    public enum Defence {
        /** the mean */
        FEDAVG,
        /** the mean with server momentum */
        FEDAVG_M,
        /** Krum */
        KRUM,
        /** Multi-Krum */
        MULTI_KRUM,
        /** coordinate-wise median */
        MEDIAN,
        /** trimmed mean */
        TRIMMED_MEAN,
        /** geometric median */
        GEOMETRIC_MEDIAN,
        /** bucketing (s = 2) before the median */
        BUCKETING_MEDIAN,
        /** nearest-neighbour mixing before the median */
        NNM_MEDIAN,
        /** nearest-neighbour mixing before Krum */
        NNM_KRUM,
        /** centred clipping with worker momentum */
        CENTERED_CLIP,
        /** learned aggregation weights */
        LEARNED
    }

    /**
     * One cell.
     *
     * @param defence the server rule
     * @param attack the attack
     * @param share the malicious share
     * @param alpha the Dirichlet skew, 0 for IID
     * @param accuracy test accuracy after the last round
     * @param backdoor the trigger's success rate (model replacement), else -1
     */
    public record Cell(String defence, ModelPoisoning.Attack attack, double share, double alpha, double accuracy,
            double backdoor) {
    }

    static final int CLIENTS = 20;
    private static final int HIDDEN = 32;
    private static final int BATCH = 16;
    private static final float LR = 0.5f;
    private static final int LOCAL_STEPS = 5;

    private RobustAggregation() {
    }

    /** The default strength of each attack. */
    static double scale(ModelPoisoning.Attack attack, int malicious) {
        return switch (attack) {
            case SIGN_FLIP -> 4;
            case GAUSSIAN -> 10;
            case IPM -> 0.5;
            case MODEL_REPLACEMENT -> CLIENTS / (double) Math.max(1, malicious);
            default -> 1;
        };
    }

    static ClientSplits.Labelled[] data(long seed) {
        ToyTasks.Data tr = ToyTasks.digits(10, 60, 8, 0.08, seed, seed + 1);
        ToyTasks.Data te = ToyTasks.digits(10, 30, 8, 0.08, seed, seed + 2);
        return new ClientSplits.Labelled[] {new ClientSplits.Labelled(tr.x(), tr.y(), 10),
                new ClientSplits.Labelled(te.x(), te.y(), 10)};
    }

    static Aggregator aggregator(Defence d, int f, int n, long seed) {
        return switch (d) {
            case FEDAVG, FEDAVG_M -> Aggregator.mean();
            case KRUM -> RobustAggregators.of(RobustAggregators.Rule.KRUM, f);
            case MULTI_KRUM -> RobustAggregators.of(RobustAggregators.Rule.MULTI_KRUM, f);
            case MEDIAN -> RobustAggregators.of(RobustAggregators.Rule.MEDIAN, f);
            case TRIMMED_MEAN -> RobustAggregators.of(RobustAggregators.Rule.TRIMMED_MEAN, Math.max(1, f));
            case GEOMETRIC_MEDIAN -> RobustAggregators.of(RobustAggregators.Rule.GEOMETRIC_MEDIAN, f);
            case BUCKETING_MEDIAN -> PreAggregation.bucketing(2, RobustAggregators.of(RobustAggregators.Rule.MEDIAN,
                    f), seed);
            case NNM_MEDIAN -> PreAggregation.nnm(f, RobustAggregators.of(RobustAggregators.Rule.MEDIAN, f));
            case NNM_KRUM -> PreAggregation.nnm(f, RobustAggregators.of(RobustAggregators.Rule.KRUM, f));
            case CENTERED_CLIP -> PreAggregation.centeredClip(0, 1, 0.5);
            case LEARNED -> LearnedAggregationWeights.byzantine(n, f, 0.5);
        };
    }

    /**
     * One cell of the grid.
     *
     * @param defence the server rule
     * @param attack the attack
     * @param share the malicious share
     * @param alpha the Dirichlet skew, {@code <= 0} for IID
     * @param omniscient whether the attacker sees the honest updates
     * @param rounds communication rounds
     * @param seed the seed
     * @return the cell
     */
    public static Cell run(Defence defence, ModelPoisoning.Attack attack, double share, double alpha,
            boolean omniscient, int rounds, long seed) {
        return run(defence, attack, -1, share, alpha, omniscient, rounds, seed);
    }

    /**
     * One cell with an explicit attack strength.
     *
     * @param defence the server rule
     * @param attack the attack
     * @param strength the attack's strength ({@link ModelPoisoning}), or {@code < 0} for the
     *     default: sign flip 4, Gaussian 10, IPM 0.5, ALIE 1, model replacement {@code n/f}
     * @param share the malicious share
     * @param alpha the Dirichlet skew, {@code <= 0} for IID
     * @param omniscient whether the attacker sees the honest updates
     * @param rounds communication rounds
     * @param seed the seed
     * @return the cell
     */
    public static Cell run(Defence defence, ModelPoisoning.Attack attack, double strength, double share,
            double alpha, boolean omniscient, int rounds, long seed) {
        ClientSplits.Labelled[] d = data(seed);
        PoisonedSets.Federation fed = PoisonedSets.federation(d[0].y(), 10, CLIENTS, share, alpha, seed + 3);
        int f = fed.malicious();
        float[] pattern = Backdoors.pattern(64, seed);
        SimpleMlp global = SimpleMlp.tanh(seed, 64, HIDDEN, 10);
        List<FederatedClient> clients = new ArrayList<>();
        for (int k = 0; k < CLIENTS; k++) {
            ClientSplits.Labelled local = d[0].subset(fed.parts()[k]);
            if (fed.byzantine()[k] && attack == ModelPoisoning.Attack.MODEL_REPLACEMENT) {
                local = poisoned(local, seed + k);
            }
            clients.add(client(SimpleMlp.tanh(seed, 64, HIDDEN, 10), local, seed + 100 + k));
        }
        ByzantineFederation loop = ByzantineFederation.of(global.parameters(), clients, fed.byzantine(),
                aggregator(defence, f, CLIENTS, seed), seed + 5).attack(attack, strength < 0 ? scale(attack, f) : strength,
                omniscient);
        if (defence == Defence.FEDAVG_M) {
            loop.server(ServerOptimizer.fedAvgM(1, 0.9f));
        }
        for (int t = 0; t < rounds; t++) {
            loop.round(LOCAL_STEPS);
        }
        double backdoor = attack == ModelPoisoning.Attack.MODEL_REPLACEMENT
                ? Backdoors.attackSuccess(global::forward, d[1].x(), d[1].y(), Backdoors.Kind.PATCH, 8, pattern, 0)
                : -1;
        return new Cell(defence.name(), attack, share, Math.max(0, alpha), accuracy(global, d[1]), backdoor);
    }

    /**
     * The attack-free baselines: FedAvg, FedProx ({@code μ = 0.1}) and FedNova of
     * {@link FederatedAlgorithms}, and the honest-only FedAvg oracle (the malicious share of
     * clients dropped).
     *
     * @param share the share dropped by the oracle
     * @param alpha the Dirichlet skew, {@code <= 0} for IID
     * @param rounds communication rounds
     * @param seed the seed
     * @return cells named FEDAVG, FEDPROX, FEDNOVA and ORACLE
     */
    public static List<Cell> baselines(double share, double alpha, int rounds, long seed) {
        ClientSplits.Labelled[] d = data(seed);
        PoisonedSets.Federation fed = PoisonedSets.federation(d[0].y(), 10, CLIENTS, share, alpha, seed + 3);
        List<Cell> out = new ArrayList<>();
        for (String name : new String[] {"FEDAVG", "FEDPROX", "FEDNOVA", "ORACLE"}) {
            SimpleMlp global = SimpleMlp.tanh(seed, 64, HIDDEN, 10);
            List<FederatedClient> clients = new ArrayList<>();
            for (int k = 0; k < CLIENTS; k++) {
                if (name.equals("ORACLE") && fed.byzantine()[k]) {
                    continue;
                }
                clients.add(client(SimpleMlp.tanh(seed, 64, HIDDEN, 10), d[0].subset(fed.parts()[k]), seed + 100 + k));
            }
            FederatedAlgorithms alg = name.equals("FEDPROX")
                    ? FederatedAlgorithms.fedProx(global.parameters(), clients, 0.1f, seed + 5)
                    : FederatedAlgorithms.fedAvg(global.parameters(), clients, seed + 5);
            alg.normalized(name.equals("FEDNOVA"));
            for (int t = 0; t < rounds; t++) {
                alg.round(LOCAL_STEPS, 1.0);
            }
            out.add(new Cell(name, ModelPoisoning.Attack.NONE, 0, Math.max(0, alpha), accuracy(global, d[1]), -1));
        }
        return out;
    }

    private static ClientSplits.Labelled poisoned(ClientSplits.Labelled local, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[local.size() * 2][];
        int[] y = new int[x.length];
        for (int i = 0; i < local.size(); i++) {
            x[2 * i] = local.x()[i];
            y[2 * i] = local.y()[i];
            x[2 * i + 1] = Backdoors.patch(local.x()[r.nextInt(local.size())], 8);
            y[2 * i + 1] = 0;
        }
        return new ClientSplits.Labelled(x, y, local.classes());
    }

    static FederatedClient client(SimpleMlp net, ClientSplits.Labelled local, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int b = Math.min(BATCH, local.size());
        return new FederatedClient(net.parameters(), local.size(), () -> {
            float[][] x = new float[b][];
            float[] y = new float[b];
            for (int i = 0; i < b; i++) {
                int j = r.nextInt(local.size());
                x[i] = local.x()[j];
                y[i] = local.y()[j];
            }
            return Functional.crossEntropy(net.forward(Var.of(Tensor.of(x))), Tensor.of(Shape.of(b), y));
        }, LR);
    }

    static double accuracy(SimpleMlp net, ClientSplits.Labelled test) {
        return ToyTasks.accuracy(net.predict(test.inputs()), test.y());
    }
}
