package io.github.zzuegg.brainiac.experiments.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ClientSplits;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.privacy.FederatedAlgorithms;
import io.github.zzuegg.brainiac.experimental.privacy.FederatedClient;
import io.github.zzuegg.brainiac.experimental.privacy.PersonalizedFederated;
import io.github.zzuegg.brainiac.experimental.privacy.SecureAggregation;
import io.github.zzuegg.brainiac.experimental.privacy.ServerOptimizer;
import io.github.zzuegg.brainiac.experimental.privacy.UserLevelDpFedAvg;
import io.github.zzuegg.brainiac.experimental.train.DpSgd;
import io.github.zzuegg.brainiac.experimental.train.MachineUnlearning;
import io.github.zzuegg.brainiac.experimental.train.OneRunAudit;
import io.github.zzuegg.brainiac.experimental.train.RdpAccountant;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 8 items 8.6.1 to 8.6.4: federated learning on non-iid clients made by
 * {@link ClientSplits} from {@link ToyTasks#digits} (10 classes) and {@link ShapeImages} (6
 * classes), with a small tanh MLP per client.
 *
 * <ul>
 *   <li>{@link #drift}: FedAvg against FedProx, SCAFFOLD, the FedOpt server optimisers (FedAdam,
 *       FedYogi, FedAvgM) and FedNova, at a Dirichlet label skew {@code α}, with centralised
 *       training at the same number of gradient steps as the upper bound. With
 *       {@code variableSteps} every client draws its local steps from 1 to {@code 2K}, the
 *       setting FedNova is built for.</li>
 *   <li>{@link #serverRates}: final accuracy over a grid of server learning rates, to measure
 *       how sensitive each server optimiser is to its rate.</li>
 *   <li>{@link #personalization}: Ditto, FedRep and Per-FedAvg against FedAvg, FedAvg with
 *       local fine-tuning, and local training alone, measured on every client's own test
 *       distribution (mean and worst-decile accuracy), under label skew or concept shift
 *       (client groups with permuted labels).</li>
 *   <li>{@link #userLevelDp}: DP-FedAvg with clients clipped as a whole at several user-level
 *       budgets, against non-private FedAvg and centralised example-level DP-SGD at the same
 *       {@code ε}, plus a one-run audit with canary users.</li>
 *   <li>{@link #secureAggregation}: exactness and cost of the masked sum against dropout.</li>
 * </ul>
 */
public final class FederatedShootout {

    /** The data set. */
    public enum Task {
        /** {@link ToyTasks#digits}, 10 classes */
        DIGITS,
        /** {@link ShapeImages}, 6 classes */
        SHAPES
    }

    /**
     * One method's outcome.
     *
     * @param method the method
     * @param accuracy the global model's test accuracy after the last round
     * @param roundsToTarget the first round reaching the target accuracy, or -1
     */
    public record Row(String method, double accuracy, int roundsToTarget) {
    }

    /**
     * One personalisation method's outcome.
     *
     * @param method the method
     * @param mean mean accuracy over clients on their own test data
     * @param worstDecile mean accuracy of the worst 10% of clients
     */
    public record Personal(String method, double mean, double worstDecile) {
    }

    /**
     * One privacy row.
     *
     * @param method the method
     * @param epsilon the spent {@code ε} (user-level for DP-FedAvg, example-level for DP-SGD)
     * @param accuracy the test accuracy
     * @param auditEpsilon the one-run audit's empirical lower bound, NaN when not audited
     */
    public record Private(String method, double epsilon, double accuracy, double auditEpsilon) {
    }

    /**
     * One secure aggregation run.
     *
     * @param dropout the dropout rate
     * @param survivors clients that sent their masked input
     * @param maxError the largest absolute error of the decoded sum against the plain sum
     * @param recovered secrets reconstructed by the server
     * @param millis the protocol's wall time
     */
    public record SecureRow(double dropout, int survivors, double maxError, int recovered, double millis) {
    }

    static final double DELTA = 1e-5;
    private static final int HIDDEN = 32;
    private static final int BATCH = 16;
    private static final float LR = 0.3f;

    private FederatedShootout() {
    }

    /**
     * Train and test data of a task.
     *
     * @param task the task
     * @param train training examples
     * @param seed the seed
     * @return {@code {train, test}}
     */
    static ClientSplits.Labelled[] data(Task task, int train, long seed) {
        if (task == Task.DIGITS) {
            ToyTasks.Data tr = ToyTasks.digits(10, train / 10, 8, 0.08, seed, seed + 1);
            ToyTasks.Data te = ToyTasks.digits(10, 40, 8, 0.08, seed, seed + 2);
            return new ClientSplits.Labelled[] {new ClientSplits.Labelled(tr.x(), tr.y(), 10),
                    new ClientSplits.Labelled(te.x(), te.y(), 10)};
        }
        ShapeImages images = new ShapeImages(8, 0.1f);
        return new ClientSplits.Labelled[] {ClientSplits.fromImages(images.generate(train, seed + 1), 6),
                ClientSplits.fromImages(images.generate(400, seed + 2), 6)};
    }

    static SimpleMlp net(int classes, long seed) {
        return SimpleMlp.tanh(seed, 64, HIDDEN, classes);
    }

    /** A client: its own replica of the network, and minibatches of its own data. */
    static FederatedClient client(SimpleMlp net, ClientSplits.Labelled local, float lr, long seed) {
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
        }, lr);
    }

    static double accuracy(SimpleMlp net, ClientSplits.Labelled test) {
        return ToyTasks.accuracy(net.predict(test.inputs()), test.y());
    }

    /**
     * Federated optimisers against client drift.
     *
     * @param task the data set
     * @param alpha the Dirichlet label skew
     * @param clients the number of clients
     * @param participation the share of clients per round
     * @param rounds communication rounds
     * @param localSteps local steps {@code K}
     * @param variableSteps whether clients draw their steps from 1 to {@code 2K}
     * @param target the accuracy whose first round is reported
     * @param seed the seed
     * @return centralised, FedAvg, FedProx, SCAFFOLD, FedAdam, FedYogi, FedAvgM, FedNova
     */
    public static List<Row> drift(Task task, double alpha, int clients, double participation, int rounds,
            int localSteps, boolean variableSteps, double target, long seed) {
        ClientSplits.Labelled[] d = data(task, 800, seed);
        int[][] parts = ClientSplits.dirichlet(d[0].y(), d[0].classes(), clients, alpha, 4, seed + 3);
        int[][] steps = new int[rounds][clients];
        SplittableRandom sr = new SplittableRandom(seed + 9);
        for (int[] row : steps) {
            for (int k = 0; k < clients; k++) {
                row[k] = variableSteps ? 1 + sr.nextInt(2 * localSteps) : localSteps;
            }
        }
        List<Row> out = new ArrayList<>();
        out.add(centralised(d, (int) Math.round(rounds * localSteps * clients * participation), target, seed));
        out.add(federated("FedAvg", d, parts, rounds, steps, participation, target, seed,
                (g, c) -> FederatedAlgorithms.fedAvg(g, c, seed)));
        out.add(federated("FedProx", d, parts, rounds, steps, participation, target, seed,
                (g, c) -> FederatedAlgorithms.fedProx(g, c, 0.1f, seed)));
        out.add(federated("SCAFFOLD", d, parts, rounds, steps, participation, target, seed,
                (g, c) -> FederatedAlgorithms.scaffold(g, c, seed)));
        out.add(federated("FedAdam", d, parts, rounds, steps, participation, target, seed,
                (g, c) -> FederatedAlgorithms.fedAvg(g, c, seed).server(ServerOptimizer.fedAdam(0.01f))));
        out.add(federated("FedYogi", d, parts, rounds, steps, participation, target, seed,
                (g, c) -> FederatedAlgorithms.fedAvg(g, c, seed).server(ServerOptimizer.fedYogi(0.01f))));
        out.add(federated("FedAvgM", d, parts, rounds, steps, participation, target, seed,
                (g, c) -> FederatedAlgorithms.fedAvg(g, c, seed).server(ServerOptimizer.fedAvgM(1f, 0.5f))));
        out.add(federated("FedNova", d, parts, rounds, steps, participation, target, seed,
                (g, c) -> FederatedAlgorithms.fedAvg(g, c, seed).normalized(true)));
        return out;
    }

    /**
     * Final accuracy of several server optimisers over a grid of server learning rates.
     *
     * @param alpha the label skew
     * @param multipliers factors applied to each method's default server rate (1 for SGD, 0.5 for
     *     FedAvgM with momentum 0.9, 0.01 for FedAdam)
     * @param rounds rounds
     * @param seed the seed
     * @return {@code [method][rate]} accuracies, methods SGD server (FedAvg at rate 1), FedAvgM, FedAdam
     */
    public static double[][] serverRates(double alpha, float[] multipliers, int rounds, long seed) {
        float[] rates = multipliers;
        ClientSplits.Labelled[] d = data(Task.DIGITS, 800, seed);
        int clients = 20;
        int[][] parts = ClientSplits.dirichlet(d[0].y(), d[0].classes(), clients, alpha, 4, seed + 3);
        int[][] steps = new int[rounds][clients];
        for (int[] row : steps) {
            Arrays.fill(row, 10);
        }
        double[][] out = new double[3][rates.length];
        for (int i = 0; i < rates.length; i++) {
            float lr = rates[i];
            out[0][i] = federated("SGD", d, parts, rounds, steps, 0.2, 2, seed, (g, c) -> FederatedAlgorithms
                    .fedAvg(g, c, seed).server(ServerOptimizer.of(io.github.zzuegg.brainiac.train.Optimizer
                            .sgd(lr)))).accuracy();
            out[1][i] = federated("FedAvgM", d, parts, rounds, steps, 0.2, 2, seed,
                    (g, c) -> FederatedAlgorithms.fedAvg(g, c, seed).server(ServerOptimizer.fedAvgM(lr * 0.5f, 0.9f)))
                    .accuracy();
            out[2][i] = federated("FedAdam", d, parts, rounds, steps, 0.2, 2, seed,
                    (g, c) -> FederatedAlgorithms.fedAvg(g, c, seed).server(ServerOptimizer.fedAdam(lr * 0.01f))).accuracy();
        }
        return out;
    }

    private interface Factory {
        FederatedAlgorithms create(List<Var> global, List<FederatedClient> clients);
    }

    private static Row centralised(ClientSplits.Labelled[] d, int steps, double target, long seed) {
        SimpleMlp net = net(d[0].classes(), seed);
        FederatedClient all = client(net, d[0], LR, seed);
        all.train(steps);
        return new Row("centralised", accuracy(net, d[1]), -1);
    }

    private static Row federated(String name, ClientSplits.Labelled[] d, int[][] parts, int rounds, int[][] steps,
            double participation, double target, long seed, Factory factory) {
        SimpleMlp global = net(d[0].classes(), seed);
        List<FederatedClient> clients = new ArrayList<>();
        for (int k = 0; k < parts.length; k++) {
            clients.add(client(net(d[0].classes(), seed), d[0].subset(parts[k]), LR, seed * 1000 + k));
        }
        FederatedAlgorithms alg = factory.create(global.parameters(), clients);
        int reached = -1;
        for (int r = 0; r < rounds; r++) {
            int[] row = steps[r];
            alg.round(k -> row[k], participation);
            if (reached < 0 && (r % 5 == 4 || r == rounds - 1) && accuracy(global, d[1]) >= target) {
                reached = r + 1;
            }
        }
        return new Row(name, accuracy(global, d[1]), reached);
    }

    /**
     * Personalised federated learning.
     *
     * @param conceptShift client groups with permuted labels (iid inputs) if true, else label
     *     skew at {@code α = 0.1}
     * @param clients the number of clients
     * @param rounds rounds
     * @param seed the seed
     * @return FedAvg, FedAvg + fine-tuning, local only, Ditto, FedRep, Per-FedAvg
     */
    public static List<Personal> personalization(boolean conceptShift, int clients, int rounds, long seed) {
        ClientSplits.Labelled[] d = data(Task.DIGITS, 1000, seed);
        int classes = d[0].classes();
        int[][] parts = conceptShift ? ClientSplits.iid(d[0].size(), clients, seed + 3)
                : ClientSplits.dirichlet(d[0].y(), classes, clients, 0.1, 6, seed + 3);
        int[][] perms = ClientSplits.labelPermutations(clients, classes, conceptShift ? 4 : 1, seed + 4);
        ClientSplits.Labelled[] train = new ClientSplits.Labelled[clients];
        ClientSplits.Labelled[] test = new ClientSplits.Labelled[clients];
        for (int k = 0; k < clients; k++) {
            train[k] = d[0].subset(parts[k]).relabel(perms[k]);
            test[k] = matchingTest(d[1], d[0].subset(parts[k]), 40, seed + k).relabel(perms[k]);
        }
        int localSteps = 10;
        double participation = 0.3;
        List<Personal> out = new ArrayList<>();
        // FedAvg and FedAvg + local fine-tuning
        SimpleMlp global = net(classes, seed);
        List<SimpleMlp> nets = new ArrayList<>();
        List<FederatedClient> fc = clients(train, classes, nets, seed);
        FederatedAlgorithms avg = FederatedAlgorithms.fedAvg(global.parameters(), fc, seed);
        for (int r = 0; r < rounds; r++) {
            avg.round(localSteps, participation);
        }
        double[] acc = new double[clients];
        for (int k = 0; k < clients; k++) {
            acc[k] = accuracy(global, test[k]);
        }
        out.add(summary("FedAvg", acc));
        for (int k = 0; k < clients; k++) {
            fc.get(k).load(snapshot(global.parameters()));
            fc.get(k).train(2 * localSteps);
            acc[k] = accuracy(nets.get(k), test[k]);
        }
        out.add(summary("FedAvg + fine-tune", acc));
        // local training alone, with as many steps as the federated clients take in total
        int localTotal = (int) Math.round(rounds * participation * localSteps) + 2 * localSteps;
        for (int k = 0; k < clients; k++) {
            SimpleMlp n = net(classes, seed);
            client(n, train[k], LR, seed + 77 * k).train(localTotal);
            acc[k] = accuracy(n, test[k]);
        }
        out.add(summary("local only", acc));
        // the personalised methods
        out.add(personal("Ditto", train, test, classes, rounds, localSteps, participation, seed,
                (g, c) -> PersonalizedFederated.ditto(g, c, 0.1f, seed)));
        out.add(personal("FedRep", train, test, classes, rounds, localSteps, participation, seed,
                (g, c) -> PersonalizedFederated.fedRep(g, c, 2, 5, seed)));
        out.add(personal("Per-FedAvg", train, test, classes, rounds, localSteps, participation, seed,
                (g, c) -> PersonalizedFederated.perFedAvg(g, c, LR, LR, seed)));
        return out;
    }

    /** Test examples drawn with the same class histogram as a client's training data. */
    private static ClientSplits.Labelled matchingTest(ClientSplits.Labelled pool, ClientSplits.Labelled local, int n,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<List<Integer>> byClass = new ArrayList<>();
        for (int c = 0; c < pool.classes(); c++) {
            byClass.add(new ArrayList<>());
        }
        for (int i = 0; i < pool.size(); i++) {
            byClass.get(pool.y()[i]).add(i);
        }
        int[] rows = new int[n];
        for (int i = 0; i < n; i++) {
            int c = local.y()[r.nextInt(local.size())];
            List<Integer> idx = byClass.get(c);
            rows[i] = idx.get(r.nextInt(idx.size()));
        }
        return pool.subset(rows);
    }

    private static List<FederatedClient> clients(ClientSplits.Labelled[] train, int classes, List<SimpleMlp> nets,
            long seed) {
        List<FederatedClient> fc = new ArrayList<>();
        for (int k = 0; k < train.length; k++) {
            SimpleMlp n = net(classes, seed);
            nets.add(n);
            fc.add(client(n, train[k], LR, seed * 1000 + k));
        }
        return fc;
    }

    private interface PersonalFactory {
        PersonalizedFederated create(List<Var> global, List<FederatedClient> clients);
    }

    private static Personal personal(String name, ClientSplits.Labelled[] train, ClientSplits.Labelled[] test,
            int classes, int rounds, int localSteps, double participation, long seed, PersonalFactory factory) {
        SimpleMlp global = net(classes, seed);
        List<SimpleMlp> nets = new ArrayList<>();
        List<FederatedClient> fc = clients(train, classes, nets, seed);
        PersonalizedFederated alg = factory.create(global.parameters(), fc);
        for (int r = 0; r < rounds; r++) {
            alg.round(localSteps, participation);
        }
        double[] acc = new double[train.length];
        for (int k = 0; k < train.length; k++) {
            alg.personalize(k, 2 * localSteps);
            acc[k] = accuracy(nets.get(k), test[k]);
        }
        return summary(name, acc);
    }

    private static Personal summary(String name, double[] acc) {
        double[] s = acc.clone();
        Arrays.sort(s);
        int tail = Math.max(1, s.length / 10);
        double worst = 0;
        for (int i = 0; i < tail; i++) {
            worst += s[i] / tail;
        }
        return new Personal(name, Arrays.stream(acc).average().orElse(0), worst);
    }

    /**
     * User-level DP-FedAvg against non-private FedAvg and example-level DP-SGD.
     *
     * @param users the number of users (each holding a few digits)
     * @param budgets the user-level budgets
     * @param rounds rounds
     * @param samplingRate the per-user sampling rate
     * @param canaries canary users, each included with probability one half, for the audit
     * @param seed the seed
     * @return non-private FedAvg, DP-FedAvg per budget (audited), DP-SGD at the first budget
     */
    public static List<Private> userLevelDp(int users, double[] budgets, int rounds, double samplingRate,
            int canaries, long seed) {
        ClientSplits.Labelled[] d = data(Task.DIGITS, 2000, seed);
        int[][] parts = ClientSplits.iid(d[0].size(), users, seed + 3);
        SplittableRandom r = new SplittableRandom(seed + 5);
        float[][] cx = new float[canaries][64];
        int[] cy = new int[canaries];
        boolean[] included = new boolean[canaries];
        for (int c = 0; c < canaries; c++) {
            for (int p = 0; p < 64; p++) {
                cx[c][p] = r.nextDouble() < 0.35 ? 1 : 0;
            }
            cy[c] = r.nextInt(10);
            included[c] = r.nextBoolean();
        }
        List<Private> out = new ArrayList<>();
        out.add(dpFedAvg("FedAvg (non-private)", d, parts, cx, cy, included, 0, rounds, samplingRate, seed));
        for (double eps : budgets) {
            double sigma = RdpAccountant.noiseMultiplierFor(eps, samplingRate, rounds, DELTA);
            out.add(dpFedAvg("DP-FedAvg eps " + eps, d, parts, cx, cy, included, sigma, rounds, samplingRate, seed));
        }
        // centralised example-level DP-SGD at the first budget
        float[][] y = new float[d[0].size()][10];
        for (int i = 0; i < y.length; i++) {
            y[i][d[0].y()[i]] = 1;
        }
        Dataset train = Dataset.of(d[0].x(), y);
        int steps = rounds * 2;
        double q = 0.05;
        double sigma = RdpAccountant.noiseMultiplierFor(budgets[0], q, steps, DELTA);
        Model m = Model.sequential().input(64).dense(HIDDEN, Activation.TANH).dense(10, Activation.SOFTMAX).seed(seed)
                .build();
        DpSgd dp = DpSgd.of(m, Loss.CROSS_ENTROPY).clipNorm(1).noiseMultiplier(sigma).learningRate(2)
                .samplingRate(q).seed(seed).train(train, steps);
        float[][] ty = new float[d[1].size()][10];
        for (int i = 0; i < ty.length; i++) {
            ty[i][d[1].y()[i]] = 1;
        }
        out.add(new Private("DP-SGD (example-level) eps " + budgets[0], dp.epsilon(DELTA),
                MachineUnlearning.accuracy(m, Dataset.of(d[1].x(), ty)), Double.NaN));
        return out;
    }

    private static Private dpFedAvg(String name, ClientSplits.Labelled[] d, int[][] parts, float[][] cx, int[] cy,
            boolean[] included, double sigma, int rounds, double q, long seed) {
        SimpleMlp global = net(10, seed);
        List<FederatedClient> fc = new ArrayList<>();
        for (int k = 0; k < parts.length; k++) {
            fc.add(client(net(10, seed), d[0].subset(parts[k]), 0.2f, seed * 1000 + k));
        }
        for (int c = 0; c < cx.length; c++) {
            if (included[c]) {
                ClientSplits.Labelled canary = new ClientSplits.Labelled(new float[][] {cx[c]}, new int[] {cy[c]}, 10);
                fc.add(client(net(10, seed), canary, 0.2f, seed * 7 + c));
            }
        }
        UserLevelDpFedAvg alg = UserLevelDpFedAvg.of(global.parameters(), fc, 1.0, sigma, q, seed);
        for (int r = 0; r < rounds; r++) {
            alg.round(5);
        }
        // the audit: score = log-probability of the canary's label
        Tensor logits = global.predict(Tensor.of(cx));
        float[] l = logits.toArray();
        double[] scores = new double[cx.length];
        for (int c = 0; c < cx.length; c++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < 10; j++) {
                max = Math.max(max, l[c * 10 + j]);
            }
            double z = 0;
            for (int j = 0; j < 10; j++) {
                z += Math.exp(l[c * 10 + j] - max);
            }
            scores[c] = l[c * 10 + cy[c]] - max - Math.log(z);
        }
        int guesses = Math.max(1, cx.length / 10);
        OneRunAudit.Result audit = OneRunAudit.audit(scores, included, guesses, guesses, 0.05);
        return new Private(name, alg.epsilon(DELTA), accuracy(global, d[1]), audit.epsilonLowerBound());
    }

    /**
     * Secure aggregation of random update vectors against dropout.
     *
     * @param clients the number of clients
     * @param dimension the vector length
     * @param dropouts the dropout rates
     * @param seed the seed
     * @return one row per rate
     */
    public static List<SecureRow> secureAggregation(int clients, int dimension, double[] dropouts, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[clients][dimension];
        for (float[] row : x) {
            for (int j = 0; j < dimension; j++) {
                row[j] = (float) r.nextGaussian() * 0.1f;
            }
        }
        List<SecureRow> out = new ArrayList<>();
        for (double p : dropouts) {
            boolean[] dropped = new boolean[clients];
            for (int k = 0; k < clients; k++) {
                dropped[k] = r.nextDouble() < p;
            }
            SecureAggregation.Result res = SecureAggregation.of(clients, clients / 2, 1 << 20, seed)
                    .aggregate(x, dropped);
            double err = 0;
            for (int j = 0; j < dimension; j++) {
                double plain = 0;
                for (int k = 0; k < clients; k++) {
                    plain += dropped[k] ? 0 : x[k][j];
                }
                err = Math.max(err, Math.abs(plain - res.sum()[j]));
            }
            out.add(new SecureRow(p, res.survivors(), err, res.recoveredSecrets(), res.nanos() / 1e6));
        }
        return out;
    }

    private static List<Tensor> snapshot(List<Var> params) {
        List<Tensor> out = new ArrayList<>();
        for (Var p : params) {
            out.add(p.value());
        }
        return out;
    }
}
