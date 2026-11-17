package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.DecoupledMomentum;
import io.github.zzuegg.brainiac.experimental.train.DiLoCo;
import io.github.zzuegg.brainiac.experimental.train.FederatedAveraging;
import io.github.zzuegg.brainiac.experimental.train.LocalWorker;
import io.github.zzuegg.brainiac.experimental.train.StreamingDiLoCo;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Loss against bytes communicated for simulated distributed training: four workers, each
 * holding a quarter of a three-arm spiral, train one small MLP. Compared are synchronous
 * data parallelism (gradients all-reduced in floats every step), {@link FederatedAveraging},
 * {@link DiLoCo}, {@link StreamingDiLoCo} (streamed fragments, 4-bit outer gradients and
 * overlapped communication, <a href="https://arxiv.org/abs/2501.18512">arXiv:2501.18512</a>)
 * and {@link DecoupledMomentum} (DeMo, <a href="https://arxiv.org/abs/2411.19870">arXiv:2411.19870</a>),
 * all with the same number of inner steps.
 */
public final class LowCommunicationTraining {
    private static final int WORKERS = 4;

    private LowCommunicationTraining() {
    }

    /**
     * One method's outcome.
     *
     * @param loss the final cross-entropy of the global model on all data
     * @param accuracy its accuracy on all data
     * @param bytes bytes sent and received by all workers together
     */
    public record Outcome(double loss, double accuracy, long bytes) {
    }

    private static final class Worker implements LocalWorker {
        private final SimpleMlp net;
        private final Tensor x;
        private final Tensor ids;
        private final Optimizer optimizer;

        Worker(SimpleMlp net, ToyTasks.Data shard, float lr) {
            this.net = net;
            this.x = shard.inputs();
            this.ids = shard.ids();
            this.optimizer = Optimizer.adam(lr);
        }

        @Override
        public List<Var> parameters() {
            return net.parameters();
        }

        @Override
        public int samples() {
            return x.shape().dim(0);
        }

        float gradient() {
            Var l = Functional.crossEntropy(net.forward(Var.of(x)), ids);
            l.backward();
            return l.value().item();
        }

        @Override
        public float train(int steps) {
            float sum = 0;
            for (int i = 0; i < steps; i++) {
                sum += gradient();
                optimizer.step(net.parameters());
            }
            return sum / steps;
        }
    }

    /**
     * Runs every method.
     *
     * @param steps inner steps per worker
     * @param period the synchronization period {@code H} of the local-update methods
     * @param seed the seed
     * @return the outcome per method
     */
    public static Map<String, Outcome> run(int steps, int period, long seed) {
        ToyTasks.Data data = ToyTasks.spiral(80, 3, 0.15, seed);
        Map<String, Outcome> out = new LinkedHashMap<>();
        long params = parameters(net(seed));

        // synchronous data parallelism: average gradients every step
        SimpleMlp dp = net(seed);
        List<Worker> dpWorkers = workers(data, seed, 0.01f);
        Optimizer adam = Optimizer.adam(0.01f);
        for (int s = 0; s < steps; s++) {
            List<Var> g = dp.parameters();
            Tensor[] sum = new Tensor[g.size()];
            for (Worker w : dpWorkers) {
                broadcast(g, w.parameters());
                w.gradient();
                for (int i = 0; i < sum.length; i++) {
                    Tensor gi = w.parameters().get(i).grad().div(WORKERS);
                    sum[i] = sum[i] == null ? gi : sum[i].add(gi);
                    w.parameters().get(i).setGrad(null);
                }
            }
            for (int i = 0; i < sum.length; i++) {
                g.get(i).setGrad(sum[i]);
            }
            adam.step(g);
        }
        out.put("data parallel", outcome(dp, data, (long) steps * WORKERS * 2 * params * 4));

        SimpleMlp fed = net(seed);
        FederatedAveraging fedAvg = new FederatedAveraging(fed.parameters(), localWorkers(data, seed), seed);
        for (int r = 0; r < steps / period; r++) {
            fedAvg.round(period, 1.0);
        }
        out.put("FedAvg", outcome(fed, data, fedAvg.transfers() * params * 4));

        SimpleMlp dil = net(seed);
        DiLoCo diloco = new DiLoCo(dil.parameters(), localWorkers(data, seed), DiLoCo.defaultOuter());
        for (int r = 0; r < steps / period; r++) {
            diloco.round(period);
        }
        out.put("DiLoCo", outcome(dil, data, diloco.transfers() * params * 4));

        SimpleMlp str = net(seed);
        StreamingDiLoCo streaming = new StreamingDiLoCo(str.parameters(), localWorkers(data, seed),
                DiLoCo.defaultOuter(), str.depth(), 4, 2, 0.5f);
        for (int s = 0; s < steps; s++) {
            streaming.step(period);
        }
        out.put("Streaming DiLoCo", outcome(str, data, streaming.bytes()));

        List<Worker> demoWorkers = workers(data, seed, 0.01f);
        DecoupledMomentum demo = new DecoupledMomentum(0.003f, 0.9f, 64, 2);
        List<List<Var>> replicas = new ArrayList<>();
        for (Worker w : demoWorkers) {
            replicas.add(w.parameters());
        }
        for (int s = 0; s < steps; s++) {
            for (Worker w : demoWorkers) {
                w.gradient();
            }
            demo.stepReplicas(replicas);
        }
        out.put("DeMo", outcome(demoWorkers.get(0).net, data, demo.bytes()));
        return out;
    }

    private static SimpleMlp net(long seed) {
        return SimpleMlp.tanh(seed, 2, 32, 32, 3);
    }

    private static long parameters(SimpleMlp net) {
        long n = 0;
        for (Var p : net.parameters()) {
            n += p.value().size();
        }
        return n;
    }

    private static List<Worker> workers(ToyTasks.Data data, long seed, float lr) {
        List<Worker> w = new ArrayList<>();
        int n = data.size();
        for (int k = 0; k < WORKERS; k++) {
            w.add(new Worker(net(seed), data.slice(k * n / WORKERS, (k + 1) * n / WORKERS), lr));
        }
        return w;
    }

    private static List<LocalWorker> localWorkers(ToyTasks.Data data, long seed) {
        return new ArrayList<>(workers(data, seed, 0.01f));
    }

    private static void broadcast(List<Var> from, List<Var> to) {
        for (int i = 0; i < from.size(); i++) {
            to.get(i).assign(from.get(i).value());
        }
    }

    private static Outcome outcome(SimpleMlp net, ToyTasks.Data data, long bytes) {
        Tensor logits = net.forward(Var.of(data.inputs())).value();
        double loss = Functional.crossEntropy(Var.of(logits), data.ids()).value().item();
        float[] l = logits.toArray();
        int correct = 0;
        for (int i = 0; i < data.size(); i++) {
            int best = 0;
            for (int c = 1; c < 3; c++) {
                best = l[i * 3 + c] > l[i * 3 + best] ? c : best;
            }
            correct += best == data.y()[i] ? 1 : 0;
        }
        return new Outcome(loss, (double) correct / data.size(), bytes);
    }
}
