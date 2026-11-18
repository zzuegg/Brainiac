package io.github.zzuegg.brainiac.experiments.numerics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.DecoupledMomentum;
import io.github.zzuegg.brainiac.experimental.train.GradientCompressor;
import io.github.zzuegg.brainiac.experimental.train.PowerSgd;
import io.github.zzuegg.brainiac.experimental.train.QuantizedGradients;
import io.github.zzuegg.brainiac.experimental.train.SparseGradientCompression;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.experiments.optim.LowCommunicationTraining;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Loss against bytes communicated for compressed data-parallel training (wave 7, items 7.3.9 to
 * 7.3.11), on the {@link LowCommunicationTraining} task: four workers, each holding a quarter of
 * a three-arm spiral, train one small MLP with Adam on the aggregated gradient every step. The
 * aggregate comes from a {@link GradientCompressor}: the float32 all-reduce, {@link PowerSgd},
 * {@link SparseGradientCompression} (top-k with and without error feedback, DGC, EF21) or
 * {@link QuantizedGradients} (QSGD, TernGrad); DeMo ({@link DecoupledMomentum}) is the existing
 * compressed-momentum baseline.
 */
public final class GradientCompressionShootout {

    private static final int WORKERS = 4;

    private GradientCompressionShootout() {
    }

    /**
     * One method's outcome.
     *
     * @param loss the final cross-entropy of the global model on all data
     * @param accuracy its accuracy
     * @param bytes bytes sent and received by all workers together
     */
    public record Outcome(double loss, double accuracy, long bytes) {
    }

    /**
     * The compressors compared, by name.
     *
     * @param seed the seed of their randomness
     * @return a fresh compressor per name
     */
    public static Map<String, Supplier<GradientCompressor>> methods(long seed) {
        Map<String, Supplier<GradientCompressor>> m = new LinkedHashMap<>();
        m.put("fp32 all-reduce", GradientCompressor::exact);
        m.put("PowerSGD r=1", () -> new PowerSgd(1, true, seed));
        m.put("PowerSGD r=2", () -> new PowerSgd(2, true, seed));
        m.put("PowerSGD r=1 no EF", () -> new PowerSgd(1, false, seed));
        m.put("top-1% no EF", () -> SparseGradientCompression.topK(0.01, false));
        m.put("top-1% EF", () -> SparseGradientCompression.topK(0.01, true));
        m.put("top-10% EF", () -> SparseGradientCompression.topK(0.1, true));
        m.put("top-0.1% EF", () -> SparseGradientCompression.topK(0.001, true));
        m.put("DGC 1%", () -> SparseGradientCompression.dgc(0.01, 0.9, 5.0, 50));
        m.put("EF21 1%", () -> SparseGradientCompression.ef21(0.01));
        m.put("QSGD s=1", () -> QuantizedGradients.qsgd(1, 128, seed));
        m.put("QSGD s=4", () -> QuantizedGradients.qsgd(4, 128, seed));
        m.put("QSGD s=16", () -> QuantizedGradients.qsgd(16, 128, seed));
        m.put("TernGrad", () -> QuantizedGradients.ternGrad(seed));
        return m;
    }

    /**
     * Runs every method with Adam as the outer optimizer, DeMo included.
     *
     * @param steps synchronous steps
     * @param seed the seed
     * @return the outcome per method, DeMo last
     */
    public static Map<String, Outcome> run(int steps, long seed) {
        List<String> all = new ArrayList<>(methods(seed).keySet());
        all.add("DeMo");
        return run(steps, seed, false, all);
    }

    /**
     * Runs chosen methods.
     *
     * @param steps synchronous steps
     * @param seed the seed
     * @param plainSgd whether the outer optimizer is plain SGD (0.3) instead of Adam (0.01);
     *     error feedback was designed for SGD, and DGC's momentum correction for momentum
     * @param names the methods, from {@link #methods} plus {@code "DeMo"}
     * @return the outcome per method
     */
    public static Map<String, Outcome> run(int steps, long seed, boolean plainSgd, List<String> names) {
        ToyTasks.Data data = ToyTasks.spiral(80, 3, 0.15, seed);
        Map<String, Outcome> out = new LinkedHashMap<>();
        for (Map.Entry<String, Supplier<GradientCompressor>> e : methods(seed).entrySet()) {
            if (!names.contains(e.getKey())) {
                continue;
            }
            GradientCompressor c = e.getValue().get();
            // DGC carries its own momentum: the outer optimizer then runs without one
            boolean dgc = e.getKey().startsWith("DGC");
            Optimizer opt = plainSgd ? Optimizer.sgd(0.3f)
                    : dgc ? Optimizer.adam(0.01f, 0f, 0.999f, 1e-8f) : Optimizer.adam(0.01f);
            SimpleMlp global = net(seed);
            List<SimpleMlp> replicas = new ArrayList<>();
            for (int w = 0; w < WORKERS; w++) {
                replicas.add(net(seed));
            }
            for (int s = 0; s < steps; s++) {
                List<Var> g = global.parameters();
                List<List<Tensor>> perParam = new ArrayList<>();
                for (int i = 0; i < g.size(); i++) {
                    perParam.add(new ArrayList<>());
                }
                for (int w = 0; w < WORKERS; w++) {
                    SimpleMlp rep = replicas.get(w);
                    List<Var> rp = rep.parameters();
                    for (int i = 0; i < g.size(); i++) {
                        rp.get(i).assign(g.get(i).value());
                    }
                    ToyTasks.Data shard = shard(data, w);
                    Functional.crossEntropy(rep.forward(Var.of(shard.inputs())), shard.ids()).backward();
                    for (int i = 0; i < g.size(); i++) {
                        perParam.get(i).add(rp.get(i).grad());
                        rp.get(i).setGrad(null);
                    }
                }
                for (int i = 0; i < g.size(); i++) {
                    g.get(i).setGrad(c.aggregate(g.get(i), perParam.get(i)));
                }
                opt.step(g);
            }
            out.put(e.getKey(), outcome(global, data, c.bytes()));
        }
        if (!names.contains("DeMo")) {
            return out;
        }
        // DeMo: replicas exchange sign-compressed DCT components of their momentum
        List<SimpleMlp> demoNets = new ArrayList<>();
        List<List<Var>> replicas = new ArrayList<>();
        for (int w = 0; w < WORKERS; w++) {
            SimpleMlp n = net(seed);
            demoNets.add(n);
            replicas.add(n.parameters());
        }
        DecoupledMomentum demo = new DecoupledMomentum(0.003f, 0.9f, 64, 2);
        for (int s = 0; s < steps; s++) {
            for (int w = 0; w < WORKERS; w++) {
                ToyTasks.Data shard = shard(data, w);
                Functional.crossEntropy(demoNets.get(w).forward(Var.of(shard.inputs())), shard.ids()).backward();
            }
            demo.stepReplicas(replicas);
        }
        out.put("DeMo", outcome(demoNets.get(0), data, demo.bytes()));
        return out;
    }

    /**
     * The relative variance {@code E|Q(g) - g|^2 / |g|^2} of each quantizer on a real gradient
     * of the task's network at initialization.
     *
     * @param seed the seed
     * @return the variance per quantizer
     */
    public static Map<String, Double> quantizerVariance(long seed) {
        ToyTasks.Data data = ToyTasks.spiral(80, 3, 0.15, seed);
        SimpleMlp n = net(seed);
        Functional.crossEntropy(n.forward(Var.of(data.inputs())), data.ids()).backward();
        Tensor g = n.weight(1).grad();
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("QSGD s=1", QuantizedGradients.qsgd(1, 128, seed).relativeVariance(g, 50));
        out.put("QSGD s=4", QuantizedGradients.qsgd(4, 128, seed).relativeVariance(g, 50));
        out.put("QSGD s=16", QuantizedGradients.qsgd(16, 128, seed).relativeVariance(g, 50));
        out.put("TernGrad", QuantizedGradients.ternGrad(seed).relativeVariance(g, 50));
        return out;
    }

    private static SimpleMlp net(long seed) {
        return SimpleMlp.tanh(seed, 2, 32, 32, 3);
    }

    private static ToyTasks.Data shard(ToyTasks.Data data, int w) {
        int n = data.size();
        return data.slice(w * n / WORKERS, (w + 1) * n / WORKERS);
    }

    private static Outcome outcome(SimpleMlp net, ToyTasks.Data data, long bytes) {
        return new Outcome(NumericsTasks.loss(net, data), NumericsTasks.accuracy(net, data), bytes);
    }
}
