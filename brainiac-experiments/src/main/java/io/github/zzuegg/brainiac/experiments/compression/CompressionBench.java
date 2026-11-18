package io.github.zzuegg.brainiac.experiments.compression;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;
import java.util.function.UnaryOperator;

/**
 * The fixed reference models every compression experiment starts from, so all pruning and
 * quantisation rows of one seed begin at the same weights: a ReLU MLP {@code 64-64-64-10} on
 * the {@link ToyTasks#digits} task (10 classes, 8x8, 10% flipped pixels), trained once per seed
 * and cached, plus the training and evaluation helpers the experiments share.
 */
public final class CompressionBench {

    /** The layer sizes of the digits MLP. */
    public static final int[] SIZES = {64, 64, 64, 10};

    /** Full-batch Adam steps of the reference training. */
    public static final int REFERENCE_STEPS = 120;

    private static final Map<Long, List<Tensor>> TRAINED = new ConcurrentHashMap<>();

    /**
     * A train and test split.
     *
     * @param train the training samples
     * @param test held-out samples of the same classes
     */
    public record Split(ToyTasks.Data train, ToyTasks.Data test) {
    }

    private CompressionBench() {
    }

    /**
     * @param seed the prototype seed
     * @return 40 training and 30 test samples per class
     */
    public static Split digits(long seed) {
        return new Split(ToyTasks.digits(10, 40, 8, 0.1, seed, seed * 31 + 7),
                ToyTasks.digits(10, 30, 8, 0.1, seed, seed * 31 + 1009));
    }

    /**
     * @param seed the seed
     * @return a freshly initialised digits MLP
     */
    public static SimpleMlp untrained(long seed) {
        return SimpleMlp.relu(seed, SIZES);
    }

    /**
     * The reference model: trained once per seed for {@link #REFERENCE_STEPS} Adam steps and
     * cached; every call returns an independent copy.
     *
     * @param seed the seed
     * @return a trained digits MLP
     */
    public static SimpleMlp trained(long seed) {
        List<Tensor> weights = TRAINED.computeIfAbsent(seed, s -> {
            SimpleMlp net = untrained(s);
            train(net::forward, net.parameters(), Optimizer.adam(0.01f), digits(s).train(), REFERENCE_STEPS);
            List<Tensor> w = new ArrayList<>();
            for (Var p : net.parameters()) {
                w.add(p.value());
            }
            return w;
        });
        SimpleMlp net = untrained(seed);
        List<Var> p = net.parameters();
        for (int i = 0; i < p.size(); i++) {
            p.get(i).assign(weights.get(i));
        }
        return net;
    }

    /**
     * Full-batch cross-entropy training.
     *
     * @param forward the logits of a batch
     * @param params the parameters the optimizer updates
     * @param optimizer the optimizer
     * @param data the data
     * @param steps the steps
     */
    public static void train(UnaryOperator<Var> forward, List<Var> params, Optimizer optimizer, ToyTasks.Data data,
            int steps) {
        Tensor x = data.inputs();
        Tensor ids = data.ids();
        for (int s = 0; s < steps; s++) {
            Functional.crossEntropy(forward.apply(Var.of(x)), ids).backward();
            optimizer.step(params);
        }
    }

    /**
     * @param forward the logits of a batch
     * @param data the data
     * @return the accuracy
     */
    public static double accuracy(UnaryOperator<Var> forward, ToyTasks.Data data) {
        return ToyTasks.accuracy(Var.noGrad(() -> forward.apply(Var.of(data.inputs())).value()), data.y());
    }

    /**
     * @param forward the logits of a batch
     * @param data the data
     * @return the mean cross-entropy
     */
    public static double loss(UnaryOperator<Var> forward, ToyTasks.Data data) {
        return Var.noGrad(() -> Functional.crossEntropy(forward.apply(Var.of(data.inputs())), data.ids()).value()
                .item());
    }

    /**
     * The MLP's forward pass with every weight passed through a transform first (a mask, a
     * fake quantiser, a straight-through estimator) and an optional activation transform.
     *
     * @param net the network
     * @param weight maps layer index and weight to the weight the layer uses
     * @param activation maps layer index and hidden pre-activation to its output; null for the
     *     network's own activation
     * @return the forward function
     */
    public static UnaryOperator<Var> forward(SimpleMlp net, BiFunction<Integer, Var, Var> weight,
            BiFunction<Integer, Var, Var> activation) {
        return x -> {
            Var h = x;
            for (int l = 0; l < net.depth(); l++) {
                Var z = h.matmul(weight.apply(l, net.weight(l))).add(net.bias(l));
                h = l == net.depth() - 1 ? z
                        : activation == null ? net.activation().apply(z) : activation.apply(l, z);
            }
            return h;
        };
    }

    /**
     * @param net the network
     * @return its weight matrices, layer by layer
     */
    public static List<Var> weights(SimpleMlp net) {
        List<Var> w = new ArrayList<>();
        for (int l = 0; l < net.depth(); l++) {
            w.add(net.weight(l));
        }
        return w;
    }

    /**
     * @param weights some weights
     * @return the share of exact zeros over all of them
     */
    public static double sparsity(List<Var> weights) {
        long zero = 0;
        long total = 0;
        for (Var w : weights) {
            for (float v : w.value().toArray()) {
                zero += v == 0 ? 1 : 0;
                total++;
            }
        }
        return (double) zero / total;
    }
}
