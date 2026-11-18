package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * A predictive-coding network on an arbitrary graph of regions (Whittington and Bogacz
 * 2017; Salvatori et al. 2022, <a href="https://arxiv.org/abs/2201.13180">arXiv:2201.13180</a>;
 * survey <a href="https://arxiv.org/abs/2407.04117">arXiv:2407.04117</a>). Every region
 * holds value neurons {@code x_r}; each region with incoming connections predicts its
 * activity from its sources, {@code mu_r = sum_s f(x_s) W_sr + b_r}, and an error neuron
 * holds the mismatch {@code e_r = x_r - mu_r}. Learning minimises the energy
 * {@code F = 1/2 sum_r |e_r|^2} in two local phases:
 *
 * <ol>
 *   <li><b>Inference</b>: with the input (and, while training, the label) clamped, the free
 *       value neurons relax by gradient descent on {@code F}:
 *       {@code x_r -= rate (e_r - f'(x_r) * sum_t e_t W_rt^T)}. Every term is available
 *       at the neuron: its own error and the errors it causes downstream.</li>
 *   <li><b>Learning</b>: each weight follows the purely local, Hebbian-like
 *       {@code dW_sr = f(x_s)^T e_r}.</li>
 * </ol>
 *
 * <p>On a feed-forward chain this approximates backpropagation; on graphs with loops,
 * such as a {@link Brain}'s region graph ({@link #fromBrain}), it still learns, which
 * backprop cannot do without unrolling. Region update rules are applied with an
 * {@link Optimizer} by writing {@code dF/dW} into the weights' gradients.
 *
 * <p><b>Incremental mode</b> ({@link Builder#incremental}; Salvatori et al., "A Stable,
 * Fast, and Fully Automatic Learning Algorithm for Predictive Coding Networks", ICLR 2024,
 * <a href="https://arxiv.org/abs/2212.00720">arXiv:2212.00720</a>): instead of relaxing to
 * equilibrium and then taking one weight step, the value neurons and the weights are
 * updated together at every inference step, an incremental EM. It needs no tuning of the
 * inference length and is reported to stay stable on cyclic graphs.
 */
public final class PredictiveCodingNetwork implements LocalLearner {

    /** The sending nonlinearity of a region. */
    public enum Nonlinearity {
        /** f(x) = x */
        LINEAR,
        /** f(x) = tanh(x) */
        TANH,
        /** f(x) = max(0, x) */
        RELU
    }

    private record Edge(int from, int to, Var weight, Tensor mask) {
    }

    private final List<String> names;
    private final int[] sizes;
    private final Nonlinearity[] nonlinearities;
    private final List<Edge> edges;
    private final Var[] biases;
    private final int input;
    private final int output;
    private final int steps;
    private final float rate;
    private final Optimizer optimizer;
    private final boolean incremental;
    private float lastEnergy;

    private PredictiveCodingNetwork(Builder b) {
        names = List.copyOf(b.names);
        sizes = b.sizes.stream().mapToInt(Integer::intValue).toArray();
        nonlinearities = b.nonlinearities.toArray(Nonlinearity[]::new);
        input = names.indexOf(b.input);
        output = names.indexOf(b.output);
        if (input < 0 || output < 0 || input == output) {
            throw new IllegalArgumentException("set distinct input and output regions");
        }
        steps = b.steps;
        rate = b.rate;
        incremental = b.incremental;
        optimizer = b.optimizer == null ? Optimizer.adam(0.005f) : b.optimizer;
        SplittableRandom r = new SplittableRandom(b.seed);
        edges = new ArrayList<>();
        int[] fanIn = new int[sizes.length];
        for (Object[] c : b.connections) {
            fanIn[names.indexOf((String) c[1])] += sizes[names.indexOf((String) c[0])];
        }
        for (Object[] c : b.connections) {
            int from = names.indexOf((String) c[0]);
            int to = names.indexOf((String) c[1]);
            double density = (Double) c[2];
            float std = (float) Math.sqrt(1.0 / (fanIn[to] * density));
            Tensor w = Tensor.normal(Shape.of(sizes[from], sizes[to]), 0, std, r);
            Tensor mask = null;
            if (density < 1) {
                float[] m = new float[sizes[from] * sizes[to]];
                for (int i = 0; i < m.length; i++) {
                    m[i] = r.nextDouble() < density ? 1 : 0;
                }
                mask = Tensor.of(w.shape(), m);
                w = w.mul(mask);
            }
            edges.add(new Edge(from, to, Var.param(w), mask));
        }
        biases = new Var[sizes.length];
        for (int i = 0; i < sizes.length; i++) {
            biases[i] = Var.param(Tensor.zeros(Shape.of(1, sizes[i])));
        }
    }

    /** @return a builder for a region graph */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * A layered network {@code input -> h1 -> ... -> output} with tanh hidden regions.
     *
     * @param seed the seed
     * @param optimizer the weight optimizer
     * @param sizes the layer sizes, input first, at least two
     * @return the network
     */
    public static PredictiveCodingNetwork layered(long seed, Optimizer optimizer, int... sizes) {
        Builder b = builder().seed(seed).optimizer(optimizer).region("input", sizes[0], Nonlinearity.LINEAR);
        String prev = "input";
        for (int i = 1; i < sizes.length; i++) {
            String name = i == sizes.length - 1 ? "output" : "h" + i;
            b.region(name, sizes[i], Nonlinearity.TANH).connect(prev, name);
            prev = name;
        }
        return b.input("input").output("output").build();
    }

    /**
     * A predictive-coding network with the regions and (sparse, possibly recurrent)
     * projections of a {@link Brain} spec, plus an {@code output} region of class scores
     * predicted by the brain's output regions. Each projection keeps its density as a fixed
     * random connection mask; regions use tanh.
     *
     * @param spec the brain's layout
     * @param inputs the input size
     * @param classes the number of classes
     * @param seed the seed
     * @param optimizer the weight optimizer
     * @return the network
     */
    public static PredictiveCodingNetwork fromBrain(Brain.Spec spec, int inputs, int classes, long seed,
            Optimizer optimizer) {
        return fromBrain(spec, inputs, classes, seed, optimizer, false);
    }

    /**
     * {@link #fromBrain(Brain.Spec, int, int, long, Optimizer)} with a choice of standard or
     * incremental predictive coding.
     *
     * @param spec the brain's layout
     * @param inputs the input size
     * @param classes the number of classes
     * @param seed the seed
     * @param optimizer the weight optimizer
     * @param incremental update the weights at every inference step ({@link Builder#incremental})
     * @return the network
     */
    public static PredictiveCodingNetwork fromBrain(Brain.Spec spec, int inputs, int classes, long seed,
            Optimizer optimizer, boolean incremental) {
        Builder b = builder().seed(seed).optimizer(optimizer).incremental(incremental)
                .region(Brain.INPUT, inputs, Nonlinearity.LINEAR);
        for (Brain.RegionSpec r : spec.regions()) {
            b.region(r.name(), r.capacity(), Nonlinearity.TANH);
        }
        b.region("output", classes, Nonlinearity.LINEAR);
        for (Brain.ProjectionSpec p : spec.projections()) {
            b.connect(p.from(), p.to(), p.density());
        }
        for (String o : spec.outputs()) {
            b.connect(o, "output");
        }
        return b.input(Brain.INPUT).output("output").build();
    }

    /** Collects regions and connections. */
    public static final class Builder {
        private final List<String> names = new ArrayList<>();
        private final List<Integer> sizes = new ArrayList<>();
        private final List<Nonlinearity> nonlinearities = new ArrayList<>();
        private final List<Object[]> connections = new ArrayList<>();
        private String input;
        private String output;
        private int steps = 20;
        private float rate = 0.1f;
        private long seed = 1;
        private Optimizer optimizer;
        private boolean incremental;

        private Builder() {
        }

        /**
         * Switches to incremental predictive coding: one weight update after every inference
         * step of a training batch, instead of one after the whole relaxation.
         *
         * @param on whether to update weights at every inference step
         * @return this builder
         */
        public Builder incremental(boolean on) {
            this.incremental = on;
            return this;
        }

        /**
         * @param name a unique name
         * @param size its neurons
         * @param f its sending nonlinearity
         * @return this builder
         */
        public Builder region(String name, int size, Nonlinearity f) {
            if (names.contains(name)) {
                throw new IllegalArgumentException("region " + name + " exists");
            }
            names.add(name);
            sizes.add(size);
            nonlinearities.add(f);
            return this;
        }

        /**
         * A dense connection: {@code from} helps predict {@code to}.
         *
         * @param from the source region
         * @param to the predicted region
         * @return this builder
         */
        public Builder connect(String from, String to) {
            return connect(from, to, 1);
        }

        /**
         * A sparse connection with a fixed random mask.
         *
         * @param from the source region
         * @param to the predicted region
         * @param density the share of connections that exist, in {@code (0, 1]}
         * @return this builder
         */
        public Builder connect(String from, String to, double density) {
            if (!names.contains(from) || !names.contains(to)) {
                throw new IllegalArgumentException("unknown region in " + from + " -> " + to);
            }
            connections.add(new Object[] {from, to, density});
            return this;
        }

        /**
         * @param name the region clamped to the input
         * @return this builder
         */
        public Builder input(String name) {
            this.input = name;
            return this;
        }

        /**
         * @param name the region clamped to the one-hot label in training and read out
         * @return this builder
         */
        public Builder output(String name) {
            this.output = name;
            return this;
        }

        /**
         * @param steps inference steps per batch
         * @param rate the inference step size
         * @return this builder
         */
        public Builder inference(int steps, float rate) {
            this.steps = steps;
            this.rate = rate;
            return this;
        }

        /**
         * @param seed the seed
         * @return this builder
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * @param optimizer the weight optimizer (Adam 0.005 by default)
         * @return this builder
         */
        public Builder optimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        /** @return the network */
        public PredictiveCodingNetwork build() {
            return new PredictiveCodingNetwork(this);
        }
    }

    /** @return the region names */
    public List<String> regions() {
        return names;
    }

    /** @return the weights and biases */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Edge e : edges) {
            p.add(e.weight());
        }
        for (int i = 0; i < biases.length; i++) {
            if (hasIncoming(i)) {
                p.add(biases[i]);
            }
        }
        return p;
    }

    /** @return the energy at the end of the last inference */
    public float lastEnergy() {
        return lastEnergy;
    }

    private boolean hasIncoming(int region) {
        for (Edge e : edges) {
            if (e.to() == region) {
                return true;
            }
        }
        return false;
    }

    private Tensor send(int region, Tensor x) {
        return switch (nonlinearities[region]) {
            case LINEAR -> x;
            case TANH -> x.tanh();
            case RELU -> x.relu();
        };
    }

    private Tensor sendDerivative(int region, Tensor x) {
        return switch (nonlinearities[region]) {
            case LINEAR -> Tensor.full(Shape.of(), 1f);
            case TANH -> LearningMath.tanhDerivative(x.tanh());
            case RELU -> x.step();
        };
    }

    private Tensor[] predictions(Tensor[] x) {
        Tensor[] mu = new Tensor[x.length];
        for (Edge e : edges) {
            Tensor term = send(e.from(), x[e.from()]).matmul(e.weight().value());
            mu[e.to()] = mu[e.to()] == null ? term : mu[e.to()].add(term);
        }
        for (int i = 0; i < mu.length; i++) {
            if (mu[i] != null) {
                mu[i] = mu[i].add(biases[i].value());
            }
        }
        return mu;
    }

    /**
     * Relaxes the value neurons.
     *
     * @param inputs the clamped input
     * @param target the clamped output, or {@code null} to leave it free
     * @return the settled values of every region
     */
    public Map<String, Tensor> settle(Tensor inputs, Tensor target) {
        Tensor[] x = relax(inputs, target);
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (int i = 0; i < x.length; i++) {
            out.put(names.get(i), x[i]);
        }
        return out;
    }

    private Tensor[] relax(Tensor inputs, Tensor target) {
        return relax(inputs, target, steps);
    }

    private Tensor[] relax(Tensor inputs, Tensor target, int inferenceSteps) {
        int batch = inputs.shape().dim(0);
        Tensor[] x = new Tensor[sizes.length];
        for (int i = 0; i < x.length; i++) {
            x[i] = Tensor.zeros(Shape.of(batch, sizes[i]));
        }
        x[input] = inputs;
        // initial sweep in declaration order: a feed-forward pass on chains
        for (int i = 0; i < x.length; i++) {
            if (i == input || !hasIncoming(i)) {
                continue;
            }
            Tensor mu = null;
            for (Edge e : edges) {
                if (e.to() == i) {
                    Tensor term = send(e.from(), x[e.from()]).matmul(e.weight().value());
                    mu = mu == null ? term : mu.add(term);
                }
            }
            x[i] = mu.add(biases[i].value());
        }
        if (target != null) {
            x[output] = target;
        }
        for (int s = 0; s < inferenceSteps; s++) {
            inferenceStep(x, target != null);
        }
        return x;
    }

    /** Clamps nothing new; moves every free value neuron one step down the energy. */
    private void inferenceStep(Tensor[] x, boolean clampOutput) {
        {
            Tensor[] err = errors(x);
            Tensor[] down = new Tensor[x.length];
            for (Edge e : edges) {
                if (err[e.to()] != null) {
                    Tensor t = err[e.to()].matmul(e.weight().value().transpose());
                    down[e.from()] = down[e.from()] == null ? t : down[e.from()].add(t);
                }
            }
            for (int i = 0; i < x.length; i++) {
                if (i == input || (i == output && clampOutput)) {
                    continue;
                }
                Tensor grad = err[i] == null ? null : err[i];
                if (down[i] != null) {
                    Tensor d = down[i].mul(sendDerivative(i, x[i]));
                    grad = grad == null ? d.neg() : grad.sub(d);
                }
                if (grad != null) {
                    x[i] = x[i].sub(grad.mul(rate));
                }
            }
            Tensor.evaluate(List.of(x));
        }
    }

    private Tensor[] errors(Tensor[] x) {
        Tensor[] mu = predictions(x);
        Tensor[] err = new Tensor[x.length];
        for (int i = 0; i < x.length; i++) {
            if (mu[i] != null) {
                err[i] = x[i].sub(mu[i]);
            }
        }
        return err;
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        int batch = inputs.shape().dim(0);
        Tensor target = LearningMath.oneHot(labels, sizes[output]);
        if (incremental) {
            Tensor[] x = relax(inputs, target, 0);
            for (int s = 0; s < steps; s++) {
                inferenceStep(x, true);
                weightStep(x, batch);
            }
            return lastEnergy;
        }
        return weightStep(relax(inputs, target), batch);
    }

    /** One local weight update at the current values; returns the energy before it. */
    private float weightStep(Tensor[] x, int batch) {
        Tensor[] err = errors(x);
        double energy = 0;
        for (Tensor e : err) {
            if (e != null) {
                energy += 0.5 * e.square().sum().item() / batch;
            }
        }
        lastEnergy = (float) energy;
        for (Edge e : edges) {
            Tensor g = send(e.from(), x[e.from()]).transpose().matmul(err[e.to()]).div(-batch);
            e.weight().setGrad(e.mask() == null ? g : g.mul(e.mask()));
        }
        for (int i = 0; i < biases.length; i++) {
            if (err[i] != null) {
                biases[i].setGrad(err[i].sum(0, true).div(-batch));
            }
        }
        optimizer.step(parameters());
        return lastEnergy;
    }

    @Override
    public Tensor predict(Tensor inputs) {
        return relax(inputs, null)[output];
    }
}
