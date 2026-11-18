package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.WeightPruning;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Continual learning by parameter isolation: every task gets its own subset of the network,
 * and later tasks may read but never change it, so old tasks cannot be forgotten at all. Both
 * methods need the task identity at test time (task-incremental learning).
 *
 * <ul>
 *   <li>{@link PackNet} (Mallya and Lazebnik 2018, <a href="https://arxiv.org/abs/1711.05769">arXiv:1711.05769</a>):
 *       train the free weights on the new task, prune the smallest of them by magnitude
 *       ({@link WeightPruning#magnitude}), give the survivors to the task and freeze them,
 *       briefly retrain the survivors, release the pruned ones for the next task. Task
 *       {@code t} runs with the weights of tasks {@code 0..t}.</li>
 *   <li>{@link Hat} (hard attention to the task, Serra et al. 2018,
 *       <a href="https://arxiv.org/abs/1801.01423">arXiv:1801.01423</a>): a learned, almost
 *       binary gate {@code sigmoid(s e_t)} per task on every hidden unit. Gradients to a
 *       weight are scaled by {@code 1 - min(a_in, a_out)} of the union of earlier tasks'
 *       gates, and a sparsity penalty keeps each task's gate small, so capacity is spent
 *       only as needed.</li>
 * </ul>
 */
public final class ParameterIsolation {

    private ParameterIsolation() {
    }

    private static Tensor scalar(float v) {
        return Tensor.full(Shape.of(), v);
    }

    /**
     * Creates PackNet over the given parameters.
     *
     * @param parameters the parameters; matrices are packed, vectors (biases) train on the
     *     first task only
     * @param keep the share of the free weights each task keeps, such as 0.5
     * @return PackNet
     */
    public static PackNet packNet(List<Var> parameters, double keep) {
        return new PackNet(parameters, keep);
    }

    /** PackNet: iterative pruning and freezing of a weight subset per task. */
    public static final class PackNet {

        private final List<Var> parameters;
        private final double keep;
        private final List<Tensor> owner = new ArrayList<>();
        private List<Tensor> saved;
        private int task;
        private boolean retraining;

        private PackNet(List<Var> parameters, double keep) {
            this.parameters = List.copyOf(parameters);
            this.keep = keep;
            for (Var p : this.parameters) {
                owner.add(Tensor.zeros(p.shape()));
            }
        }

        /**
         * Starts training a task: only free weights train.
         *
         * @param id the task, counting from 0
         */
        public void startTask(int id) {
            this.task = id;
            this.retraining = false;
        }

        private Tensor trainable(int i) {
            Var p = parameters.get(i);
            if (p.shape().rank() < 2) {
                return Tensor.full(p.shape(), task == 0 && !retraining ? 1f : 0f);
            }
            return owner.get(i).equalTo(scalar(retraining ? task + 1 : 0));
        }

        /**
         * Wraps an optimizer so that it changes only the weights the current phase may train.
         *
         * @param inner the optimizer
         * @return the masked optimizer
         */
        public Optimizer wrap(Optimizer inner) {
            return new Optimizer() {
                @Override
                public void step(List<Var> ps) {
                    List<Tensor> before = new ArrayList<>();
                    for (Var p : parameters) {
                        before.add(p.value());
                    }
                    inner.step(ps);
                    for (int i = 0; i < parameters.size(); i++) {
                        Var p = parameters.get(i);
                        p.assign(Tensor.where(trainable(i), p.value(), before.get(i)));
                    }
                }

                @Override
                public float learningRate() {
                    return inner.learningRate();
                }

                @Override
                public void learningRate(float learningRate) {
                    inner.learningRate(learningRate);
                }
            };
        }

        /**
         * Prunes the free weights after training the task: the largest {@code keep} share
         * (per output unit) joins the task, the rest is zeroed and stays free. Then the
         * task's survivors may be retrained.
         */
        public void prune() {
            prune(keep);
        }

        /**
         * {@link #prune()} with a task-specific keep share, for example {@code 1 / (tasks - t)}
         * to give every task the same capacity.
         *
         * @param share the share of the free weights this task keeps
         */
        public void prune(double share) {
            for (int i = 0; i < parameters.size(); i++) {
                Var p = parameters.get(i);
                if (p.shape().rank() < 2) {
                    continue;
                }
                Tensor free = owner.get(i).equalTo(scalar(0));
                int in = p.shape().dim(0);
                int out = p.shape().dim(1);
                double freePerColumn = free.sum().item() / out;
                long kept = Math.round(share * freePerColumn);
                Tensor pruned = WeightPruning.magnitude(p.value().mask(free), 1 - kept / (double) in);
                Tensor mine = pruned.abs().greater(scalar(0)).mask(free);
                owner.set(i, owner.get(i).add(mine.mul(task + 1)));
                // the released weights restart from zero; other tasks' weights stay as they are
                p.assign(Tensor.where(free, pruned, p.value()));
            }
            retraining = true;
        }

        /**
         * Runs the network as task {@code id} would: weights of later tasks and free weights
         * are zeroed until {@link #restore()}.
         *
         * @param id the task
         */
        public void use(int id) {
            saved = new ArrayList<>();
            for (int i = 0; i < parameters.size(); i++) {
                Var p = parameters.get(i);
                saved.add(p.value());
                if (p.shape().rank() >= 2) {
                    Tensor o = owner.get(i);
                    Tensor visible = o.greater(scalar(0.5f)).mask(o.less(scalar(id + 1.5f)));
                    p.assign(p.value().mask(visible));
                }
            }
        }

        /** Undoes {@link #use(int)}. */
        public void restore() {
            for (int i = 0; i < parameters.size(); i++) {
                parameters.get(i).assign(saved.get(i));
            }
            saved = null;
        }

        /**
         * The share of the packed weights a task owns.
         *
         * @param id the task
         * @return the share in {@code [0, 1]}
         */
        public double capacity(int id) {
            double mine = 0;
            double total = 0;
            for (int i = 0; i < parameters.size(); i++) {
                if (parameters.get(i).shape().rank() >= 2) {
                    mine += owner.get(i).equalTo(scalar(id + 1)).sum().item();
                    total += owner.get(i).shape().size();
                }
            }
            return mine / total;
        }
    }

    /**
     * Creates a HAT network: an MLP with per-task hard-attention gates on its hidden units and
     * a shared output layer whose columns freeze with the task that owns them.
     *
     * @param inputs input features
     * @param hidden hidden widths
     * @param outputs output units (all classes)
     * @param tasks the most tasks it will learn
     * @param random the initializer
     * @return the network
     */
    public static Hat hat(int inputs, int[] hidden, int outputs, int tasks, RandomGenerator random) {
        return new Hat(inputs, hidden, outputs, tasks, random);
    }

    /** Hard attention to the task. */
    public static final class Hat {

        /** The largest gate slope, at which the gates are practically binary. */
        public static final float S_MAX = 400f;

        private final List<Var> weights = new ArrayList<>();
        private final List<Var> biases = new ArrayList<>();
        private final List<Var> embeddings = new ArrayList<>();
        private final List<Tensor> cumulative = new ArrayList<>();
        private final int tasks;
        private final int outputs;
        private final float[] frozenOutputs;
        private float sparsity = 0.05f;

        private Hat(int inputs, int[] hidden, int outputs, int tasks, RandomGenerator random) {
            this.tasks = tasks;
            this.outputs = outputs;
            int in = inputs;
            for (int h : hidden) {
                weights.add(Var.param(Tensor.normal(Shape.of(in, h), 0f, (float) Math.sqrt(2.0 / in), random)));
                biases.add(Var.param(Tensor.zeros(Shape.of(h))));
                embeddings.add(Var.param(Tensor.normal(Shape.of(tasks, h), 0f, 0.1f, random)));
                cumulative.add(Tensor.zeros(Shape.of(1, h)));
                in = h;
            }
            weights.add(Var.param(Tensor.normal(Shape.of(in, outputs), 0f, (float) Math.sqrt(1.0 / in), random)));
            biases.add(Var.param(Tensor.zeros(Shape.of(outputs))));
            this.frozenOutputs = new float[outputs];
        }

        /**
         * Sets the weight of the gate sparsity penalty.
         *
         * @param lambda the weight; the paper uses 0.75 with SGD, the default 0.05 suits Adam
         * @return this
         */
        public Hat sparsity(float lambda) {
            this.sparsity = lambda;
            return this;
        }

        /**
         * The annealed gate slope for batch {@code b} of {@code batches} in an epoch, from
         * {@code 1/S_MAX} to {@code S_MAX}.
         *
         * @param b the batch, from 0
         * @param batches batches per epoch
         * @return the slope
         */
        public static float slope(int b, int batches) {
            return 1 / S_MAX + (S_MAX - 1 / S_MAX) * b / Math.max(1, batches - 1);
        }

        private Var gate(int layer, int task, float s) {
            return embeddings.get(layer).slice(0, task, task + 1).mul(s).sigmoid();
        }

        /**
         * The logits for a task.
         *
         * @param x inputs {@code [batch, inputs]}
         * @param task the task
         * @param s the gate slope; {@link #S_MAX} for inference
         * @return {@code [batch, outputs]}
         */
        public Var forward(Var x, int task, float s) {
            Var h = x;
            for (int l = 0; l < embeddings.size(); l++) {
                h = h.matmul(weights.get(l)).add(biases.get(l)).relu().mul(gate(l, task, s));
            }
            int last = weights.size() - 1;
            return h.matmul(weights.get(last)).add(biases.get(last));
        }

        /**
         * The gate sparsity penalty: the mean gate value on units no earlier task uses.
         *
         * @param task the task being trained
         * @param s the gate slope
         * @return a scalar to add to the loss
         */
        public Var regularizer(int task, float s) {
            Var num = null;
            double den = 0;
            for (int l = 0; l < embeddings.size(); l++) {
                Tensor free = cumulative.get(l).neg().add(1f);
                Var term = gate(l, task, s).mul(free).sum();
                num = num == null ? term : num.add(term);
                den += free.sum().item();
            }
            return num.div((float) Math.max(den, 1e-6)).mul(sparsity);
        }

        /** @return every trainable parameter */
        public List<Var> parameters() {
            List<Var> out = new ArrayList<>(weights);
            out.addAll(biases);
            out.addAll(embeddings);
            return out;
        }

        /**
         * Wraps an optimizer so that it respects earlier tasks' gates: weight and bias
         * gradients into units earlier tasks use are scaled down (to zero for binary gates),
         * frozen output columns do not move, and the embeddings stay in {@code [-6, 6]}.
         *
         * <p>The gate embeddings get their own optimizer: the paper uses SGD with a gradient
         * compensation for the annealed slope, which an adaptive optimizer with a larger step
         * size replaces (Adam's steps do not shrink with the gradient's scale).
         *
         * @param inner the optimizer of the weights and biases
         * @param gates the optimizer of the gate embeddings
         * @return the constrained optimizer, which steps both
         */
        public Optimizer wrap(Optimizer inner, Optimizer gates) {
            return new Optimizer() {
                @Override
                public void step(List<Var> ps) {
                    int hidden = embeddings.size();
                    List<Tensor> allowW = new ArrayList<>();
                    List<Tensor> allowB = new ArrayList<>();
                    for (int l = 0; l <= hidden; l++) {
                        Tensor zeros = Tensor.zeros(weights.get(l).shape());
                        Tensor outUse = l < hidden ? cumulative.get(l)
                                : Tensor.of(Shape.of(1, outputs), frozenOutputs);
                        // inputs are always "used"; hidden-to-hidden weights take min(a_in, a_out);
                        // output columns freeze with their task
                        Tensor block = outUse.add(zeros);
                        if (l > 0 && l < hidden) {
                            block = block.minimum(cumulative.get(l - 1).transpose().add(zeros));
                        }
                        allowW.add(block.neg().add(1f));
                        allowB.add(outUse.reshape(outUse.shape().dim(1)).neg().add(1f));
                    }
                    List<Tensor> before = new ArrayList<>();
                    for (int l = 0; l <= hidden; l++) {
                        Var w = weights.get(l);
                        Var b = biases.get(l);
                        before.add(w.value());
                        before.add(b.value());
                        if (w.grad() != null) {
                            w.setGrad(w.grad().mul(allowW.get(l)));
                        }
                        if (b.grad() != null) {
                            b.setGrad(b.grad().mul(allowB.get(l)));
                        }
                    }
                    List<Var> net = new ArrayList<>(weights);
                    net.addAll(biases);
                    inner.step(net);
                    gates.step(embeddings);
                    for (int l = 0; l <= hidden; l++) {
                        // momentum must not move what the gradient mask protects
                        Var w = weights.get(l);
                        Var b = biases.get(l);
                        Tensor aw = allowW.get(l);
                        Tensor ab = allowB.get(l);
                        w.assign(before.get(2 * l).add(w.value().sub(before.get(2 * l)).mul(aw)));
                        b.assign(before.get(2 * l + 1).add(b.value().sub(before.get(2 * l + 1)).mul(ab)));
                    }
                    for (Var e : embeddings) {
                        e.assign(e.value().minimum(Tensor.full(e.shape(), 6f)).maximum(Tensor.full(e.shape(), -6f)));
                    }
                }

                @Override
                public float learningRate() {
                    return inner.learningRate();
                }

                @Override
                public void learningRate(float learningRate) {
                    inner.learningRate(learningRate);
                }
            };
        }

        /**
         * Ends a task: its gates join the union of used units, and its output columns freeze.
         *
         * @param task the task
         * @param classes the output columns the task owns
         */
        public void finishTask(int task, int[] classes) {
            for (int l = 0; l < embeddings.size(); l++) {
                cumulative.set(l, cumulative.get(l).maximum(gate(l, task, S_MAX).value()));
            }
            for (int c : classes) {
                frozenOutputs[c] = 1;
            }
        }

        /**
         * The share of hidden units a task's gates open.
         *
         * @param task the task
         * @return the share in {@code [0, 1]}
         */
        public double capacity(int task) {
            double open = 0;
            double total = 0;
            for (int l = 0; l < embeddings.size(); l++) {
                Tensor g = gate(l, task, S_MAX).value();
                open += g.greater(scalar(0.5f)).sum().item();
                total += g.shape().size();
            }
            return open / total;
        }

        /** @return the most tasks */
        public int tasks() {
            return tasks;
        }
    }
}
