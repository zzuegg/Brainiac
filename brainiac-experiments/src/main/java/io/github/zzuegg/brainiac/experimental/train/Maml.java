package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;

/**
 * Model-agnostic meta-learning: learns an initialization from which a few gradient steps on
 * a new task's small support set already fit that task. Each meta step adapts the weights to
 * every task with {@code innerSteps} SGD steps on its support loss, measures the adapted
 * weights' query loss, and moves the initialization along the gradient of that query loss
 * with respect to the <em>initial</em> weights.
 *
 * <p>Works with any model: the tasks compute their losses with the model's own parameter
 * {@link Var}s, and this class swaps the adapted values into them ({@link Var#assign}) and
 * back. Two flavours of meta-gradient:
 *
 * <ul>
 *   <li><b>First order</b> (FOMAML): the query gradient at the adapted weights, ignoring how
 *       the adaptation depends on the start. Cheap and usually almost as good.</li>
 *   <li><b>Second order</b> (full MAML): the query gradient back-propagated through the
 *       inner steps, {@code (I - a H_k) ... (I - a H_1) g}, with each Hessian-vector product
 *       {@code H v} exact from nested autograd ({@link Var#gradients(Var, List, boolean)}: the
 *       gradient of {@code grad . v}). The losses' operations need second derivatives.</li>
 * </ul>
 *
 * <p>Two variants switch on with fluent flags, and combine:
 *
 * <ul>
 *   <li><b>ANIL</b> ({@link #anil(Collection)}, "almost no inner loop"): only the chosen
 *       parameters, usually the head, adapt in the inner loop; the body is still meta-trained
 *       through the outer step. Raghu et al., "Rapid Learning or Feature Reuse?" (2019),
 *       <a href="https://arxiv.org/abs/1909.09157">arXiv:1909.09157</a>.</li>
 *   <li><b>Meta-SGD</b> ({@link #metaSgd()}): every parameter gets its own learned inner
 *       learning rate (a tensor of the parameter's shape, starting at {@code innerRate}),
 *       meta-trained along with the initialization: the inner step is
 *       {@code w - alpha * grad}. Li et al., "Meta-SGD" (2017),
 *       <a href="https://arxiv.org/abs/1707.09835">arXiv:1707.09835</a>.</li>
 * </ul>
 *
 * <p>Paper: Finn, Abbeel, Levine, "Model-Agnostic Meta-Learning for Fast Adaptation of Deep
 * Networks" (2017), <a href="https://arxiv.org/abs/1703.03400">arXiv:1703.03400</a>.
 */
public final class Maml {

    /** One task: a support loss to adapt on and a query loss to judge the adaptation. */
    public interface Task {

        /**
         * Computes the loss on the support set with the model's current parameters.
         *
         * @return a scalar
         */
        Var supportLoss();

        /**
         * Computes the loss on the query set with the model's current parameters.
         *
         * @return a scalar
         */
        Var queryLoss();
    }

    private final List<Var> parameters;
    private final Optimizer outer;
    private final float innerRate;
    private final int innerSteps;
    private final boolean secondOrder;
    private Set<Var> adaptable;
    private List<Var> rates;

    /**
     * Creates the meta-learner.
     *
     * @param parameters the model parameters (the initialization being learned)
     * @param outer the optimizer for the meta step, such as Adam
     * @param innerRate the SGD rate of the inner adaptation
     * @param innerSteps the adaptation steps per task
     * @param secondOrder whether to differentiate through the inner steps
     */
    public Maml(List<Var> parameters, Optimizer outer, float innerRate, int innerSteps, boolean secondOrder) {
        this.parameters = List.copyOf(parameters);
        this.outer = outer;
        this.innerRate = innerRate;
        this.innerSteps = innerSteps;
        this.secondOrder = secondOrder;
    }

    /**
     * ANIL: adapts only these parameters in the inner loop (and in {@link #adapt}); the
     * others stay at their meta-learned values for every task.
     *
     * @param inner the parameters to adapt, a subset of the model's, usually the head
     * @return this
     */
    public Maml anil(Collection<Var> inner) {
        Set<Var> set = Collections.newSetFromMap(new IdentityHashMap<>());
        set.addAll(inner);
        for (Var v : set) {
            if (parameters.stream().noneMatch(p -> p == v)) {
                throw new IllegalArgumentException("an inner parameter is not one of the model's parameters");
            }
        }
        this.adaptable = set;
        return this;
    }

    /**
     * Meta-SGD: learns one inner learning rate per weight, starting at {@code innerRate}; the
     * outer optimizer trains them with the initialization.
     *
     * @return this
     */
    public Maml metaSgd() {
        List<Var> r = new ArrayList<>();
        for (Var p : parameters) {
            r.add(Var.param(Tensor.full(p.shape(), innerRate).to(p.value().device())));
        }
        this.rates = List.copyOf(r);
        return this;
    }

    /** @return the learned per-weight inner rates of {@link #metaSgd()}, or an empty list */
    public List<Var> rates() {
        return rates == null ? List.of() : rates;
    }

    /** @return the number of weights the inner loop adapts per task */
    public long adaptedWeights() {
        long n = 0;
        for (Var p : parameters) {
            n += adapts(p) ? p.shape().size() : 0;
        }
        return n;
    }

    private boolean adapts(Var p) {
        return adaptable == null || adaptable.contains(p);
    }

    /**
     * Runs one meta step over a batch of tasks.
     *
     * @param tasks the tasks
     * @return the mean query loss after adaptation, before the meta step
     */
    public float metaStep(List<? extends Task> tasks) {
        List<Tensor> start = values(parameters);
        Tensor[] meta = new Tensor[parameters.size()];
        Tensor[] metaRates = rates == null ? null : new Tensor[parameters.size()];
        double total = 0;
        for (Task task : tasks) {
            List<List<Tensor>> path = new ArrayList<>();
            List<Tensor[]> grads = new ArrayList<>();
            List<Tensor> w = start;
            for (int s = 0; s < innerSteps; s++) {
                path.add(w);
                Tensor[] g = gradient(w, task::supportLoss);
                grads.add(g);
                w = sgd(w, g);
            }
            assign(w);
            Var q = task.queryLoss();
            total += q.value().item();
            Tensor[] g = backward(q);
            for (int s = innerSteps - 1; s >= 0; s--) {
                if (metaRates != null) {
                    Tensor[] inner = grads.get(s);
                    for (int i = 0; i < g.length; i++) {
                        if (adapts(parameters.get(i))) {
                            Tensor d = inner[i].mul(g[i]).neg();
                            metaRates[i] = metaRates[i] == null ? d : metaRates[i].add(d);
                        }
                    }
                }
                if (secondOrder) {
                    Tensor[] v = new Tensor[g.length];
                    for (int i = 0; i < g.length; i++) {
                        v[i] = adapts(parameters.get(i)) ? g[i].mul(rate(i))
                                : Tensor.zeros(g[i].shape()).to(g[i].device());
                    }
                    Tensor[] hv = hessianVector(path.get(s), v, task::supportLoss);
                    for (int i = 0; i < g.length; i++) {
                        g[i] = g[i].sub(hv[i]);
                    }
                }
            }
            for (int i = 0; i < g.length; i++) {
                meta[i] = meta[i] == null ? g[i] : meta[i].add(g[i]);
            }
        }
        assign(start);
        List<Var> stepped = new ArrayList<>(parameters);
        for (int i = 0; i < parameters.size(); i++) {
            parameters.get(i).setGrad(meta[i].div(tasks.size()));
            if (metaRates != null) {
                Var r = rates.get(i);
                r.setGrad(metaRates[i] == null ? Tensor.zeros(r.shape()).to(r.value().device())
                        : metaRates[i].div(tasks.size()));
                stepped.add(r);
            }
        }
        outer.step(stepped);
        return (float) (total / tasks.size());
    }

    /**
     * Adapts the model to a task in place with the inner SGD steps (for evaluation); call
     * {@link #restore(List)} with {@link #snapshot()} to undo.
     *
     * @param task the task
     * @param steps the adaptation steps
     */
    public void adapt(Task task, int steps) {
        List<Tensor> w = values(parameters);
        for (int s = 0; s < steps; s++) {
            w = sgd(w, gradient(w, task::supportLoss));
        }
        assign(w);
    }

    /** @return the current parameter values */
    public List<Tensor> snapshot() {
        return values(parameters);
    }

    /**
     * Restores parameter values from a {@link #snapshot()}.
     *
     * @param values the values
     */
    public void restore(List<Tensor> values) {
        assign(values);
    }

    private List<Tensor> sgd(List<Tensor> w, Tensor[] g) {
        List<Tensor> out = new ArrayList<>(w.size());
        for (int i = 0; i < w.size(); i++) {
            out.add(adapts(parameters.get(i)) ? w.get(i).sub(g[i].mul(rate(i))) : w.get(i));
        }
        return out;
    }

    private Tensor rate(int i) {
        if (rates != null) {
            return rates.get(i).value();
        }
        Var p = parameters.get(i);
        return Tensor.full(Shape.of(1), innerRate).to(p.value().device());
    }

    private Tensor[] gradient(List<Tensor> at, Supplier<Var> loss) {
        assign(at);
        return backward(loss.get());
    }

    private Tensor[] backward(Var loss) {
        for (Var p : parameters) {
            p.zeroGrad();
        }
        loss.backward();
        Tensor[] g = new Tensor[parameters.size()];
        for (int i = 0; i < g.length; i++) {
            Var p = parameters.get(i);
            g[i] = p.grad() == null ? Tensor.zeros(p.shape()).to(p.value().device()) : p.grad();
            p.zeroGrad();
        }
        return g;
    }

    /** {@code H(at) v}, exactly, by differentiating {@code grad . v} (core nested autograd). */
    private Tensor[] hessianVector(List<Tensor> at, Tensor[] v, Supplier<Var> loss) {
        assign(at);
        List<Var> g = Var.gradients(loss.get(), parameters, true);
        Var dot = null;
        for (int i = 0; i < v.length; i++) {
            Var term = g.get(i).mul(Var.of(v[i])).sum();
            dot = dot == null ? term : dot.add(term);
        }
        List<Var> hv = Var.gradients(dot, parameters, false);
        Tensor[] out = new Tensor[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = hv.get(i).value();
        }
        return out;
    }

    private void assign(List<Tensor> w) {
        for (int i = 0; i < parameters.size(); i++) {
            parameters.get(i).assign(w.get(i));
        }
    }

    private static List<Tensor> values(List<Var> params) {
        List<Tensor> out = new ArrayList<>(params.size());
        for (Var p : params) {
            out.add(p.value());
        }
        return out;
    }
}
