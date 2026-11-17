package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
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
     * Runs one meta step over a batch of tasks.
     *
     * @param tasks the tasks
     * @return the mean query loss after adaptation, before the meta step
     */
    public float metaStep(List<? extends Task> tasks) {
        List<Tensor> start = values(parameters);
        Tensor[] meta = new Tensor[parameters.size()];
        double total = 0;
        for (Task task : tasks) {
            List<List<Tensor>> path = new ArrayList<>();
            List<Tensor> w = start;
            for (int s = 0; s < innerSteps; s++) {
                path.add(w);
                w = sgd(w, gradient(w, task::supportLoss));
            }
            assign(w);
            Var q = task.queryLoss();
            total += q.value().item();
            Tensor[] g = backward(q);
            if (secondOrder) {
                for (int s = innerSteps - 1; s >= 0; s--) {
                    Tensor[] hv = hessianVector(path.get(s), g, task::supportLoss);
                    for (int i = 0; i < g.length; i++) {
                        g[i] = g[i].sub(hv[i].mul(innerRate));
                    }
                }
            }
            for (int i = 0; i < g.length; i++) {
                meta[i] = meta[i] == null ? g[i] : meta[i].add(g[i]);
            }
        }
        assign(start);
        for (int i = 0; i < parameters.size(); i++) {
            parameters.get(i).setGrad(meta[i].div(tasks.size()));
        }
        outer.step(parameters);
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
            out.add(w.get(i).sub(g[i].mul(innerRate)));
        }
        return out;
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
