package io.github.zzuegg.brainiac.experimental.diffopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Hypergradients of a validation loss with respect to hyperparameters that enter the training
 * loss (per-parameter weight decay, per-example data weights, augmentation strengths), on
 * autograd {@link Var}s with double backpropagation.
 *
 * <p>At a training optimum {@code w*(lambda)} the implicit function theorem gives
 * {@code dL_val/dlambda = dL_val/dlambda|_direct - (d^2 L_train / dlambda dw) H^-1 dL_val/dw}
 * with {@code H} the training Hessian (Pedregosa, ICML 2016; Lorraine, Vicol and Duvenaud,
 * AISTATS 2020, <a href="https://arxiv.org/abs/1911.02590">arXiv:1911.02590</a>). The
 * inverse-Hessian-vector product is approximated by
 * <ul>
 *   <li>{@link Method#IFT_NEUMANN}: the truncated Neumann series
 *       {@code H^-1 v ~ alpha sum_{j=0}^{K} (I - alpha H)^j v};</li>
 *   <li>{@link Method#IFT_CG}: {@code K} conjugate-gradient iterations;</li>
 * </ul>
 * each needing only Hessian-vector products, so memory is constant in the number of
 * hyperparameters. The alternatives differentiate through the inner optimisation itself:
 * {@link Method#UNROLLED} backpropagates through all {@code K} inner gradient steps,
 * {@link Method#TRUNCATED} only through the last {@code T} of them (Shaban et al., AISTATS 2019).
 */
public final class Hypergradients {

    /** How the hypergradient is computed. */
    public enum Method {
        /** Implicit function theorem with a Neumann-series inverse. */
        IFT_NEUMANN,
        /** Implicit function theorem with a conjugate-gradient inverse. */
        IFT_CG,
        /** Backpropagation through all inner steps. */
        UNROLLED,
        /** Backpropagation through the last few inner steps. */
        TRUNCATED
    }

    /**
     * The bilevel problem.
     *
     * @param trainLoss builds {@code L_train(w, lambda)} from the weights and hyperparameters
     * @param validationLoss builds {@code L_val(w)}
     */
    public record Problem(BiFunction<List<Var>, Var, Var> trainLoss, Function<List<Var>, Var> validationLoss) {

        /**
         * Checks the functions.
         *
         * @param trainLoss the training loss
         * @param validationLoss the validation loss
         */
        public Problem {
            Objects.requireNonNull(trainLoss, "trainLoss");
            Objects.requireNonNull(validationLoss, "validationLoss");
        }
    }

    private final Method method;
    private final int terms;
    private final double alpha;
    private final int innerSteps;
    private final double innerLearningRate;
    private final int truncation;

    private Hypergradients(Method method, int terms, double alpha, int innerSteps, double innerLearningRate,
            int truncation) {
        this.method = Objects.requireNonNull(method, "method");
        this.terms = terms;
        this.alpha = alpha;
        this.innerSteps = innerSteps;
        this.innerLearningRate = innerLearningRate;
        this.truncation = truncation;
    }

    /**
     * An implicit-function-theorem hypergradient.
     *
     * @param method {@link Method#IFT_NEUMANN} or {@link Method#IFT_CG}
     * @param terms Neumann terms or CG iterations, for example 10
     * @param alpha the Neumann step, below {@code 1 / lambda_max(H)}
     * @param innerSteps gradient steps on the training loss before each hypergradient (warm start)
     * @param innerLearningRate the inner step size
     * @return the estimator
     */
    public static Hypergradients implicit(Method method, int terms, double alpha, int innerSteps,
            double innerLearningRate) {
        if (method != Method.IFT_NEUMANN && method != Method.IFT_CG) {
            throw new IllegalArgumentException("not an implicit method: " + method);
        }
        return new Hypergradients(method, terms, alpha, innerSteps, innerLearningRate, 0);
    }

    /**
     * An unrolled or truncated hypergradient through {@code innerSteps} gradient steps.
     *
     * @param innerSteps inner steps per hypergradient
     * @param innerLearningRate the inner step size
     * @param truncation steps to backpropagate through ({@code innerSteps} for full unrolling)
     * @return the estimator
     */
    public static Hypergradients unrolled(int innerSteps, double innerLearningRate, int truncation) {
        return new Hypergradients(truncation >= innerSteps ? Method.UNROLLED : Method.TRUNCATED, 0, 0, innerSteps,
                innerLearningRate, truncation);
    }

    /**
     * The method.
     *
     * @return the method
     */
    public Method method() {
        return method;
    }

    /**
     * Advances the inner optimisation (updating {@code weights} in place) and returns the
     * hypergradient at the new weights.
     *
     * @param problem the bilevel problem
     * @param weights the model weights
     * @param hyper the hyperparameters
     * @return {@code dL_val / dlambda}, shaped like {@code hyper}
     */
    public Tensor hypergradient(Problem problem, List<Var> weights, Var hyper) {
        if (method == Method.UNROLLED || method == Method.TRUNCATED) {
            return unroll(problem, weights, hyper);
        }
        Var lam = Var.of(hyper.value());
        for (int s = 0; s < innerSteps; s++) {
            List<Var> g = Var.gradients(problem.trainLoss().apply(weights, lam), weights, false);
            for (int i = 0; i < weights.size(); i++) {
                Var w = weights.get(i);
                w.assign(w.value().sub(g.get(i).value().mul((float) innerLearningRate)));
            }
        }
        List<Var> fixed = detached(weights);
        List<Tensor> v = values(Var.gradients(problem.validationLoss().apply(fixed), fixed, false), fixed);
        List<Tensor> p = method == Method.IFT_NEUMANN ? neumann(problem, fixed, lam, v) : cg(problem, fixed, lam, v);
        // mixed term: d/dlambda (grad_w L_train . p)
        Var lamParam = Var.param(hyper.value());
        List<Var> fw = detached(weights);
        List<Var> gw = Var.gradients(problem.trainLoss().apply(fw, lamParam), fw, true);
        Var dot = dot(gw, p);
        Var mixed = Var.gradients(dot, List.of(lamParam), false).get(0);
        Tensor direct = mixed == null ? Tensor.zeros(hyper.shape()) : mixed.value();
        return direct.neg();
    }

    private static List<Var> detached(List<Var> ws) {
        List<Var> out = new ArrayList<>();
        for (Var w : ws) {
            out.add(Var.param(w.value()));
        }
        return out;
    }

    private static List<Tensor> values(List<Var> grads, List<Var> like) {
        List<Tensor> out = new ArrayList<>();
        for (int i = 0; i < grads.size(); i++) {
            out.add(grads.get(i) == null ? Tensor.zeros(like.get(i).shape()) : grads.get(i).value());
        }
        return out;
    }

    private static Var dot(List<Var> a, List<Tensor> b) {
        Var s = null;
        for (int i = 0; i < a.size(); i++) {
            if (a.get(i) == null) {
                continue;
            }
            Var t = a.get(i).mul(b.get(i)).sum();
            s = s == null ? t : s.add(t);
        }
        return s;
    }

    /** {@code H v} at fixed weights. */
    private static List<Tensor> hvp(Problem problem, List<Var> weights, Var lam, List<Tensor> v) {
        List<Var> ws = detached(weights);
        List<Var> g = Var.gradients(problem.trainLoss().apply(ws, lam), ws, true);
        return values(Var.gradients(dot(g, v), ws, false), ws);
    }

    private List<Tensor> neumann(Problem problem, List<Var> w, Var lam, List<Tensor> v) {
        List<Tensor> cur = new ArrayList<>(v);
        List<Tensor> sum = new ArrayList<>(v);
        for (int j = 0; j < terms; j++) {
            List<Tensor> hv = hvp(problem, w, lam, cur);
            for (int i = 0; i < cur.size(); i++) {
                cur.set(i, cur.get(i).sub(hv.get(i).mul((float) alpha)));
                sum.set(i, sum.get(i).add(cur.get(i)));
            }
        }
        for (int i = 0; i < sum.size(); i++) {
            sum.set(i, sum.get(i).mul((float) alpha));
        }
        return sum;
    }

    private List<Tensor> cg(Problem problem, List<Var> w, Var lam, List<Tensor> b) {
        List<Tensor> x = new ArrayList<>();
        List<Tensor> r = new ArrayList<>(b);
        List<Tensor> p = new ArrayList<>(b);
        for (Tensor t : b) {
            x.add(Tensor.zeros(t.shape()));
        }
        double rr = sq(r);
        for (int it = 0; it < terms && rr > 1e-20; it++) {
            List<Tensor> ap = hvp(problem, w, lam, p);
            double pap = 0;
            for (int i = 0; i < p.size(); i++) {
                pap += dotT(p.get(i), ap.get(i));
            }
            double a = rr / pap;
            for (int i = 0; i < x.size(); i++) {
                x.set(i, x.get(i).add(p.get(i).mul((float) a)));
                r.set(i, r.get(i).sub(ap.get(i).mul((float) a)));
            }
            double rn = sq(r);
            for (int i = 0; i < p.size(); i++) {
                p.set(i, r.get(i).add(p.get(i).mul((float) (rn / rr))));
            }
            rr = rn;
        }
        return x;
    }

    private static double sq(List<Tensor> t) {
        double s = 0;
        for (Tensor x : t) {
            s += dotT(x, x);
        }
        return s;
    }

    private static double dotT(Tensor a, Tensor b) {
        double[] x = a.toDoubleArray();
        double[] y = b.toDoubleArray();
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            s += x[i] * y[i];
        }
        return s;
    }

    private Tensor unroll(Problem problem, List<Var> weights, Var hyper) {
        Var lam = Var.param(hyper.value());
        List<Var> cur = new ArrayList<>();
        for (Var w : weights) {
            cur.add(Var.of(w.value()));
        }
        int start = innerSteps - truncation;
        for (int s = 0; s < innerSteps; s++) {
            boolean track = s >= start;
            if (s == start) {
                List<Var> fresh = new ArrayList<>();
                for (Var w : cur) {
                    fresh.add(Var.param(w.value()));
                }
                cur = fresh;
            }
            List<Var> current = cur;
            if (track) {
                List<Var> g = Var.gradients(problem.trainLoss().apply(current, lam), current, true);
                List<Var> next = new ArrayList<>();
                for (int i = 0; i < current.size(); i++) {
                    next.add(g.get(i) == null ? current.get(i)
                            : current.get(i).sub(g.get(i).mul((float) innerLearningRate)));
                }
                cur = next;
            } else {
                List<Var> ps = new ArrayList<>();
                for (Var w : current) {
                    ps.add(Var.param(w.value()));
                }
                List<Var> g = Var.gradients(problem.trainLoss().apply(ps, Var.of(lam.value())), ps, false);
                List<Var> next = new ArrayList<>();
                for (int i = 0; i < ps.size(); i++) {
                    next.add(Var.of(ps.get(i).value().sub(g.get(i).value().mul((float) innerLearningRate))));
                }
                cur = next;
            }
        }
        Var val = problem.validationLoss().apply(cur);
        Var g = Var.gradients(val, List.of(lam), false).get(0);
        for (int i = 0; i < weights.size(); i++) {
            weights.get(i).assign(cur.get(i).value());
        }
        return g == null ? Tensor.zeros(hyper.shape()) : g.value();
    }
}
