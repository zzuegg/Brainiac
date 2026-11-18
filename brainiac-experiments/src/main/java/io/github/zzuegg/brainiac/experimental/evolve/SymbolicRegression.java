package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Genetic programming for symbolic regression: a population of expression trees over
 * {@code + - * /} (protected), {@code sin}, {@code exp} (clamped), input variables and
 * constants evolves by subtree crossover, subtree mutation and point mutation, with
 * constant optimisation (hill climbing on an elite's constants), algebraic simplification
 * (constant folding, {@code x - x}, {@code x * 1} ...) and a parsimony pressure (a size
 * limit plus a per-node error charge).
 *
 * <p>Parents are picked by {@link Selection#LEXICASE} or {@link Selection#TOURNAMENT}.
 * Lexicase (here epsilon-lexicase, the variant for continuous errors) shuffles the
 * training cases and keeps, case by case, only the candidates within {@code epsilon} (the
 * median absolute deviation of that case's errors) of the best on it, until one is left.
 * Because no aggregate error is formed, a candidate that is exact on a subset of cases
 * survives, which keeps specialists and diversity; tournament selection on the mean
 * squared error is the ablation.
 *
 * <p>Papers: Koza, <i>Genetic Programming</i> (MIT Press 1992); Helmuth, Spector,
 * Matheson, "Solving Uncompromising Problems with Lexicase Selection" (IEEE TEC 2015,
 * <a href="https://ieeexplore.ieee.org/document/6920034">doi</a>); La Cava, Spector,
 * Danai, "Epsilon-Lexicase Selection for Regression" (GECCO 2016); Cranmer, "Interpretable
 * Machine Learning for Science with PySR and SymbolicRegression.jl" (2023),
 * <a href="https://arxiv.org/abs/2305.01582">arXiv:2305.01582</a>.
 */
public final class SymbolicRegression {

    /** Parent selection. */
    public enum Selection {
        /** Epsilon-lexicase over the training cases. */
        LEXICASE,
        /** Tournament of seven on mean squared error plus parsimony. */
        TOURNAMENT
    }

    /** An immutable expression tree. */
    public static final class Expr {
        final char op;   // + - * / s(in) e(xp) x(variable) c(onstant)
        final Expr a;
        final Expr b;
        final int variable;
        final double value;
        final int size;

        private Expr(char op, Expr a, Expr b, int variable, double value) {
            this.op = op;
            this.a = a;
            this.b = b;
            this.variable = variable;
            this.value = value;
            this.size = 1 + (a == null ? 0 : a.size) + (b == null ? 0 : b.size);
        }

        static Expr var(int i) {
            return new Expr('x', null, null, i, 0);
        }

        static Expr constant(double v) {
            return new Expr('c', null, null, 0, v);
        }

        static Expr unary(char op, Expr a) {
            return new Expr(op, a, null, 0, 0);
        }

        static Expr binary(char op, Expr a, Expr b) {
            return new Expr(op, a, b, 0, 0);
        }

        /**
         * Evaluates the expression.
         *
         * @param x the variables
         * @return its value (protected: division by near-zero gives 1, exp is clamped)
         */
        public double eval(double[] x) {
            return switch (op) {
                case 'x' -> x[variable];
                case 'c' -> value;
                case '+' -> a.eval(x) + b.eval(x);
                case '-' -> a.eval(x) - b.eval(x);
                case '*' -> a.eval(x) * b.eval(x);
                case '/' -> {
                    double d = b.eval(x);
                    yield Math.abs(d) < 1e-9 ? 1 : a.eval(x) / d;
                }
                case 's' -> Math.sin(a.eval(x));
                default -> Math.exp(Math.min(30, a.eval(x)));
            };
        }

        /** @return the number of nodes */
        public int size() {
            return size;
        }

        @Override
        public String toString() {
            return switch (op) {
                case 'x' -> "x" + variable;
                case 'c' -> String.format("%.4g", value);
                case 's' -> "sin(" + a + ")";
                case 'e' -> "exp(" + a + ")";
                default -> "(" + a + " " + op + " " + b + ")";
            };
        }
    }

    /**
     * A finished run.
     *
     * @param best the expression with the lowest training error
     * @param trainError its mean squared error on the training data
     * @param generations generations run (fewer when it hit zero error)
     */
    public record Fit(Expr best, double trainError, int generations) {
    }

    private final Selection selection;
    private final int population;
    private final int generations;
    private final int maxSize;
    private final double parsimony;

    /**
     * Configures the search.
     *
     * @param selection the parent selection
     * @param population the population size, such as 300
     * @param generations the most generations
     * @param maxSize the largest tree allowed
     * @param parsimony the error charged per node, such as 1e-4
     */
    public SymbolicRegression(Selection selection, int population, int generations, int maxSize, double parsimony) {
        this.selection = selection;
        this.population = population;
        this.generations = generations;
        this.maxSize = maxSize;
        this.parsimony = parsimony;
    }

    /**
     * Evolves an expression for {@code y = f(x)}.
     *
     * @param x the inputs {@code [n][vars]}
     * @param y the targets
     * @param seed the seed
     * @return the best expression
     */
    public Fit fit(double[][] x, double[] y, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int vars = x[0].length;
        List<Expr> pop = new ArrayList<>();
        for (int i = 0; i < population; i++) {
            pop.add(simplify(random(r, vars, 2 + i % 4, i % 2 == 0)));
        }
        Expr best = null;
        double bestErr = Double.POSITIVE_INFINITY;
        int g = 0;
        for (; g < generations; g++) {
            double[][] errors = new double[pop.size()][];
            double[] score = new double[pop.size()];
            for (int i = 0; i < pop.size(); i++) {
                errors[i] = caseErrors(pop.get(i), x, y);
                double mse = 0;
                for (double e : errors[i]) {
                    mse += e * e / y.length;
                }
                score[i] = Double.isFinite(mse) ? mse : Double.POSITIVE_INFINITY;
                if (score[i] + parsimony * pop.get(i).size() < bestErr + (best == null ? 0 : parsimony * best.size())) {
                    best = pop.get(i);
                    bestErr = score[i];
                }
            }
            if (bestErr < 1e-12) {
                break;
            }
            Expr tuned = tuneConstants(best, x, y, r);
            double tunedErr = mse(tuned, x, y);
            if (tunedErr < bestErr) {
                best = tuned;
                bestErr = tunedErr;
            }
            double[] eps = selection == Selection.LEXICASE ? epsilons(errors) : null;
            List<Expr> next = new ArrayList<>();
            next.add(best);
            while (next.size() < pop.size()) {
                Expr p = pick(pop, errors, score, eps, r);
                double u = r.nextDouble();
                Expr child;
                if (u < 0.6) {
                    child = crossover(p, pick(pop, errors, score, eps, r), r);
                } else if (u < 0.8) {
                    child = replaceAt(p, r.nextInt(p.size()), random(r, vars, 1 + r.nextInt(3), r.nextBoolean()));
                } else {
                    child = pointMutate(p, r, vars);
                }
                child = simplify(child);
                next.add(child.size() <= maxSize ? child : p);
            }
            pop = next;
        }
        return new Fit(best, bestErr, g);
    }

    private static double[] caseErrors(Expr e, double[][] x, double[] y) {
        double[] out = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            double v = e.eval(x[i]) - y[i];
            out[i] = Double.isFinite(v) ? Math.abs(v) : 1e12;
        }
        return out;
    }

    /**
     * Mean squared error of an expression.
     *
     * @param e the expression
     * @param x the inputs
     * @param y the targets
     * @return the error, {@code +infinity} if not finite
     */
    public static double mse(Expr e, double[][] x, double[] y) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            double v = e.eval(x[i]) - y[i];
            s += v * v / y.length;
        }
        return Double.isFinite(s) ? s : Double.POSITIVE_INFINITY;
    }

    /** Epsilon-lexicase's per-case tolerance: the median absolute deviation of the case's errors. */
    private static double[] epsilons(double[][] errors) {
        int cases = errors[0].length;
        double[] eps = new double[cases];
        double[] col = new double[errors.length];
        for (int c = 0; c < cases; c++) {
            for (int i = 0; i < errors.length; i++) {
                col[i] = errors[i][c];
            }
            double med = median(col);
            double[] dev = new double[col.length];
            for (int i = 0; i < col.length; i++) {
                dev[i] = Math.abs(col[i] - med);
            }
            eps[c] = median(dev);
        }
        return eps;
    }

    private static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        return s[s.length / 2];
    }

    private Expr pick(List<Expr> pop, double[][] errors, double[] score, double[] eps, SplittableRandom r) {
        if (selection == Selection.TOURNAMENT) {
            int best = -1;
            for (int t = 0; t < 7; t++) {
                int i = r.nextInt(pop.size());
                if (best < 0 || score[i] + parsimony * pop.get(i).size()
                        < score[best] + parsimony * pop.get(best).size()) {
                    best = i;
                }
            }
            return pop.get(best);
        }
        int cases = errors[0].length;
        int[] order = new int[cases];
        for (int i = 0; i < cases; i++) {
            order[i] = i;
        }
        for (int i = cases - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        List<Integer> pool = new ArrayList<>();
        for (int i = 0; i < pop.size(); i++) {
            pool.add(i);
        }
        for (int c : order) {
            if (pool.size() == 1) {
                break;
            }
            double best = Double.POSITIVE_INFINITY;
            for (int i : pool) {
                best = Math.min(best, errors[i][c]);
            }
            double limit = best + eps[c];
            List<Integer> kept = new ArrayList<>();
            for (int i : pool) {
                if (errors[i][c] <= limit) {
                    kept.add(i);
                }
            }
            pool = kept;
        }
        return pop.get(pool.get(r.nextInt(pool.size())));
    }

    private static Expr random(SplittableRandom r, int vars, int depth, boolean full) {
        if (depth <= 0 || (!full && r.nextDouble() < 0.3)) {
            return r.nextDouble() < 0.7 ? Expr.var(r.nextInt(vars)) : Expr.constant(Math.round(r.nextDouble() * 4 - 2));
        }
        double u = r.nextDouble();
        if (u < 0.15) {
            return Expr.unary(r.nextBoolean() ? 's' : 'e', random(r, vars, depth - 1, full));
        }
        char op = "+-*/".charAt(r.nextInt(4));
        return Expr.binary(op, random(r, vars, depth - 1, full), random(r, vars, depth - 1, full));
    }

    private static Expr nodeAt(Expr e, int index) {
        if (index == 0) {
            return e;
        }
        index--;
        if (e.a != null) {
            if (index < e.a.size) {
                return nodeAt(e.a, index);
            }
            index -= e.a.size;
        }
        return nodeAt(e.b, index);
    }

    private static Expr replaceAt(Expr e, int index, Expr with) {
        if (index == 0) {
            return with;
        }
        index--;
        if (e.a != null && index < e.a.size) {
            return new Expr(e.op, replaceAt(e.a, index, with), e.b, e.variable, e.value);
        }
        index -= e.a == null ? 0 : e.a.size;
        return new Expr(e.op, e.a, replaceAt(e.b, index, with), e.variable, e.value);
    }

    private static Expr crossover(Expr p, Expr q, SplittableRandom r) {
        return replaceAt(p, r.nextInt(p.size()), nodeAt(q, r.nextInt(q.size())));
    }

    private static Expr pointMutate(Expr p, SplittableRandom r, int vars) {
        int i = r.nextInt(p.size());
        Expr n = nodeAt(p, i);
        Expr m = switch (n.op) {
            case 'x' -> Expr.var(r.nextInt(vars));
            case 'c' -> Expr.constant(n.value + r.nextGaussian() * 0.5);
            case 's', 'e' -> Expr.unary(n.op == 's' ? 'e' : 's', n.a);
            default -> Expr.binary("+-*/".charAt(r.nextInt(4)), n.a, n.b);
        };
        return replaceAt(p, i, m);
    }

    /** Hill-climbs the constants of an expression on the training error. */
    private static Expr tuneConstants(Expr e, double[][] x, double[] y, SplittableRandom r) {
        List<Integer> constants = new ArrayList<>();
        for (int i = 0; i < e.size(); i++) {
            if (nodeAt(e, i).op == 'c') {
                constants.add(i);
            }
        }
        if (constants.isEmpty()) {
            return e;
        }
        Expr best = e;
        double err = mse(e, x, y);
        for (int it = 0; it < 30; it++) {
            int i = constants.get(r.nextInt(constants.size()));
            double v = nodeAt(best, i).value;
            double step = (Math.abs(v) + 0.1) * Math.exp(-r.nextDouble() * 6) * (r.nextBoolean() ? 1 : -1);
            Expr cand = replaceAt(best, i, Expr.constant(v + step));
            double ce = mse(cand, x, y);
            if (ce < err) {
                best = cand;
                err = ce;
            }
        }
        return best;
    }

    /**
     * Simplifies an expression: folds constant subtrees and removes identities such as
     * {@code e + 0}, {@code e * 1}, {@code e * 0}, {@code e - e} and {@code e / e}.
     *
     * @param e the expression
     * @return an equivalent, usually smaller, expression
     */
    public static Expr simplify(Expr e) {
        if (e.a == null) {
            return e;
        }
        Expr a = simplify(e.a);
        Expr b = e.b == null ? null : simplify(e.b);
        boolean ca = a.op == 'c';
        boolean cb = b != null && b.op == 'c';
        Expr folded = new Expr(e.op, a, b, e.variable, e.value);
        if (ca && (b == null || cb)) {
            double v = folded.eval(new double[0]);
            return Double.isFinite(v) ? Expr.constant(v) : folded;
        }
        String sa = a.toString();
        String sb = b == null ? "" : b.toString();
        return switch (e.op) {
            case '+' -> ca && a.value == 0 ? b : cb && b.value == 0 ? a : folded;
            case '-' -> cb && b.value == 0 ? a : sa.equals(sb) ? Expr.constant(0) : folded;
            case '*' -> (ca && a.value == 0) || (cb && b.value == 0) ? Expr.constant(0)
                    : ca && a.value == 1 ? b : cb && b.value == 1 ? a : folded;
            case '/' -> cb && b.value == 1 ? a : sa.equals(sb) ? Expr.constant(1) : folded;
            default -> folded;
        };
    }
}
