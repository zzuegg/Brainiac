package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Differentiable inductive logic programming (dILP) for one binary target predicate
 * {@code T(X, Y)} over binary background predicates. Candidate clauses come from three
 * templates, with body predicates drawn from the background and {@code T} itself (so
 * recursive definitions are reachable):
 * <ul>
 *   <li>{@code T(X,Y) :- P(X,Y)} (copy),</li>
 *   <li>{@code T(X,Y) :- P(Y,X)} (inverse),</li>
 *   <li>{@code T(X,Y) :- P(X,Z), Q(Z,Y)} (chain).</li>
 * </ul>
 * The definition has two clause slots, each a softmax over its candidates; as in the
 * paper's program templates, the first slot may not mention {@code T} (a base case) and
 * the second may (a recursive or a second base case). Forward chaining
 * runs for a fixed number of steps on fuzzy valuations in {@code [0, 1]}: a chain is
 * {@code max_Z P(X,Z) Q(Z,Y)}, the two slots and the previous valuation are combined by
 * the probabilistic sum {@code a + b - ab}. The loss is the cross-entropy of the final
 * valuation against the target relation; after training, the argmax clause of each slot
 * reads off as a logic program, which runs unchanged on larger worlds.
 *
 * <p>Paper: Evans and Grefenstette, "Learning Explanatory Rules from Noisy Data" (JAIR
 * 2018), <a href="https://arxiv.org/abs/1711.04574">arXiv:1711.04574</a>.
 */
public final class DifferentiableIlp {

    private record Clause(int kind, int p, int q) {
    }

    private final int background;
    private final int steps;
    private final List<Clause> candidates = new ArrayList<>();
    private final Var[] slots = new Var[2];
    private final float[] baseMask;

    /**
     * Creates the learner.
     *
     * @param background the number of background predicates
     * @param steps forward-chaining steps (recursion depth)
     * @param seed initialises the clause weights
     */
    public DifferentiableIlp(int background, int steps, long seed) {
        this.background = background;
        this.steps = steps;
        int preds = background + 1;   // the last one is the target itself
        for (int p = 0; p < preds; p++) {
            candidates.add(new Clause(0, p, -1));
            candidates.add(new Clause(1, p, -1));
            for (int q = 0; q < preds; q++) {
                candidates.add(new Clause(2, p, q));
            }
        }
        baseMask = new float[candidates.size()];
        for (int c = 0; c < candidates.size(); c++) {
            Clause cl = candidates.get(c);
            baseMask[c] = cl.p() == background || cl.q() == background ? -1e9f : 0;
        }
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < 2; i++) {
            float[] w = new float[candidates.size()];
            for (int c = 0; c < w.length; c++) {
                w[c] = (float) (0.1 * r.nextGaussian());
            }
            slots[i] = Var.param(Tensor.of(Shape.of(candidates.size()), w));
        }
    }

    private static Var probSum(Var a, Var b) {
        return a.add(b).sub(a.mul(b));
    }

    private static Var chain(Var p, Var q) {
        int n = p.shape().dim(0);
        return p.reshape(n, n, 1).mul(q.reshape(1, n, n)).max(1, false);
    }

    /**
     * Runs forward chaining.
     *
     * @param relations the background relations, each {@code [n, n]} with 0/1 entries
     * @return the target's valuation {@code [n, n]}
     */
    public Var forward(List<Tensor> relations) {
        int n = relations.get(0).shape().dim(0);
        Var[] w = new Var[2];
        for (int i = 0; i < 2; i++) {
            Var logits = i == 0 ? slots[i].add(Tensor.of(Shape.of(baseMask.length), baseMask)) : slots[i];
            w[i] = logits.sub(logits.logsumexp(0, true)).exp();
        }
        Var t = Var.of(Tensor.zeros(Shape.of(n, n)));
        for (int s = 0; s < steps; s++) {
            List<Var> preds = new ArrayList<>();
            for (Tensor rel : relations) {
                preds.add(Var.of(rel));
            }
            preds.add(t);
            Var[] slot = new Var[2];
            for (int c = 0; c < candidates.size(); c++) {
                Clause cl = candidates.get(c);
                Var value = switch (cl.kind()) {
                    case 0 -> preds.get(cl.p());
                    case 1 -> preds.get(cl.p()).transpose();
                    default -> chain(preds.get(cl.p()), preds.get(cl.q()));
                };
                for (int i = 0; i < 2; i++) {
                    Var term = value.mul(w[i].slice(0, c, c + 1));
                    slot[i] = slot[i] == null ? term : slot[i].add(term);
                }
            }
            t = probSum(t, probSum(slot[0], slot[1]));
        }
        return t;
    }

    /**
     * The binary cross-entropy of the valuation against a target relation.
     *
     * @param relations the background relations
     * @param target the target relation {@code [n, n]}
     * @return the mean loss
     */
    public Var loss(List<Tensor> relations, Tensor target) {
        Var p = forward(relations).clamp(1e-4f, 1 - 1e-4f);
        return p.log().mul(target).add(p.neg().add(1f).log().mul(target.neg().add(1f))).neg().mean();
    }

    private int argmax(int slot) {
        float[] w = slots[slot].value().toArray();
        int best = -1;
        for (int c = 0; c < w.length; c++) {
            float v = w[c] + (slot == 0 ? baseMask[c] : 0);
            if (best < 0 || v > w[best] + (slot == 0 ? baseMask[best] : 0)) {
                best = c;
            }
        }
        return best;
    }

    /** @return the two clause slots' weights */
    public List<Var> parameters() {
        return List.of(slots);
    }

    /**
     * Reads off the program: the most probable clause of each slot.
     *
     * @param names the background predicates' names
     * @return two clauses, such as {@code "T(X,Y) :- parent(X,Z), parent(Z,Y)"}
     */
    public List<String> program(String... names) {
        List<String> out = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            Clause cl = candidates.get(argmax(i));
            String p = cl.p() == background ? "T" : names[cl.p()];
            out.add(switch (cl.kind()) {
                case 0 -> "T(X,Y) :- " + p + "(X,Y)";
                case 1 -> "T(X,Y) :- " + p + "(Y,X)";
                default -> "T(X,Y) :- " + p + "(X,Z), " + (cl.q() == background ? "T" : names[cl.q()]) + "(Z,Y)";
            });
        }
        return out;
    }

    /**
     * Runs the extracted (argmax) program crisply: the logic program, not the soft mixture.
     *
     * @param relations the background relations
     * @return the derived target relation, 0/1 entries {@code [n][n]}
     */
    public boolean[][] runProgram(List<Tensor> relations) {
        int n = relations.get(0).shape().dim(0);
        List<boolean[][]> preds = new ArrayList<>();
        for (Tensor rel : relations) {
            float[] v = rel.toArray();
            boolean[][] b = new boolean[n][n];
            for (int i = 0; i < n * n; i++) {
                b[i / n][i % n] = v[i] > 0.5f;
            }
            preds.add(b);
        }
        boolean[][] t = new boolean[n][n];
        preds.add(t);
        Clause[] chosen = {candidates.get(argmax(0)), candidates.get(argmax(1))};
        for (int s = 0; s < Math.max(steps, n); s++) {
            boolean[][] next = new boolean[n][n];
            for (int x = 0; x < n; x++) {
                for (int y = 0; y < n; y++) {
                    boolean v = t[x][y];
                    for (Clause cl : chosen) {
                        boolean[][] p = preds.get(cl.p());
                        v |= switch (cl.kind()) {
                            case 0 -> p[x][y];
                            case 1 -> p[y][x];
                            default -> {
                                boolean any = false;
                                boolean[][] q = preds.get(cl.q());
                                for (int z = 0; z < n && !any; z++) {
                                    any = p[x][z] && q[z][y];
                                }
                                yield any;
                            }
                        };
                    }
                    next[x][y] = v;
                }
            }
            for (int x = 0; x < n; x++) {
                System.arraycopy(next[x], 0, t[x], 0, n);
            }
        }
        return t;
    }
}
