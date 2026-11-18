package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.DifferentiableIlp;
import io.github.zzuegg.brainiac.experimental.nn.GinLayer;
import io.github.zzuegg.brainiac.experimental.nn.NeuralLogicMachine;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Learning first-order rules from relational examples and running them on larger worlds:
 * differentiable ILP ({@link DifferentiableIlp}) and Neural Logic Machines
 * ({@link NeuralLogicMachine}) against a graph network (two GIN layers per edge
 * direction and a pair MLP). Every model trains on random 10-person (or 10-node) worlds
 * and is tested on 50-node worlds.
 *
 * <p>Tasks, each a binary target over binary background relations:
 * <ul>
 *   <li>{@link Task#GRANDPARENT}: from {@code parent} in a random family tree;</li>
 *   <li>{@link Task#UNCLE}: from {@code parent} and {@code sibling} (uncle or aunt:
 *       a sibling of a parent);</li>
 *   <li>{@link Task#REACHABILITY}: from {@code edge} in a random sparse directed graph,
 *       a recursive rule.</li>
 * </ul>
 * The family-tree generator is local; wave 7 WP1's knowledge-graph item shares the idea.
 */
public final class RuleLearning {

    /** The rule-learning tasks. */
    public enum Task {
        /** {@code grandparent(X,Y) :- parent(X,Z), parent(Z,Y)} */
        GRANDPARENT,
        /** {@code uncle(X,Y) :- sibling(X,Z), parent(Z,Y)} */
        UNCLE,
        /** {@code reach(X,Y) :- edge(X,Y); reach(X,Y) :- edge(X,Z), reach(Z,Y)} */
        REACHABILITY
    }

    /** The learners. */
    public enum Learner {
        /** Differentiable ILP with two clause slots. */
        DILP,
        /** A Neural Logic Machine of depth 4. */
        NLM,
        /** GIN node embeddings with a pair classifier. */
        GNN
    }

    /**
     * Scores.
     *
     * @param trainAccuracy pair accuracy on held-out small worlds
     * @param testAccuracy pair accuracy on 50-node worlds
     * @param positiveRecall recall of the true target pairs on 50-node worlds
     * @param program the extracted rules (dILP), or empty
     */
    public record Scores(double trainAccuracy, double testAccuracy, double positiveRecall, String program) {
    }

    /**
     * A world.
     *
     * @param relations the background relations
     * @param target the target relation
     */
    public record World(List<Tensor> relations, Tensor target) {
    }

    private RuleLearning() {
    }

    /**
     * Draws a world.
     *
     * @param task the task
     * @param n people or nodes
     * @param r the random source
     * @return the world
     */
    public static World world(Task task, int n, SplittableRandom r) {
        boolean[][] base = new boolean[n][n];
        boolean[][] sibling = new boolean[n][n];
        if (task == Task.REACHABILITY) {
            for (int i = 0; i < n; i++) {
                int out = r.nextInt(3) == 0 ? 0 : 1;
                for (int e = 0; e < out; e++) {
                    int j = r.nextInt(n);
                    if (j != i) {
                        base[i][j] = true;
                    }
                }
            }
        } else {
            // a family forest: the first few people are founders, everyone else gets one or two parents
            int founders = Math.max(2, n / 5);
            for (int c = founders; c < n; c++) {
                int p1 = r.nextInt(c);
                base[p1][c] = true;
                if (r.nextBoolean()) {
                    int p2 = r.nextInt(c);
                    base[p2][c] = true;
                }
            }
            for (int a = 0; a < n; a++) {
                for (int b = 0; b < n; b++) {
                    for (int p = 0; p < n && a != b && !sibling[a][b]; p++) {
                        sibling[a][b] = base[p][a] && base[p][b];
                    }
                }
            }
        }
        boolean[][] target = new boolean[n][n];
        switch (task) {
            case GRANDPARENT -> compose(base, base, target);
            case UNCLE -> compose(sibling, base, target);
            default -> {
                for (int i = 0; i < n; i++) {
                    System.arraycopy(base[i], 0, target[i], 0, n);
                }
                for (int k = 0; k < n; k++) {   // Warshall
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n; j++) {
                            target[i][j] |= target[i][k] && target[k][j];
                        }
                    }
                }
            }
        }
        List<Tensor> rels = new ArrayList<>(List.of(tensor(base)));
        if (task == Task.UNCLE) {
            rels.add(tensor(sibling));
        }
        return new World(rels, tensor(target));
    }

    private static void compose(boolean[][] p, boolean[][] q, boolean[][] into) {
        int n = p.length;
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                for (int z = 0; z < n && !into[x][y]; z++) {
                    into[x][y] = p[x][z] && q[z][y];
                }
            }
        }
    }

    private static Tensor tensor(boolean[][] b) {
        int n = b.length;
        float[] v = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                v[i * n + j] = b[i][j] ? 1 : 0;
            }
        }
        return Tensor.of(Shape.of(n, n), v);
    }

    /** A GIN pair classifier: two GIN layers on each edge direction, then an MLP on (h_x, h_y). */
    private static final class GnnPairs {
        final List<GinLayer> forward = new ArrayList<>();
        final List<GinLayer> backward = new ArrayList<>();
        final Var w1;
        final Var w2;

        GnnPairs(int relations, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            int in = 1 + 2 * relations;
            for (int l = 0; l < 2; l++) {
                forward.add(new GinLayer(l == 0 ? in : 32, 32, 16, r));
                backward.add(new GinLayer(l == 0 ? in : 32, 32, 16, r));
            }
            w1 = Var.param(Tensor.normal(Shape.of(64, 32), 0, 0.15f, r));
            w2 = Var.param(Tensor.normal(Shape.of(32, 1), 0, 0.15f, r));
        }

        Var logits(List<Tensor> relations) {
            int n = relations.get(0).shape().dim(0);
            List<Var> f = new ArrayList<>(List.of(Var.of(Tensor.full(Shape.of(n, 1), 1f))));
            Tensor union = relations.get(0);
            for (Tensor rel : relations) {
                f.add(Var.of(rel.sum(1, true)));
                f.add(Var.of(rel.sum(0, false).reshape(n, 1)));
                union = union.add(rel);
            }
            Var h = Var.concat(1, f.toArray(Var[]::new));
            Var a = Var.of(union);
            Var at = Var.of(union.transpose());
            for (int l = 0; l < 2; l++) {
                h = Var.concat(1, forward.get(l).forward(List.of(h, a), true).relu(),
                        backward.get(l).forward(List.of(h, at), true).relu());
            }
            Var hx = h.reshape(n, 1, 32).expand(Shape.of(n, n, 32));
            Var hy = h.reshape(1, n, 32).expand(Shape.of(n, n, 32));
            return Var.concat(2, hx, hy).matmul(w1).relu().matmul(w2).reshape(n, n);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            forward.forEach(g -> p.addAll(g.parameters()));
            backward.forEach(g -> p.addAll(g.parameters()));
            p.add(w1);
            p.add(w2);
            return p;
        }
    }

    /**
     * Trains a learner on 10-node worlds and tests it on 50-node worlds.
     *
     * @param task the task
     * @param learner the learner
     * @param steps training steps (one world each)
     * @param seed the seed
     * @return the scores
     */
    public static Scores run(Task task, Learner learner, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int background = task == Task.UNCLE ? 2 : 1;
        DifferentiableIlp dilp = new DifferentiableIlp(background, task == Task.REACHABILITY ? 6 : 2, seed);
        NeuralLogicMachine nlm = new NeuralLogicMachine(background, 8, 4, seed);
        GnnPairs gnn = new GnnPairs(background, seed);
        List<Var> params = switch (learner) {
            case DILP -> dilp.parameters();
            case NLM -> nlm.parameters();
            case GNN -> gnn.parameters();
        };
        Optimizer opt = Optimizer.adam(learner == Learner.DILP ? 0.1f : 0.01f);
        for (int s = 0; s < steps; s++) {
            World w = world(task, 10, r);
            Var loss = switch (learner) {
                case DILP -> dilp.loss(w.relations(), w.target());
                case NLM -> nlm.loss(w.relations(), w.target());
                case GNN -> {
                    Var z = gnn.logits(w.relations());
                    yield z.softplus().sub(z.mul(w.target())).mean();
                }
            };
            loss.backward();
            opt.step(params);
        }
        double[] small = score(task, learner, dilp, nlm, gnn, 10, new SplittableRandom(seed + 1));
        double[] large = score(task, learner, dilp, nlm, gnn, 50, new SplittableRandom(seed + 2));
        String program = learner == Learner.DILP ? String.join("; ", dilp.program(
                task == Task.REACHABILITY ? new String[] {"edge"} : task == Task.UNCLE
                        ? new String[] {"parent", "sibling"} : new String[] {"parent"})) : "";
        return new Scores(small[0], large[0], large[1], program);
    }

    private static double[] score(Task task, Learner learner, DifferentiableIlp dilp, NeuralLogicMachine nlm,
            GnnPairs gnn, int n, SplittableRandom r) {
        int right = 0;
        int total = 0;
        int pos = 0;
        int found = 0;
        for (int k = 0; k < 4; k++) {
            World w = world(task, n, r);
            float[] y = w.target().toArray();
            boolean[] pred = new boolean[n * n];
            switch (learner) {
                case DILP -> {
                    boolean[][] t = dilp.runProgram(w.relations());
                    for (int i = 0; i < n * n; i++) {
                        pred[i] = t[i / n][i % n];
                    }
                }
                case NLM -> {
                    float[] z = Var.noGrad(() -> nlm.forward(w.relations())).value().toArray();
                    for (int i = 0; i < n * n; i++) {
                        pred[i] = z[i] > 0;
                    }
                }
                case GNN -> {
                    float[] z = Var.noGrad(() -> gnn.logits(w.relations())).value().toArray();
                    for (int i = 0; i < n * n; i++) {
                        pred[i] = z[i] > 0;
                    }
                }
            }
            for (int i = 0; i < n * n; i++) {
                boolean t = y[i] > 0.5f;
                right += pred[i] == t ? 1 : 0;
                total++;
                pos += t ? 1 : 0;
                found += t && pred[i] ? 1 : 0;
            }
        }
        return new double[] {right / (double) total, pos == 0 ? 1 : found / (double) pos};
    }
}
