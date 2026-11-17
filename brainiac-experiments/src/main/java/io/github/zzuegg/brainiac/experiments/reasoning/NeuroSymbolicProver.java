package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * An AlphaGeometry-style neuro-symbolic prover (Trinh et al., "Solving olympiad geometry
 * without human demonstrations", <a href="https://doi.org/10.1038/s41586-023-06747-5">Nature
 * 2024</a>) on toy algebra (item 5.9). Terms are sums of variables ({@code a+b+c}, written
 * {@code abc}); premises are equalities between terms; the goal is an equality.
 * <ul>
 *   <li>The <b>symbolic engine</b> (the deductive database) closes the equalities over a
 *       fixed universe of terms, the ones the problem mentions: union-find with the
 *       congruence rule "if {@code x = y} and {@code x} is part of {@code T}, then
 *       {@code T = T[x := y]}", applied when {@code T[x := y]} is in the universe. Like
 *       AlphaGeometry's engine it is sound and fast but never invents a new object, so a
 *       proof that must pass through a term the problem never mentions is out of its reach.</li>
 *   <li>The <b>neural proposer</b> suggests such an auxiliary term (AlphaGeometry's
 *       auxiliary construction) from the candidates one rewrite away from what is already
 *       known equal to the goal's left side. It is a small network (reading the variable
 *       counts of the candidate and of both goal sides, and the candidate's count distance to
 *       each side) trained only on synthetic problems whose generating rewrite chains say
 *       which auxiliary terms were needed, as AlphaGeometry learns from synthetic proofs.</li>
 * </ul>
 * The loop alternates: deduce; if the goal is not proved, add the proposer's best auxiliary
 * term; repeat within a budget. Baselines: the symbolic engine alone, and the same loop with
 * auxiliary terms picked at random from the same candidates.
 */
public final class NeuroSymbolicProver {

    static final int VARS = 8;

    /**
     * A problem.
     *
     * @param premises equalities, each {@code {left, right}} as sorted variable strings
     * @param goal the equality to prove
     * @param path the generating rewrite chain from the goal's left to its right side
     */
    public record Problem(List<String[]> premises, String[] goal, List<String> path) {
    }

    /**
     * Proof rates.
     *
     * @param symbolic proved by the engine alone
     * @param neural proved with the trained proposer
     * @param random proved with random auxiliary terms
     * @param neuralAux mean auxiliary terms added when the neural loop proved a goal
     * @param needAux the share of test problems the engine alone cannot prove
     */
    public record Result(double symbolic, double neural, double random, double neuralAux, double needAux) {
    }

    private NeuroSymbolicProver() {
    }

    static String sorted(String s) {
        char[] c = s.toCharArray();
        Arrays.sort(c);
        return new String(c);
    }

    /** {@code t} without the multiset {@code x}, or null if {@code x} is not part of it. */
    static String minus(String t, String x) {
        StringBuilder b = new StringBuilder(t);
        for (char ch : x.toCharArray()) {
            int at = b.indexOf(String.valueOf(ch));
            if (at < 0) {
                return null;
            }
            b.deleteCharAt(at);
        }
        return b.toString();
    }

    /** The symbolic engine: union-find over a universe of terms, closed under congruence. */
    static final class Engine {
        final List<String> universe = new ArrayList<>();
        final Map<String, String> parent = new HashMap<>();
        final List<String[]> premises;

        Engine(Problem p) {
            this.premises = p.premises();
            for (String[] e : p.premises()) {
                term(e[0]);
                term(e[1]);
            }
            term(p.goal()[0]);
            term(p.goal()[1]);
        }

        void term(String t) {
            if (!parent.containsKey(t)) {
                parent.put(t, t);
                universe.add(t);
            }
        }

        String find(String t) {
            String r = t;
            while (!parent.get(r).equals(r)) {
                r = parent.get(r);
            }
            return r;
        }

        boolean union(String a, String b) {
            String ra = find(a);
            String rb = find(b);
            if (ra.equals(rb)) {
                return false;
            }
            parent.put(ra, rb);
            return true;
        }

        /** Every equal pair {@code (x, y)} of terms, both orders. */
        List<String[]> equalities() {
            List<String[]> out = new ArrayList<>();
            for (String x : universe) {
                for (String y : universe) {
                    if (!x.equals(y) && find(x).equals(find(y))) {
                        out.add(new String[] {x, y});
                    }
                }
            }
            return out;
        }

        void deduce() {
            for (String[] e : premises) {
                union(e[0], e[1]);
            }
            boolean changed = true;
            while (changed) {
                changed = false;
                List<String[]> eq = equalities();
                for (String t : List.copyOf(universe)) {
                    for (String[] xy : eq) {
                        String rest = minus(t, xy[0]);
                        if (rest != null) {
                            String t2 = sorted(rest + xy[1]);
                            if (parent.containsKey(t2)) {
                                changed |= union(t, t2);
                            }
                        }
                    }
                }
            }
        }

        boolean proves(String[] goal) {
            return find(goal[0]).equals(find(goal[1]));
        }

        /** Terms one rewrite away from the class of {@code from} that the universe lacks. */
        List<String> candidates(String from) {
            Set<String> out = new LinkedHashSet<>();
            List<String[]> eq = equalities();
            for (String t : universe) {
                if (!find(t).equals(find(from))) {
                    continue;
                }
                for (String[] xy : eq) {
                    String rest = minus(t, xy[0]);
                    if (rest != null) {
                        String t2 = sorted(rest + xy[1]);
                        if (!parent.containsKey(t2) && t2.length() <= 6) {
                            out.add(t2);
                        }
                    }
                }
            }
            return new ArrayList<>(out);
        }
    }

    static String randomTerm(int length, SplittableRandom r) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < length; i++) {
            b.append((char) ('a' + r.nextInt(VARS)));
        }
        return sorted(b.toString());
    }

    /**
     * Generates a problem whose proof rewrites the goal's left side {@code depth} times.
     *
     * @param depth the rewrites on the generating chain
     * @param distractors extra premises
     * @param r the random source
     * @return the problem
     */
    public static Problem generate(int depth, int distractors, SplittableRandom r) {
        while (true) {
            List<String[]> premises = new ArrayList<>();
            String t = randomTerm(3, r);
            List<String> path = new ArrayList<>(List.of(t));
            for (int d = 0; d < depth; d++) {
                char x = t.charAt(r.nextInt(t.length()));
                String y = randomTerm(1 + r.nextInt(2), r);
                if (y.equals(String.valueOf(x))) {
                    break;
                }
                premises.add(new String[] {String.valueOf(x), y});
                t = sorted(minus(t, String.valueOf(x)) + y);
                if (path.contains(t)) {
                    break;
                }
                path.add(t);
            }
            if (path.size() != depth + 1) {
                continue;
            }
            for (int i = 0; i < distractors; i++) {
                premises.add(new String[] {randomTerm(1, r), randomTerm(1 + r.nextInt(2), r)});
            }
            java.util.Collections.shuffle(premises, new java.util.Random(r.nextLong()));
            return new Problem(premises, new String[] {path.getFirst(), path.getLast()}, path);
        }
    }

    static final int FEATURES = 3 * VARS + 3;

    static float[] features(String candidate, String[] goal) {
        float[] f = new float[FEATURES];
        for (char c : candidate.toCharArray()) {
            f[c - 'a'] += 1;
        }
        for (char c : goal[1].toCharArray()) {
            f[VARS + c - 'a'] += 1;
        }
        for (char c : goal[0].toCharArray()) {
            f[2 * VARS + c - 'a'] += 1;
        }
        f[3 * VARS] = candidate.length() / 4f;
        float toRight = 0;
        float toLeft = 0;
        for (int v = 0; v < VARS; v++) {
            toRight += Math.abs(f[v] - f[VARS + v]);
            toLeft += Math.abs(f[v] - f[2 * VARS + v]);
        }
        f[3 * VARS + 1] = toRight / 4f;
        f[3 * VARS + 2] = toLeft / 4f;
        return f;
    }

    static Tensor rows(List<float[]> rows) {
        int w = rows.getFirst().length;
        float[] flat = new float[rows.size() * w];
        for (int i = 0; i < rows.size(); i++) {
            System.arraycopy(rows.get(i), 0, flat, i * w, w);
        }
        return Tensor.of(Shape.of(rows.size(), w), flat);
    }

    /**
     * Trains the proposer on synthetic problems: a candidate is positive when it lies on the
     * generating rewrite chain.
     *
     * @param problems how many synthetic problems
     * @param steps optimizer steps
     * @param seed the seed
     * @return the network, scoring a candidate's features
     */
    static Model trainProposer(int problems, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> xs = new ArrayList<>();
        List<Float> ys = new ArrayList<>();
        for (int i = 0; i < problems; i++) {
            Problem p = generate(2 + r.nextInt(3), 3, r);
            Engine e = new Engine(p);
            e.deduce();
            // walk the chain, as the proof would, labelling the candidates at each stage
            for (int s = 1; s < p.path().size() - 1 && !e.proves(p.goal()); s++) {
                for (String c : e.candidates(p.goal()[0])) {
                    xs.add(features(c, p.goal()));
                    ys.add(p.path().contains(c) ? 1f : 0f);
                }
                e.term(p.path().get(s));
                e.deduce();
            }
        }
        Model net = Model.sequential().input(FEATURES).dense(32, Activation.TANH).dense(1).seed(seed).build();
        Optimizer opt = Optimizer.adam(1e-2f);
        Tensor x = rows(xs);
        float[] y = new float[ys.size()];
        for (int i = 0; i < y.length; i++) {
            y[i] = ys.get(i);
        }
        Tensor yt = Tensor.of(Shape.of(y.length, 1), y);
        for (int s = 0; s < steps; s++) {
            Var z = net.forward(Var.of(x), true);
            Var loss = z.softplus().sub(z.mul(yt)).mean();
            loss.backward();
            opt.step(net.parameters());
        }
        return net;
    }

    /**
     * The proof loop.
     *
     * @param p the problem
     * @param budget the most auxiliary terms
     * @param net the proposer, or {@code null} to pick candidates at random
     * @param r the random source
     * @return the auxiliary terms used, or -1 if the goal was not proved
     */
    static int prove(Problem p, int budget, Model net, SplittableRandom r) {
        Engine e = new Engine(p);
        for (int aux = 0; ; aux++) {
            e.deduce();
            if (e.proves(p.goal())) {
                return aux;
            }
            if (aux == budget) {
                return -1;
            }
            List<String> cands = e.candidates(p.goal()[0]);
            if (cands.isEmpty()) {
                return -1;
            }
            String pick;
            if (net == null) {
                pick = cands.get(r.nextInt(cands.size()));
            } else {
                List<float[]> f = new ArrayList<>();
                for (String c : cands) {
                    f.add(features(c, p.goal()));
                }
                float[] s = net.forward(Var.of(rows(f)), false).value().to(DType.FLOAT32).toArray();
                int best = 0;
                for (int i = 1; i < s.length; i++) {
                    if (s[i] > s[best]) {
                        best = i;
                    }
                }
                pick = cands.get(best);
            }
            e.term(pick);
        }
    }

    /**
     * @param problems test problems
     * @param budget auxiliary terms allowed per problem
     * @param trainProblems synthetic training problems for the proposer
     * @param seed the seed
     * @return the proof rates
     */
    public static Result run(int problems, int budget, int trainProblems, long seed) {
        Model net = trainProposer(trainProblems, 300, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int sym = 0;
        int neural = 0;
        int random = 0;
        int neuralAux = 0;
        for (int i = 0; i < problems; i++) {
            Problem p = generate(2 + i % 3, 3, r);
            sym += prove(p, 0, null, r) >= 0 ? 1 : 0;
            int n = prove(p, budget, net, r);
            if (n >= 0) {
                neural++;
                neuralAux += n;
            }
            random += prove(p, budget, null, r) >= 0 ? 1 : 0;
        }
        double m = problems;
        return new Result(sym / m, neural / m, random / m, neural == 0 ? 0 : neuralAux / (double) neural,
                1 - sym / m);
    }
}
