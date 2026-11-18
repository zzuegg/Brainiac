package io.github.zzuegg.brainiac.experimental.logic;

import io.github.zzuegg.brainiac.experimental.data.FamilyTreeGraph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * A saturation-based first-order resolution prover (Robinson, JACM 1965) with the Otter
 * given-clause loop (McCune): the negated conjecture and the axioms go to the passive set; each
 * iteration selects a given clause, discards it if an active clause subsumes it, removes the active
 * clauses it subsumes (backward subsumption), makes it active and adds to the passive set its
 * factors and its binary resolvents with every active clause (most general unifiers with occurs
 * check). New clauses are simplified (duplicate literals), tautologies, duplicates, clauses over
 * the weight, length or term-depth limit and clauses subsumed by an active clause are dropped. The
 * empty clause ends the search with a proof, which is read off the parent links.
 *
 * <p>Clause selection, the decisive heuristic, is one of {@link Selection}: oldest first (fair,
 * complete), lightest first (symbol count), the classic 1:4 interleaving of both, or a learned
 * priority ({@link ClauseSelector#priority}, ENIGMA by Jakubův &amp; Urban, CICM 2017) interleaved 1:4 with
 * age, as E does with ENIGMA models.
 *
 * <p>Clauses are written as strings ({@code "~parent(X,Y) | ~parent(Y,Z) | grandparent(X,Z)"};
 * identifiers that start with an upper-case letter are variables). Equality has no built-in
 * treatment: problems that need it state its axioms. The problem generators (family relations
 * over {@link FamilyTreeGraph}, transitive chains, equivalence-relation lemmas, pigeonhole, and
 * group theory in the relational formulation) produce the benchmark families of 13.2.12.
 */
public final class ResolutionProver {

    private ResolutionProver() {
    }

    /** Clause selection strategies. */
    public enum Selection {
        /** Oldest first (breadth-first, fair). */
        AGE,
        /** Fewest symbols first. */
        WEIGHT,
        /** One oldest clause for every four lightest. */
        AGE_WEIGHT,
        /** One oldest clause for every four best-scored by a {@link ClauseSelector}. */
        LEARNED
    }

    /**
     * A term: a variable ({@code symbol < 0}, index {@code -symbol - 1}) or a function or constant
     * applied to arguments.
     *
     * @param symbol the symbol index, negative for variables
     * @param args the arguments (empty for constants and variables)
     */
    public record Term(int symbol, Term[] args) {

        boolean isVar() {
            return symbol < 0;
        }

        int var() {
            return -symbol - 1;
        }

        static Term var(int v) {
            return new Term(-v - 1, NO_ARGS);
        }

        int depth() {
            int d = 0;
            for (Term a : args) {
                d = Math.max(d, a.depth());
            }
            return d + (args.length > 0 ? 1 : 0);
        }

        int weight() {
            int w = 1;
            for (Term a : args) {
                w += a.weight();
            }
            return w;
        }

        boolean same(Term o) {
            if (symbol != o.symbol || args.length != o.args.length) {
                return false;
            }
            for (int i = 0; i < args.length; i++) {
                if (!args[i].same(o.args[i])) {
                    return false;
                }
            }
            return true;
        }
    }

    private static final Term[] NO_ARGS = new Term[0];

    /**
     * A literal.
     *
     * @param positive its sign
     * @param predicate the predicate symbol
     * @param args the arguments
     */
    public record Literal(boolean positive, int predicate, Term[] args) {
    }

    /** A clause with its derivation. */
    public static final class Clause {
        private final Literal[] literals;
        private final int id;
        private final int[] parents;
        private final boolean goal;
        private final int weight;
        private final int variables;
        private double score;
        private boolean dead;

        Clause(Literal[] literals, int id, int[] parents, boolean goal) {
            this.literals = literals;
            this.id = id;
            this.parents = parents;
            this.goal = goal;
            int w = 0;
            int maxVar = -1;
            for (Literal l : literals) {
                w += 1;
                for (Term t : l.args()) {
                    w += t.weight();
                    maxVar = Math.max(maxVar, maxVar(t));
                }
            }
            this.weight = w;
            this.variables = maxVar + 1;
        }

        /** @return the literals */
        public List<Literal> literals() {
            return List.of(literals);
        }

        /** @return the clause number (its age) */
        public int id() {
            return id;
        }

        /** @return the parents' numbers (empty for input clauses) */
        public int[] parents() {
            return parents.clone();
        }

        /** @return whether the clause is or descends from the negated conjecture */
        public boolean goal() {
            return goal;
        }

        /** @return the symbol count */
        public int weight() {
            return weight;
        }

        /** @return whether this is the empty clause */
        public boolean empty() {
            return literals.length == 0;
        }
    }

    private static int maxVar(Term t) {
        if (t.isVar()) {
            return t.var();
        }
        int m = -1;
        for (Term a : t.args()) {
            m = Math.max(m, maxVar(a));
        }
        return m;
    }

    /**
     * A problem: axioms plus the clauses of the negated conjecture.
     *
     * @param name the name
     * @param symbols symbol names by index
     * @param arities symbol arities by index
     * @param clauses the input clauses (goal clauses flagged)
     */
    public record Problem(String name, List<String> symbols, List<Integer> arities, List<Clause> clauses) {

        /**
         * Starts a problem.
         *
         * @param name the name
         * @return a builder
         */
        public static Builder builder(String name) {
            return new Builder(name);
        }

        /** @return the symbols occurring in the negated conjecture */
        public BitSet conjectureSymbols() {
            BitSet b = new BitSet();
            for (Clause c : clauses) {
                if (c.goal) {
                    for (Literal l : c.literals) {
                        b.set(l.predicate());
                        for (Term t : l.args()) {
                            symbolsOf(t, b);
                        }
                    }
                }
            }
            return b;
        }

        private static void symbolsOf(Term t, BitSet b) {
            if (!t.isVar()) {
                b.set(t.symbol());
                for (Term a : t.args()) {
                    symbolsOf(a, b);
                }
            }
        }

        /**
         * Renders a clause.
         *
         * @param c the clause
         * @return its text
         */
        public String show(Clause c) {
            if (c.literals.length == 0) {
                return "$false";
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < c.literals.length; i++) {
                Literal l = c.literals[i];
                sb.append(i > 0 ? " | " : "").append(l.positive() ? "" : "~").append(symbols.get(l.predicate()));
                appendArgs(sb, l.args());
            }
            return sb.toString();
        }

        private void appendArgs(StringBuilder sb, Term[] args) {
            if (args.length == 0) {
                return;
            }
            sb.append('(');
            for (int i = 0; i < args.length; i++) {
                sb.append(i > 0 ? "," : "");
                Term t = args[i];
                if (t.isVar()) {
                    sb.append('X').append(t.var());
                } else {
                    sb.append(symbols.get(t.symbol()));
                    appendArgs(sb, t.args());
                }
            }
            sb.append(')');
        }
    }

    /** Builds a {@link Problem} from clause strings. */
    public static final class Builder {
        private final String name;
        private final List<String> symbols = new ArrayList<>();
        private final List<Integer> arities = new ArrayList<>();
        private final Map<String, Integer> index = new HashMap<>();
        private final List<Clause> clauses = new ArrayList<>();

        Builder(String name) {
            this.name = name;
        }

        /**
         * Adds an axiom.
         *
         * @param clause the clause text
         * @return this builder
         */
        public Builder axiom(String clause) {
            clauses.add(new Clause(normalize(parse(clause)), clauses.size(), new int[0], false));
            return this;
        }

        /**
         * Adds a clause of the negated conjecture.
         *
         * @param clause the clause text
         * @return this builder
         */
        public Builder goal(String clause) {
            clauses.add(new Clause(normalize(parse(clause)), clauses.size(), new int[0], true));
            return this;
        }

        /**
         * Builds the problem.
         *
         * @param nameSuffix appended to the name
         * @return the problem
         */
        public Problem build(String... nameSuffix) {
            return new Problem(name + String.join("", nameSuffix), List.copyOf(symbols), List.copyOf(arities),
                List.copyOf(clauses));
        }

        private int symbol(String s, int arity) {
            Integer i = index.get(s + "/" + arity);
            if (i == null) {
                i = symbols.size();
                symbols.add(s);
                arities.add(arity);
                index.put(s + "/" + arity, i);
            }
            return i;
        }

        private Literal[] parse(String text) {
            String[] parts = text.split("\\|");
            Literal[] out = new Literal[parts.length];
            Map<String, Integer> vars = new HashMap<>();
            for (int i = 0; i < parts.length; i++) {
                String p = parts[i].trim();
                boolean pos = !p.startsWith("~");
                if (!pos) {
                    p = p.substring(1).trim();
                }
                int[] at = {0};
                Term t = term(p, at, vars);
                out[i] = new Literal(pos, symbolOfPredicate(t), t.args());
            }
            return out;
        }

        private int symbolOfPredicate(Term t) {
            return t.symbol();
        }

        private Term term(String s, int[] at, Map<String, Integer> vars) {
            int start = at[0];
            while (at[0] < s.length() && (Character.isLetterOrDigit(s.charAt(at[0])) || s.charAt(at[0]) == '_')) {
                at[0]++;
            }
            String id = s.substring(start, at[0]);
            if (id.isEmpty()) {
                throw new IllegalArgumentException("bad term at " + start + " in " + s);
            }
            List<Term> args = new ArrayList<>();
            if (at[0] < s.length() && s.charAt(at[0]) == '(') {
                at[0]++;
                while (true) {
                    args.add(term(s, at, vars));
                    char c = s.charAt(at[0]++);
                    if (c == ')') {
                        break;
                    }
                    if (c != ',') {
                        throw new IllegalArgumentException("expected , or ) in " + s);
                    }
                }
            }
            if (Character.isUpperCase(id.charAt(0))) {
                return Term.var(vars.computeIfAbsent(id, k -> vars.size()));
            }
            return new Term(symbol(id, args.size()), args.toArray(NO_ARGS));
        }
    }

    // ---- unification and matching -----------------------------------------------------------

    private static Term deref(Term t, Term[] b) {
        while (t.isVar() && b[t.var()] != null) {
            t = b[t.var()];
        }
        return t;
    }

    private static boolean occurs(int v, Term t, Term[] b) {
        t = deref(t, b);
        if (t.isVar()) {
            return t.var() == v;
        }
        for (Term a : t.args()) {
            if (occurs(v, a, b)) {
                return true;
            }
        }
        return false;
    }

    private static boolean unify(Term s, Term t, Term[] b) {
        s = deref(s, b);
        t = deref(t, b);
        if (s.isVar() && t.isVar() && s.var() == t.var()) {
            return true;
        }
        if (s.isVar()) {
            if (occurs(s.var(), t, b)) {
                return false;
            }
            b[s.var()] = t;
            return true;
        }
        if (t.isVar()) {
            if (occurs(t.var(), s, b)) {
                return false;
            }
            b[t.var()] = s;
            return true;
        }
        if (s.symbol() != t.symbol() || s.args().length != t.args().length) {
            return false;
        }
        for (int i = 0; i < s.args().length; i++) {
            if (!unify(s.args()[i], t.args()[i], b)) {
                return false;
            }
        }
        return true;
    }

    private static boolean unifyArgs(Term[] x, Term[] y, Term[] b) {
        for (int i = 0; i < x.length; i++) {
            if (!unify(x[i], y[i], b)) {
                return false;
            }
        }
        return true;
    }

    private static Term apply(Term t, Term[] b) {
        t = deref(t, b);
        if (t.isVar() || t.args().length == 0) {
            return t;
        }
        Term[] a = new Term[t.args().length];
        for (int i = 0; i < a.length; i++) {
            a[i] = apply(t.args()[i], b);
        }
        return new Term(t.symbol(), a);
    }

    private static Term shift(Term t, int offset) {
        if (t.isVar()) {
            return Term.var(t.var() + offset);
        }
        if (t.args().length == 0) {
            return t;
        }
        Term[] a = new Term[t.args().length];
        for (int i = 0; i < a.length; i++) {
            a[i] = shift(t.args()[i], offset);
        }
        return new Term(t.symbol(), a);
    }

    /** One-way matching: binds only pattern variables; target variables are rigid. */
    private static boolean match(Term p, Term t, Term[] b) {
        if (p.isVar()) {
            if (b[p.var()] == null) {
                b[p.var()] = t;
                return true;
            }
            return b[p.var()].same(t);
        }
        if (t.isVar() || p.symbol() != t.symbol() || p.args().length != t.args().length) {
            return false;
        }
        for (int i = 0; i < p.args().length; i++) {
            if (!match(p.args()[i], t.args()[i], b)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Whether {@code c} subsumes {@code d}: some substitution maps every literal of {@code c} onto
     * a literal of {@code d} (and {@code c} is no longer than {@code d}).
     *
     * @param c the general clause
     * @param d the specific clause
     * @return whether {@code c} subsumes {@code d}
     */
    public static boolean subsumes(Clause c, Clause d) {
        if (c.literals.length > d.literals.length) {
            return false;
        }
        return subsumes(c.literals, 0, d.literals, new Term[Math.max(1, c.variables)]);
    }

    private static boolean subsumes(Literal[] c, int i, Literal[] d, Term[] b) {
        if (i == c.length) {
            return true;
        }
        Literal l = c[i];
        for (Literal m : d) {
            if (m.positive() != l.positive() || m.predicate() != l.predicate()) {
                continue;
            }
            Term[] nb = b.clone();
            boolean ok = true;
            for (int k = 0; k < l.args().length && ok; k++) {
                ok = match(l.args()[k], m.args()[k], nb);
            }
            if (ok && subsumes(c, i + 1, d, nb)) {
                return true;
            }
        }
        return false;
    }

    // ---- normalisation ------------------------------------------------------------------------

    private static Term rename(Term t, Map<Integer, Integer> map) {
        if (t.isVar()) {
            return Term.var(map.computeIfAbsent(t.var(), k -> map.size()));
        }
        if (t.args().length == 0) {
            return t;
        }
        Term[] a = new Term[t.args().length];
        for (int i = 0; i < a.length; i++) {
            a[i] = rename(t.args()[i], map);
        }
        return new Term(t.symbol(), a);
    }

    private static String key(Term t) {
        if (t.isVar()) {
            return "_";
        }
        StringBuilder sb = new StringBuilder().append(t.symbol());
        if (t.args().length > 0) {
            sb.append('(');
            for (Term a : t.args()) {
                sb.append(key(a)).append(',');
            }
            sb.append(')');
        }
        return sb.toString();
    }

    private static String key(Literal l) {
        StringBuilder sb = new StringBuilder().append(l.positive() ? '+' : '-').append(l.predicate()).append('(');
        for (Term t : l.args()) {
            sb.append(key(t)).append(',');
        }
        return sb.append(')').toString();
    }

    /** Removes duplicate literals, sorts them by a variable-blind key and renames variables. */
    static Literal[] normalize(Literal[] lits) {
        List<Literal> sorted = new ArrayList<>(Arrays.asList(lits));
        sorted.sort(Comparator.comparing(ResolutionProver::key));
        Map<Integer, Integer> map = new HashMap<>();
        List<Literal> out = new ArrayList<>();
        for (Literal l : sorted) {
            Term[] a = new Term[l.args().length];
            for (int i = 0; i < a.length; i++) {
                a[i] = rename(l.args()[i], map);
            }
            Literal n = new Literal(l.positive(), l.predicate(), a);
            boolean dup = false;
            for (Literal o : out) {
                if (sameLiteral(o, n)) {
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                out.add(n);
            }
        }
        return out.toArray(new Literal[0]);
    }

    private static boolean sameLiteral(Literal a, Literal b) {
        if (a.positive() != b.positive() || a.predicate() != b.predicate()) {
            return false;
        }
        for (int i = 0; i < a.args().length; i++) {
            if (!a.args()[i].same(b.args()[i])) {
                return false;
            }
        }
        return true;
    }

    private static boolean tautology(Literal[] lits) {
        for (int i = 0; i < lits.length; i++) {
            for (int j = i + 1; j < lits.length; j++) {
                if (lits[i].positive() != lits[j].positive() && lits[i].predicate() == lits[j].predicate()) {
                    boolean same = true;
                    for (int k = 0; k < lits[i].args().length && same; k++) {
                        same = lits[i].args()[k].same(lits[j].args()[k]);
                    }
                    if (same) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private static String clauseKey(Literal[] lits) {
        StringBuilder sb = new StringBuilder();
        for (Literal l : lits) {
            sb.append(l.positive() ? '+' : '-').append(l.predicate()).append('(');
            for (Term t : l.args()) {
                exactKey(t, sb);
                sb.append(',');
            }
            sb.append(')');
        }
        return sb.toString();
    }

    private static void exactKey(Term t, StringBuilder sb) {
        if (t.isVar()) {
            sb.append('V').append(t.var());
            return;
        }
        sb.append(t.symbol());
        if (t.args().length > 0) {
            sb.append('(');
            for (Term a : t.args()) {
                exactKey(a, sb);
                sb.append(',');
            }
            sb.append(')');
        }
    }

    // ---- the given-clause loop --------------------------------------------------------------

    /**
     * Search limits.
     *
     * @param maxGiven the given-clause budget
     * @param maxWeight clauses heavier than this are dropped
     * @param maxLiterals clauses longer than this are dropped
     * @param maxDepth clauses with deeper terms are dropped
     * @param maxClauses the search gives up once this many clauses are kept
     */
    public record Limits(int maxGiven, int maxWeight, int maxLiterals, int maxDepth, int maxClauses) {

        /**
         * Default limits with a given budget.
         *
         * @param maxGiven the given-clause budget
         * @return limits with weight 30, 8 literals, term depth 3 and 40 kept clauses per given clause
         */
        public static Limits of(int maxGiven) {
            return new Limits(maxGiven, 30, 8, 3, 40 * maxGiven);
        }
    }

    /**
     * A proof attempt.
     *
     * @param proved whether the empty clause was derived
     * @param saturated whether the passive set ran empty without a proof (the clauses are
     *     satisfiable, up to the limits)
     * @param given given clauses processed
     * @param generated clauses generated
     * @param millis wall time
     * @param proof the clauses of the proof (empty if none)
     * @param givenClauses every clause processed as given, in order
     */
    public record Result(boolean proved, boolean saturated, int given, int generated, double millis,
            List<Clause> proof, List<Clause> givenClauses) {
    }

    /**
     * Runs the given-clause loop.
     *
     * @param problem the problem
     * @param selection the clause selection
     * @param limits the limits
     * @param selector the learned scorer for {@link Selection#LEARNED}, else ignored (may be null)
     * @return the result
     */
    public static Result prove(Problem problem, Selection selection, Limits limits, ClauseSelector selector) {
        long t0 = System.nanoTime();
        List<Clause> all = new ArrayList<>();
        PriorityQueue<Clause> byAge = new PriorityQueue<>(Comparator.comparingInt(Clause::id));
        PriorityQueue<Clause> byWeight = new PriorityQueue<>(
            Comparator.comparingInt(Clause::weight).thenComparingInt(Clause::id));
        PriorityQueue<Clause> byScore = new PriorityQueue<>(
            Comparator.comparingDouble((Clause c) -> c.score).thenComparingInt(Clause::weight)
                .thenComparingInt(Clause::id));
        Set<String> seen = new HashSet<>();
        Set<Clause> selected = new HashSet<>();
        List<Clause> active = new ArrayList<>();
        Map<Integer, List<Clause>> activeByPred = new HashMap<>();
        boolean learned = selection == Selection.LEARNED && selector != null;
        Clause empty = null;
        for (Clause c : problem.clauses()) {
            Clause copy = new Clause(c.literals, all.size(), new int[0], c.goal);
            all.add(copy);
            if (seen.add(clauseKey(copy.literals))) {
                if (learned) {
                    copy.score = selector.priority(problem, copy);
                }
                byAge.add(copy);
                byWeight.add(copy);
                byScore.add(copy);
            }
        }
        List<Clause> givenList = new ArrayList<>();
        int generated = 0;
        boolean saturated = false;
        outer:
        while (givenList.size() < limits.maxGiven() && all.size() < limits.maxClauses()) {
            Clause g = null;
            boolean useAge = switch (selection) {
                case AGE -> true;
                case WEIGHT -> false;
                case AGE_WEIGHT, LEARNED -> givenList.size() % 5 == 0;
            };
            PriorityQueue<Clause> q = useAge ? byAge : selection == Selection.LEARNED && learned ? byScore : byWeight;
            while (!q.isEmpty()) {
                Clause c = q.poll();
                if (!selected.contains(c)) {
                    g = c;
                    break;
                }
            }
            if (g == null) {
                saturated = true;
                break;
            }
            selected.add(g);
            boolean subsumed = false;
            for (Clause a : active) {
                if (!a.dead && subsumes(a, g)) {
                    subsumed = true;
                    break;
                }
            }
            if (subsumed) {
                continue;
            }
            givenList.add(g);
            for (Clause a : active) {
                if (!a.dead && subsumes(g, a)) {
                    a.dead = true;
                }
            }
            active.add(g);
            for (Literal l : g.literals) {
                List<Clause> lst = activeByPred.computeIfAbsent(l.predicate(), k -> new ArrayList<>());
                if (lst.isEmpty() || lst.get(lst.size() - 1) != g) {
                    lst.add(g);
                }
            }
            List<Literal[]> fresh = new ArrayList<>();
            List<int[]> freshParents = new ArrayList<>();
            List<Boolean> freshGoal = new ArrayList<>();
            // factors
            for (int i = 0; i < g.literals.length; i++) {
                for (int j = i + 1; j < g.literals.length; j++) {
                    Literal a = g.literals[i];
                    Literal b = g.literals[j];
                    if (a.positive() != b.positive() || a.predicate() != b.predicate()) {
                        continue;
                    }
                    Term[] bind = new Term[Math.max(1, g.variables)];
                    if (unifyArgs(a.args(), b.args(), bind)) {
                        List<Literal> out = new ArrayList<>();
                        for (int k = 0; k < g.literals.length; k++) {
                            if (k != j) {
                                out.add(applyLiteral(g.literals[k], bind));
                            }
                        }
                        fresh.add(out.toArray(new Literal[0]));
                        freshParents.add(new int[] {g.id});
                        freshGoal.add(g.goal);
                    }
                }
            }
            // binary resolvents with active clauses that share a predicate with the given clause
            Set<Clause> partners = new LinkedHashSet<>();
            for (Literal l : g.literals) {
                for (Clause c : activeByPred.getOrDefault(l.predicate(), List.of())) {
                    if (!c.dead) {
                        partners.add(c);
                    }
                }
            }
            for (Clause c : partners) {
                int offset = g.variables;
                Literal[] cl = new Literal[c.literals.length];
                for (int k = 0; k < cl.length; k++) {
                    Term[] a = new Term[c.literals[k].args().length];
                    for (int m = 0; m < a.length; m++) {
                        a[m] = shift(c.literals[k].args()[m], offset);
                    }
                    cl[k] = new Literal(c.literals[k].positive(), c.literals[k].predicate(), a);
                }
                for (int i = 0; i < g.literals.length; i++) {
                    for (int j = 0; j < cl.length; j++) {
                        Literal a = g.literals[i];
                        Literal b = cl[j];
                        if (a.positive() == b.positive() || a.predicate() != b.predicate()) {
                            continue;
                        }
                        Term[] bind = new Term[Math.max(1, offset + c.variables)];
                        if (!unifyArgs(a.args(), b.args(), bind)) {
                            continue;
                        }
                        List<Literal> out = new ArrayList<>();
                        for (int k = 0; k < g.literals.length; k++) {
                            if (k != i) {
                                out.add(applyLiteral(g.literals[k], bind));
                            }
                        }
                        for (int k = 0; k < cl.length; k++) {
                            if (k != j) {
                                out.add(applyLiteral(cl[k], bind));
                            }
                        }
                        fresh.add(out.toArray(new Literal[0]));
                        freshParents.add(new int[] {g.id, c.id});
                        freshGoal.add(g.goal || c.goal);
                    }
                }
            }
            for (int f = 0; f < fresh.size(); f++) {
                generated++;
                Literal[] lits = normalize(fresh.get(f));
                if (lits.length == 0) {
                    empty = new Clause(lits, all.size(), freshParents.get(f), freshGoal.get(f));
                    all.add(empty);
                    break outer;
                }
                if (lits.length > limits.maxLiterals() || tautology(lits)) {
                    continue;
                }
                Clause c = new Clause(lits, all.size(), freshParents.get(f), freshGoal.get(f));
                if (c.weight > limits.maxWeight() || depth(lits) > limits.maxDepth()
                        || !seen.add(clauseKey(lits))) {
                    continue;
                }
                boolean sub = false;
                for (Clause a : active) {
                    if (!a.dead && a.literals.length <= lits.length && subsumes(a, c)) {
                        sub = true;
                        break;
                    }
                }
                if (sub) {
                    continue;
                }
                all.add(c);
                if (learned) {
                    c.score = selector.priority(problem, c);
                }
                byAge.add(c);
                byWeight.add(c);
                if (learned) {
                    byScore.add(c);
                }
            }
        }
        List<Clause> proof = new ArrayList<>();
        if (empty != null) {
            BitSet in = new BitSet();
            List<Integer> stack = new ArrayList<>(List.of(empty.id));
            while (!stack.isEmpty()) {
                int id = stack.remove(stack.size() - 1);
                if (in.get(id)) {
                    continue;
                }
                in.set(id);
                for (int p : all.get(id).parents) {
                    stack.add(p);
                }
            }
            for (int id = in.nextSetBit(0); id >= 0; id = in.nextSetBit(id + 1)) {
                proof.add(all.get(id));
            }
        }
        return new Result(empty != null, saturated && empty == null, givenList.size(), generated,
            (System.nanoTime() - t0) / 1e6, proof, givenList);
    }

    private static int depth(Literal[] lits) {
        int d = 0;
        for (Literal l : lits) {
            for (Term t : l.args()) {
                d = Math.max(d, t.depth());
            }
        }
        return d;
    }

    private static Literal applyLiteral(Literal l, Term[] b) {
        Term[] a = new Term[l.args().length];
        for (int i = 0; i < a.length; i++) {
            a[i] = apply(l.args()[i], b);
        }
        return new Literal(l.positive(), l.predicate(), a);
    }

    // ---- problem generators ---------------------------------------------------------------------

    /**
     * Family-relation theorems: the training triples of a {@link FamilyTreeGraph} as facts, rules
     * for grandparent, child/parent inversion, sibling symmetry and two distractor relations
     * (uncle, ancestor), and one held-out triple as the conjecture.
     *
     * @param data the family graph
     * @param maxProblems the most problems (one per held-out triple)
     * @return the problems
     */
    public static List<Problem> family(FamilyTreeGraph.Data data, int maxProblems) {
        List<Problem> out = new ArrayList<>();
        for (int[] goal : data.test()) {
            if (out.size() >= maxProblems) {
                break;
            }
            Builder b = Problem.builder("family");
            b.axiom("~parent(X,Y) | ~parent(Y,Z) | grandparent(X,Z)");
            b.axiom("~parent(X,Y) | child(Y,X)");
            b.axiom("~child(X,Y) | parent(Y,X)");
            b.axiom("~sibling(X,Y) | sibling(Y,X)");
            b.axiom("~sibling(X,P) | ~parent(P,Y) | uncle(X,Y)");
            b.axiom("~parent(X,Y) | ancestor(X,Y)");
            b.axiom("~ancestor(X,Y) | ~ancestor(Y,Z) | ancestor(X,Z)");
            for (int[] t : data.train()) {
                b.axiom(FamilyTreeGraph.RELATIONS.get(t[1]) + "(p" + t[0] + ",p" + t[2] + ")");
            }
            b.goal("~" + FamilyTreeGraph.RELATIONS.get(goal[1]) + "(p" + goal[0] + ",p" + goal[2] + ")");
            out.add(b.build("-" + FamilyTreeGraph.RELATIONS.get(goal[1])));
        }
        return out;
    }

    /**
     * A transitive chain {@code r(c0,c1), ..., r(c(n-1),cn)} with transitivity and random
     * distractor edges among other constants; the conjecture is {@code r(c0, cn)}.
     *
     * @param n the chain length
     * @param distractors distractor edges
     * @param seed the seed
     * @return the problem
     */
    public static Problem chain(int n, int distractors, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Builder b = Problem.builder("chain-" + n);
        b.axiom("~r(X,Y) | ~r(Y,Z) | r(X,Z)");
        for (int i = 0; i < n; i++) {
            b.axiom("r(c" + i + ",c" + (i + 1) + ")");
        }
        int pool = Math.max(3, distractors / 2);
        for (int i = 0; i < distractors; i++) {
            b.axiom("r(d" + r.nextInt(pool) + ",d" + r.nextInt(pool) + ")");
        }
        b.goal("~r(c0,c" + n + ")");
        return b.build();
    }

    /**
     * An equivalence-relation lemma: reflexivity, symmetry and transitivity of {@code e}, a chain
     * {@code e(c0,c1), ..., e(c(n-1),cn)} and the conjecture {@code e(cn, c0)} (needs symmetry and
     * transitivity), or for {@code n = 0} the lemma {@code e(X,Y) -> e(Y,Y)} with Skolem constants.
     *
     * @param n the chain length
     * @return the problem
     */
    public static Problem equivalence(int n) {
        Builder b = Problem.builder("equiv-" + n);
        b.axiom("e(X,X)");
        b.axiom("~e(X,Y) | e(Y,X)");
        b.axiom("~e(X,Y) | ~e(Y,Z) | e(X,Z)");
        b.axiom("~e(X,Y) | f(X) | ~f(Y)");
        if (n == 0) {
            b.goal("e(a,b)");
            b.goal("f(a)");
            b.goal("~f(b)");
            return b.build();
        }
        for (int i = 0; i < n; i++) {
            b.axiom("e(c" + i + ",c" + (i + 1) + ")");
        }
        b.goal("~e(c" + n + ",c0)");
        return b.build();
    }

    /**
     * The propositional pigeonhole formula PHP(n+1, n): unsatisfiable, exponential for resolution.
     *
     * @param holes the holes
     * @return the problem (every clause is a goal clause)
     */
    public static Problem pigeonhole(int holes) {
        Builder b = Problem.builder("php-" + holes);
        for (int p = 0; p <= holes; p++) {
            StringBuilder sb = new StringBuilder();
            for (int h = 0; h < holes; h++) {
                sb.append(h > 0 ? " | " : "").append("p").append(p).append('h').append(h);
            }
            b.goal(sb.toString());
        }
        for (int h = 0; h < holes; h++) {
            for (int p = 0; p <= holes; p++) {
                for (int q = p + 1; q <= holes; q++) {
                    b.axiom("~p" + p + "h" + h + " | ~p" + q + "h" + h);
                }
            }
        }
        return b.build();
    }

    /** The group-theory theorems available to {@link #group}. */
    public static final List<String> GROUP_THEOREMS = List.of("right-identity", "right-inverse",
        "right-inverse-of-product", "exponent-2-commutative", "absorb-inverse-pair", "double-inverse");

    /**
     * A group-theory theorem in the relational formulation (Otter's classic encoding):
     * {@code p(x,y,z)} means {@code x*y = z}; axioms are closure {@code p(X,Y,m(X,Y))}, left
     * identity, left inverse and both directions of associativity.
     *
     * @param theorem an index into {@link #GROUP_THEOREMS}
     * @return the problem
     */
    public static Problem group(int theorem) {
        Builder b = Problem.builder("group-" + GROUP_THEOREMS.get(theorem));
        b.axiom("p(X,Y,m(X,Y))");
        b.axiom("p(e,X,X)");
        b.axiom("p(i(X),X,e)");
        b.axiom("~p(X,Y,U) | ~p(Y,Z,V) | ~p(U,Z,W) | p(X,V,W)");
        b.axiom("~p(X,Y,U) | ~p(Y,Z,V) | ~p(X,V,W) | p(U,Z,W)");
        switch (theorem) {
            case 0 -> b.goal("~p(a,e,a)");
            case 1 -> b.goal("~p(a,i(a),e)");
            case 2 -> b.goal("~p(m(a,b),i(m(a,b)),e)");
            case 3 -> {
                b.axiom("p(X,X,e)");
                b.goal("p(a,b,c)");
                b.goal("~p(b,a,c)");
            }
            case 4 -> b.goal("~p(a,m(i(a),a),a)");
            default -> b.goal("~p(i(i(a)),e,a)");
        }
        return b.build();
    }
}
