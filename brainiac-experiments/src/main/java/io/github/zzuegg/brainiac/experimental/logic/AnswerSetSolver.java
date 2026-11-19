package io.github.zzuegg.brainiac.experimental.logic;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Answer set programming on {@link CdclSolver} (Gelfond &amp; Lifschitz, ICLP 1988; Lin &amp; Zhao,
 * "ASSAT", AIJ 2004; Gebser, Kaufmann &amp; Schaub, "Conflict-driven answer set solving", AIJ 2012).
 *
 * <p>Programs are written in a small subset of the gringo language: facts {@code p(a,1).},
 * normal rules {@code h(X) :- b(X,Y), not c(Y), X != Y.}, integrity constraints
 * {@code :- p(X), q(X).} and choice rules {@code { a(X) } :- d(X).} or with cardinality bounds and
 * one conditional element {@code 1 { col(X,C) : colour(C) } 1 :- node(X).}. Terms are variables
 * (upper case), constants and integers, with {@code +} and {@code -} in arguments and builtins
 * ({@code == != < <= > >=}). Element conditions must be fact predicates, and choice rules need a
 * non-empty positive body.
 *
 * <ul>
 *   <li>Grounding is semi-naive and bottom up: each round instantiates every rule against the
 *       atoms that may be true so far, emitting only instantiations that use at least one atom
 *       new in the previous round, until no atom is added. Atoms derivable by negation-free rules
 *       from facts are certain and simplified away.</li>
 *   <li>The ground program becomes CNF by Clark's completion: one Tseitin variable per rule body,
 *       {@code body -> head}, every atom implies the disjunction of its supporting bodies, choice
 *       bounds by sequential counters, constraints as clauses.</li>
 *   <li>For tight programs the models of the completion are the answer sets. For non-tight ones
 *       (positive recursion, such as reachability in Hamiltonian cycles) a model may contain an
 *       unfounded set; the solver checks every model against the least fixpoint of its supported
 *       atoms and adds the loop formula of the unfounded set (each of its atoms implies one of its
 *       external supports) to the incremental CDCL solver until the model is stable. With loop
 *       formulas off, the solver returns completion models as they are (the ASSAT baseline that
 *       admits unsupported cycles).</li>
 *   <li>Enumeration blocks each answer set on its atoms and continues.</li>
 * </ul>
 */
public final class AnswerSetSolver {

    private AnswerSetSolver() {
    }

    // ---- syntax ----------------------------------------------------------------------------

    private static final int VAR = 0;
    private static final int CONST = 1;
    private static final int ARITH = 2;

    private record Term(int kind, String name, char op, Term left, Term right) {
    }

    private record Atom(String pred, Term[] args) {
    }

    private record Builtin(String op, Term left, Term right) {
    }

    private enum Kind {
        NORMAL, CONSTRAINT, CHOICE
    }

    private record Rule(Kind kind, Atom head, int lower, int upper, List<Atom> condition, List<Atom> pos,
            List<Atom> neg, List<Builtin> builtins) {
    }

    /** A parsed program. */
    public static final class Program {
        private final List<Rule> rules = new ArrayList<>();

        private Program() {
        }

        /**
         * Parses a program.
         *
         * @param text statements, each ended by a full stop
         * @return the program
         */
        public static Program parse(String text) {
            Program p = new Program();
            p.add(text);
            return p;
        }

        /**
         * Adds statements.
         *
         * @param text statements, each ended by a full stop
         * @return this program
         */
        public Program add(String text) {
            for (String raw : text.split("\\.")) {
                String s = raw.replaceAll("%[^\n]*", "").trim();
                if (!s.isEmpty()) {
                    rules.add(statement(s));
                }
            }
            return this;
        }

        /** @return the number of non-ground statements */
        public int size() {
            return rules.size();
        }
    }

    private static List<String> splitTop(String s, char sep) {
        List<String> out = new ArrayList<>();
        int depth = 0;
        int start = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(' || c == '{') {
                depth++;
            } else if (c == ')' || c == '}') {
                depth--;
            } else if (c == sep && depth == 0) {
                out.add(s.substring(start, i).trim());
                start = i + 1;
            }
        }
        out.add(s.substring(start).trim());
        return out;
    }

    private static Rule statement(String s) {
        String head;
        String body;
        int arrow = s.indexOf(":-");
        if (arrow >= 0) {
            head = s.substring(0, arrow).trim();
            body = s.substring(arrow + 2).trim();
        } else {
            head = s;
            body = "";
        }
        List<Atom> pos = new ArrayList<>();
        List<Atom> neg = new ArrayList<>();
        List<Builtin> builtins = new ArrayList<>();
        if (!body.isEmpty()) {
            for (String lit : splitTop(body, ',')) {
                literal(lit, pos, neg, builtins);
            }
        }
        if (head.isEmpty()) {
            return new Rule(Kind.CONSTRAINT, null, 0, 0, List.of(), pos, neg, builtins);
        }
        int open = head.indexOf('{');
        if (open >= 0) {
            int close = head.lastIndexOf('}');
            String lo = head.substring(0, open).trim();
            String hi = head.substring(close + 1).trim();
            String inner = head.substring(open + 1, close).trim();
            List<Atom> cond = new ArrayList<>();
            int colon = inner.indexOf(':');
            Atom element;
            if (colon >= 0) {
                element = atom(inner.substring(0, colon).trim());
                for (String c : splitTop(inner.substring(colon + 1).trim(), ',')) {
                    cond.add(atom(c));
                }
            } else {
                element = atom(inner);
            }
            return new Rule(Kind.CHOICE, element, lo.isEmpty() ? 0 : Integer.parseInt(lo),
                hi.isEmpty() ? Integer.MAX_VALUE : Integer.parseInt(hi), cond, pos, neg, builtins);
        }
        return new Rule(Kind.NORMAL, atom(head), 0, 0, List.of(), pos, neg, builtins);
    }

    private static final String[] OPS = {"!=", "==", "<=", ">=", "<", ">", "="};

    private static void literal(String lit, List<Atom> pos, List<Atom> neg, List<Builtin> builtins) {
        if (lit.startsWith("not ")) {
            neg.add(atom(lit.substring(4).trim()));
            return;
        }
        if (!lit.contains("(") || lit.indexOf('(') > firstOp(lit) && firstOp(lit) >= 0) {
            for (String op : OPS) {
                int i = lit.indexOf(op);
                if (i > 0) {
                    builtins.add(new Builtin(op.equals("=") ? "==" : op, term(lit.substring(0, i).trim()),
                        term(lit.substring(i + op.length()).trim())));
                    return;
                }
            }
        }
        pos.add(atom(lit));
    }

    private static int firstOp(String lit) {
        int best = -1;
        for (String op : OPS) {
            int i = lit.indexOf(op);
            if (i >= 0 && (best < 0 || i < best)) {
                best = i;
            }
        }
        return best;
    }

    private static Atom atom(String s) {
        int open = s.indexOf('(');
        if (open < 0) {
            return new Atom(s.trim(), new Term[0]);
        }
        String name = s.substring(0, open).trim();
        List<String> args = splitTop(s.substring(open + 1, s.lastIndexOf(')')), ',');
        Term[] ts = new Term[args.size()];
        for (int i = 0; i < ts.length; i++) {
            ts[i] = term(args.get(i));
        }
        return new Atom(name, ts);
    }

    private static Term term(String s) {
        s = s.trim();
        // left-associative + and - (not a leading minus)
        for (int i = s.length() - 1; i > 0; i--) {
            char c = s.charAt(i);
            if (c == '+' || c == '-') {
                return new Term(ARITH, null, c, term(s.substring(0, i)), term(s.substring(i + 1)));
            }
        }
        if (Character.isUpperCase(s.charAt(0)) || s.charAt(0) == '_') {
            return new Term(VAR, s, ' ', null, null);
        }
        return new Term(CONST, s, ' ', null, null);
    }

    // ---- grounding -------------------------------------------------------------------------

    /** A ground rule: atoms by id; {@code head} -1 for constraints. */
    private record GroundRule(Kind kind, int head, int lower, int upper, int[] elements, int[] pos, int[] neg) {
    }

    /** The grounder's state. */
    private static final class Grounder {
        final List<String> names = new ArrayList<>();
        final Map<String, Integer> ids = new HashMap<>();
        final Map<String, List<String[]>> byPred = new HashMap<>();
        final List<Integer> round = new ArrayList<>();
        final List<GroundRule> rules = new ArrayList<>();
        final Set<String> ruleKeys = new HashSet<>();

        int intern(String pred, String[] args) {
            String key = args.length == 0 ? pred : pred + "(" + String.join(",", args) + ")";
            Integer id = ids.get(key);
            if (id == null) {
                id = names.size();
                ids.put(key, id);
                names.add(key);
                round.add(Integer.MAX_VALUE);
            }
            return id;
        }

        /** Marks an atom possible in the given round; returns whether it is new. */
        boolean possible(String pred, String[] args, int r) {
            int id = intern(pred, args);
            if (round.get(id) != Integer.MAX_VALUE) {
                return false;
            }
            round.set(id, r);
            byPred.computeIfAbsent(pred + "/" + args.length, k -> new ArrayList<>()).add(args);
            return true;
        }
    }

    private static String eval(Term t, Map<String, String> b) {
        return switch (t.kind()) {
            case VAR -> b.get(t.name());
            case CONST -> t.name();
            default -> {
                String l = eval(t.left(), b);
                String r = eval(t.right(), b);
                if (l == null || r == null) {
                    yield null;
                }
                long x = Long.parseLong(l);
                long y = Long.parseLong(r);
                yield Long.toString(t.op() == '+' ? x + y : x - y);
            }
        };
    }

    private static boolean match(Atom a, String[] args, Map<String, String> b, List<String> boundHere) {
        for (int i = 0; i < args.length; i++) {
            Term t = a.args()[i];
            if (t.kind() == VAR) {
                String v = b.get(t.name());
                if (v == null) {
                    b.put(t.name(), args[i]);
                    boundHere.add(t.name());
                } else if (!v.equals(args[i])) {
                    return false;
                }
            } else {
                String v = eval(t, b);
                if (v == null || !v.equals(args[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean builtin(Builtin bi, Map<String, String> b) {
        String l = eval(bi.left(), b);
        String r = eval(bi.right(), b);
        int cmp;
        boolean numeric = l.matches("-?\\d+") && r.matches("-?\\d+");
        cmp = numeric ? Long.compare(Long.parseLong(l), Long.parseLong(r)) : l.compareTo(r);
        return switch (bi.op()) {
            case "==" -> cmp == 0;
            case "!=" -> cmp != 0;
            case "<" -> cmp < 0;
            case "<=" -> cmp <= 0;
            case ">" -> cmp > 0;
            default -> cmp >= 0;
        };
    }

    private static String[] groundArgs(Atom a, Map<String, String> b) {
        String[] out = new String[a.args().length];
        for (int i = 0; i < out.length; i++) {
            out[i] = eval(a.args()[i], b);
            if (out[i] == null) {
                throw new IllegalArgumentException("unsafe variable in " + a.pred());
            }
        }
        return out;
    }

    /** Joins the positive body; calls back for complete bindings. */
    private static void join(Grounder g, List<Atom> body, int i, Map<String, String> b, boolean usedNew, int r,
            List<Builtin> builtins, java.util.function.Consumer<Map<String, String>> out) {
        if (i == body.size()) {
            if (!usedNew && r > 0) {
                return;
            }
            for (Builtin bi : builtins) {
                if (!builtin(bi, b)) {
                    return;
                }
            }
            out.accept(b);
            return;
        }
        Atom a = body.get(i);
        List<String[]> cands = g.byPred.getOrDefault(a.pred() + "/" + a.args().length, List.of());
        int n = cands.size();
        for (int k = 0; k < n; k++) {
            String[] args = cands.get(k);
            int rd = g.round.get(g.ids.get(args.length == 0 ? a.pred() : a.pred() + "(" + String.join(",", args) + ")"));
            if (rd >= r && r > 0) {
                continue; // atoms of the current round wait for the next one
            }
            List<String> bound = new ArrayList<>();
            if (match(a, args, b, bound)) {
                join(g, body, i + 1, b, usedNew || rd == r - 1, r, builtins, out);
            }
            for (String v : bound) {
                b.remove(v);
            }
        }
    }

    /** The ground program. */
    private static Grounder ground(Program p) {
        Grounder g = new Grounder();
        boolean changed = true;
        for (int r = 0; changed; r++) {
            changed = false;
            int rr = r;
            List<Object[]> pendingAtoms = new ArrayList<>();
            for (Rule rule : p.rules) {
                join(g, rule.pos(), 0, new HashMap<>(), false, rr, rule.builtins(), b -> {
                    int[] pos = rule.pos().stream().mapToInt(a -> g.intern(a.pred(), groundArgs(a, b))).toArray();
                    int[] neg = rule.neg().stream().mapToInt(a -> g.intern(a.pred(), groundArgs(a, b))).toArray();
                    switch (rule.kind()) {
                        case NORMAL -> {
                            String[] h = groundArgs(rule.head(), b);
                            int head = g.intern(rule.head().pred(), h);
                            addRule(g, new GroundRule(Kind.NORMAL, head, 0, 0, new int[0], pos, neg));
                            pendingAtoms.add(new Object[] {rule.head().pred(), h});
                        }
                        case CONSTRAINT -> addRule(g, new GroundRule(Kind.CONSTRAINT, -1, 0, 0, new int[0], pos, neg));
                        default -> {
                            List<Integer> elems = new ArrayList<>();
                            Map<String, String> eb = new HashMap<>(b);
                            joinAll(g, rule.condition(), 0, eb, eb2 -> {
                                String[] h = groundArgs(rule.head(), eb2);
                                elems.add(g.intern(rule.head().pred(), h));
                                pendingAtoms.add(new Object[] {rule.head().pred(), h});
                            });
                            addRule(g, new GroundRule(Kind.CHOICE, -1, rule.lower(), rule.upper(),
                                elems.stream().mapToInt(Integer::intValue).toArray(), pos, neg));
                        }
                    }
                });
            }
            for (Object[] a : pendingAtoms) {
                changed |= g.possible((String) a[0], (String[]) a[1], rr);
            }
        }
        return g;
    }

    private static void joinAll(Grounder g, List<Atom> body, int i, Map<String, String> b,
            java.util.function.Consumer<Map<String, String>> out) {
        if (i == body.size()) {
            out.accept(b);
            return;
        }
        Atom a = body.get(i);
        for (String[] args : g.byPred.getOrDefault(a.pred() + "/" + a.args().length, List.of())) {
            List<String> bound = new ArrayList<>();
            if (match(a, args, b, bound)) {
                joinAll(g, body, i + 1, b, out);
            }
            for (String v : bound) {
                b.remove(v);
            }
        }
    }

    private static void addRule(Grounder g, GroundRule r) {
        String key = r.kind() + ":" + r.head() + ":" + r.lower() + ":" + r.upper() + ":" + Arrays.toString(r.elements())
            + Arrays.toString(r.pos()) + Arrays.toString(r.neg());
        if (g.ruleKeys.add(key)) {
            g.rules.add(r);
        }
    }

    // ---- solving ---------------------------------------------------------------------------

    /**
     * A solving run.
     *
     * @param models the answer sets (or completion models, without loop formulas), as atom sets
     * @param unsupported models returned that are not answer sets (only without loop formulas)
     * @param groundRules ground rules after simplification
     * @param atoms ground atoms that are neither certain nor impossible
     * @param clauses CNF clauses of the completion (before loop formulas)
     * @param loopFormulas loop-formula clauses added
     * @param firstMillis time to the first model (including grounding)
     * @param totalMillis total time
     * @param complete whether enumeration finished (no further model exists)
     */
    public record Result(List<Set<String>> models, int unsupported, int groundRules, int atoms, int clauses,
            int loopFormulas, double firstMillis, double totalMillis, boolean complete) {
    }

    /**
     * Grounds and solves a program.
     *
     * @param program the program
     * @param maxModels the most models to enumerate
     * @param loopFormulas whether to add loop formulas for unfounded sets (off: plain completion)
     * @return the result
     */
    public static Result solve(Program program, int maxModels, boolean loopFormulas) {
        long t0 = System.nanoTime();
        Grounder g = ground(program);
        int na = g.names.size();
        // certain atoms: least model of the negation-free rules over certain bodies
        boolean[] certain = new boolean[na];
        boolean[] possible = new boolean[na];
        for (int a = 0; a < na; a++) {
            possible[a] = g.round.get(a) != Integer.MAX_VALUE;
        }
        boolean grew = true;
        while (grew) {
            grew = false;
            for (GroundRule r : g.rules) {
                if (r.kind() == Kind.NORMAL && !certain[r.head()] && r.neg().length == 0 && allIn(r.pos(), certain)) {
                    certain[r.head()] = true;
                    grew = true;
                }
            }
        }
        // simplification
        List<GroundRule> rules = new ArrayList<>();
        boolean trivialUnsat = false;
        for (GroundRule r : g.rules) {
            boolean drop = false;
            List<Integer> pos = new ArrayList<>();
            for (int a : r.pos()) {
                if (!possible[a]) {
                    drop = true;
                } else if (!certain[a]) {
                    pos.add(a);
                }
            }
            List<Integer> neg = new ArrayList<>();
            for (int a : r.neg()) {
                if (certain[a]) {
                    drop = true;
                } else if (possible[a]) {
                    neg.add(a);
                }
            }
            if (drop || r.kind() == Kind.NORMAL && certain[r.head()]) {
                continue;
            }
            rules.add(new GroundRule(r.kind(), r.head(), r.lower(), r.upper(), r.elements(),
                pos.stream().mapToInt(Integer::intValue).toArray(), neg.stream().mapToInt(Integer::intValue).toArray()));
        }
        // CNF by completion
        int[] var = new int[na];
        int next = 1;
        int atoms = 0;
        for (int a = 0; a < na; a++) {
            if (possible[a] && !certain[a]) {
                var[a] = next++;
                atoms++;
            }
        }
        int atomVars = next - 1;
        List<int[]> cnf = new ArrayList<>();
        int[] body = new int[rules.size()]; // 0 = true
        List<List<Integer>> supports = new ArrayList<>();
        for (int a = 0; a < na; a++) {
            supports.add(new ArrayList<>());
        }
        for (int i = 0; i < rules.size(); i++) {
            GroundRule r = rules.get(i);
            int[] lits = new int[r.pos().length + r.neg().length];
            int k = 0;
            for (int a : r.pos()) {
                lits[k++] = var[a];
            }
            for (int a : r.neg()) {
                lits[k++] = -var[a];
            }
            if (lits.length == 0) {
                body[i] = 0;
            } else if (lits.length == 1) {
                body[i] = lits[0];
            } else {
                int b = next++;
                body[i] = b;
                int[] back = new int[lits.length + 1];
                back[0] = b;
                for (int j = 0; j < lits.length; j++) {
                    cnf.add(new int[] {-b, lits[j]});
                    back[j + 1] = -lits[j];
                }
                cnf.add(back);
            }
            switch (r.kind()) {
                case CONSTRAINT -> {
                    if (body[i] == 0) {
                        trivialUnsat = true;
                    } else {
                        cnf.add(new int[] {-body[i]});
                    }
                }
                case NORMAL -> {
                    cnf.add(body[i] == 0 ? new int[] {var[r.head()]} : new int[] {-body[i], var[r.head()]});
                    supports.get(r.head()).add(i);
                }
                default -> {
                    List<Integer> el = new ArrayList<>();
                    for (int a : r.elements()) {
                        if (certain[a]) {
                            continue;
                        }
                        supports.get(a).add(i);
                        el.add(var[a]);
                    }
                    int certainCount = 0;
                    for (int a : r.elements()) {
                        certainCount += certain[a] ? 1 : 0;
                    }
                    int lo = Math.max(0, r.lower() - certainCount);
                    int hi = r.upper() == Integer.MAX_VALUE ? Integer.MAX_VALUE : r.upper() - certainCount;
                    if (hi < 0 || lo > el.size()) {
                        if (body[i] == 0) {
                            trivialUnsat = true;
                        } else {
                            cnf.add(new int[] {-body[i]});
                        }
                    } else if (lo > 0 || hi < el.size()) {
                        next = cardinality(el, lo, Math.min(hi, el.size()), body[i], next, cnf);
                    }
                }
            }
        }
        for (int a = 0; a < na; a++) {
            if (var[a] == 0) {
                continue;
            }
            List<Integer> sup = supports.get(a);
            boolean free = false;
            int[] cl = new int[sup.size() + 1];
            cl[0] = -var[a];
            for (int j = 0; j < sup.size(); j++) {
                cl[j + 1] = body[sup.get(j)];
                free |= body[sup.get(j)] == 0;
            }
            if (!free) {
                cnf.add(cl);
            }
        }
        int completionClauses = cnf.size();
        CdclSolver solver = new CdclSolver(next - 1, CdclSolver.Options.defaults());
        boolean ok = !trivialUnsat;
        for (int[] c : cnf) {
            ok &= solver.addClause(c);
        }
        List<Set<String>> models = new ArrayList<>();
        int loops = 0;
        int unsupported = 0;
        double first = -1;
        boolean complete = false;
        while (models.size() < maxModels) {
            if (!ok || solver.solve() != CdclSolver.Status.SAT) {
                complete = true;
                break;
            }
            boolean[] m = solver.model();
            // the atoms founded by the model: least fixpoint of supports with positive bodies inside
            boolean[] in = certain.clone();
            boolean more = true;
            while (more) {
                more = false;
                for (int i = 0; i < rules.size(); i++) {
                    GroundRule r = rules.get(i);
                    if (body[i] != 0 && !m[Math.abs(body[i]) - 1] == body[i] > 0) {
                        continue;
                    }
                    if (!allIn(r.pos(), in)) {
                        continue;
                    }
                    if (r.kind() == Kind.NORMAL && !in[r.head()]) {
                        in[r.head()] = true;
                        more = true;
                    } else if (r.kind() == Kind.CHOICE) {
                        for (int a : r.elements()) {
                            if (!in[a] && var[a] != 0 && m[var[a] - 1]) {
                                in[a] = true;
                                more = true;
                            }
                        }
                    }
                }
            }
            BitSet unfounded = new BitSet();
            for (int a = 0; a < na; a++) {
                if (var[a] != 0 && m[var[a] - 1] && !in[a]) {
                    unfounded.set(a);
                }
            }
            if (!unfounded.isEmpty() && loopFormulas) {
                // external supports: bodies of rules for an unfounded atom whose positive body avoids the set
                Set<Integer> ext = new LinkedHashSet<>();
                for (int a = unfounded.nextSetBit(0); a >= 0; a = unfounded.nextSetBit(a + 1)) {
                    for (int i : supports.get(a)) {
                        boolean inside = false;
                        for (int p : rules.get(i).pos()) {
                            inside |= unfounded.get(p);
                        }
                        if (!inside) {
                            ext.add(body[i]);
                        }
                    }
                }
                if (ext.contains(0)) {
                    throw new IllegalStateException("an unfounded set with an unconditional support");
                }
                for (int a = unfounded.nextSetBit(0); a >= 0; a = unfounded.nextSetBit(a + 1)) {
                    int[] cl = new int[ext.size() + 1];
                    cl[0] = -var[a];
                    int k = 1;
                    for (int b : ext) {
                        cl[k++] = b;
                    }
                    ok &= solver.addClause(cl);
                    loops++;
                }
                continue;
            }
            if (!unfounded.isEmpty()) {
                unsupported++;
            }
            Set<String> model = new TreeSet<>();
            int[] block = new int[atomVars];
            for (int a = 0; a < na; a++) {
                if (certain[a] || var[a] != 0 && m[var[a] - 1]) {
                    model.add(g.names.get(a));
                }
                if (var[a] != 0) {
                    block[var[a] - 1] = m[var[a] - 1] ? -var[a] : var[a];
                }
            }
            models.add(model);
            if (first < 0) {
                first = (System.nanoTime() - t0) / 1e6;
            }
            ok &= block.length > 0 && solver.addClause(block);
        }
        return new Result(models, unsupported, rules.size(), atoms, completionClauses, loops, first,
            (System.nanoTime() - t0) / 1e6, complete);
    }

    private static boolean allIn(int[] xs, boolean[] set) {
        for (int x : xs) {
            if (!set[x]) {
                return false;
            }
        }
        return true;
    }

    /**
     * {@code body -> lo <= sum(xs) <= hi} by a sequential counter with full equivalences
     * ({@code s(i, j)}: at least {@code j} of the first {@code i} are true).
     */
    private static int cardinality(List<Integer> xs, int lo, int hi, int body, int next, List<int[]> cnf) {
        int n = xs.size();
        int top = Math.min(n, Math.max(lo, hi + 1));
        int[][] s = new int[n + 1][top + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= Math.min(i, top); j++) {
                s[i][j] = next++;
            }
        }
        for (int i = 1; i <= n; i++) {
            int x = xs.get(i - 1);
            for (int j = 1; j <= Math.min(i, top); j++) {
                int a = s[i][j];
                int b = j <= i - 1 ? s[i - 1][j] : 0;      // 0: false
                int c = j == 1 ? 0 : s[i - 1][j - 1];      // for j == 1, s(i-1, 0) is true
                boolean cTrue = j == 1;
                // a <-> b or (x and c)
                if (b != 0) {
                    cnf.add(new int[] {-b, a});
                }
                cnf.add(cTrue ? new int[] {-x, a} : new int[] {-x, -c, a});
                List<Integer> b1 = new ArrayList<>(List.of(-a, x));
                if (b != 0) {
                    b1.add(b);
                }
                cnf.add(b1.stream().mapToInt(Integer::intValue).toArray());
                if (!cTrue) {
                    List<Integer> b2 = new ArrayList<>(List.of(-a, c));
                    if (b != 0) {
                        b2.add(b);
                    }
                    cnf.add(b2.stream().mapToInt(Integer::intValue).toArray());
                }
            }
        }
        if (lo > 0) {
            cnf.add(body == 0 ? new int[] {s[n][lo]} : new int[] {-body, s[n][lo]});
        }
        if (hi < n) {
            cnf.add(body == 0 ? new int[] {-s[n][hi + 1]} : new int[] {-body, -s[n][hi + 1]});
        }
        return next;
    }
}
