package io.github.zzuegg.brainiac.experimental.symbolic;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.function.ToDoubleFunction;

/**
 * Forward state-space search for {@link StripsTask}s, in the forms classical planners use:
 * A* (optimal with an admissible heuristic, with re-opening), weighted A*, eager greedy best-first
 * search (GBFS: every generated successor is evaluated) and lazy GBFS (Richter &amp; Helmert,
 * ICAPS 2009: a successor is inserted with its parent's value and evaluated only when it is
 * expanded, which saves most evaluations when the branching factor is high).
 *
 * <p>Several heuristics can be combined as in LAMA (Richter &amp; Westphal, JAIR 2010): every
 * evaluator gets its own open list and the lists are used in alternation. With preferred
 * operators on, every list has a twin that only receives successors reached by a preferred
 * operator of their parent (the actions of the relaxed plan, see
 * {@link RelaxationHeuristics}); as in Fast Downward the preferred lists get a priority boost of
 * 1000 whenever a new best heuristic value is found.
 *
 * <p>Evaluators may be path dependent (LAMA's landmark count): they then receive a per-node
 * datum propagated from the parent along each generating action.
 */
public final class StateSpaceSearch {

    private StateSpaceSearch() {
    }

    /** The search algorithm. */
    public enum Method {
        /** A* on {@code g + h} with the first evaluator, re-opening closed nodes. */
        ASTAR,
        /** Weighted A* on {@code g + w h}. */
        WEIGHTED_ASTAR,
        /** Eager greedy best-first search. */
        GBFS,
        /** Lazy greedy best-first search (deferred evaluation). */
        LAZY_GBFS
    }

    /**
     * A heuristic value and the preferred operators that came with it.
     *
     * @param h the estimate; infinity marks a recognised dead end
     * @param preferred ids of preferred actions applicable in the state (may be empty)
     */
    public record Evaluation(double h, int[] preferred) {

        private static final int[] NONE = new int[0];

        /**
         * @param h the estimate
         * @return an evaluation without preferred operators
         */
        public static Evaluation of(double h) {
            return new Evaluation(h, NONE);
        }
    }

    /** A state evaluator (a heuristic), optionally path dependent. */
    public interface Evaluator {

        /**
         * Evaluates a state.
         *
         * @param state the state
         * @param data the node's path-dependent datum (null for path-independent evaluators)
         * @return the evaluation
         */
        Evaluation evaluate(BitSet state, Object data);

        /**
         * @param init the initial state
         * @return the datum of the root node
         */
        default Object initialData(BitSet init) {
            return null;
        }

        /**
         * Propagates the datum along an action.
         *
         * @param parentData the parent's datum
         * @param parent the parent state
         * @param action the action
         * @param child the child state
         * @return the child's datum
         */
        default Object progress(Object parentData, BitSet parent, StripsTask.Action action, BitSet child) {
            return null;
        }

        /**
         * Wraps a plain state function.
         *
         * @param h the heuristic
         * @return the evaluator
         */
        static Evaluator of(ToDoubleFunction<BitSet> h) {
            return (s, d) -> Evaluation.of(h.applyAsDouble(s));
        }
    }

    /**
     * A search result.
     *
     * @param plan the actions from the initial state to a goal, empty if none was found (or the
     *     initial state is a goal)
     * @param found whether a goal was reached
     * @param cost the plan cost
     * @param expanded nodes expanded
     * @param evaluated heuristic evaluations (per evaluator, summed)
     * @param generated successor nodes generated
     * @param millis wall time
     */
    public record Result(List<StripsTask.Action> plan, boolean found, double cost, int expanded, int evaluated,
            int generated, double millis) {
    }

    private record Entry(double key, double tie, long order, int node, int action) {
    }

    private static final Comparator<Entry> ORDER = Comparator.comparingDouble(Entry::key)
        .thenComparingDouble(Entry::tie).thenComparingLong(Entry::order);

    /**
     * A* with one evaluator.
     *
     * @param task the task
     * @param h the heuristic
     * @param maxExpansions the expansion budget
     * @return the result
     */
    public static Result astar(StripsTask task, Evaluator h, int maxExpansions) {
        return search(task, Method.ASTAR, List.of(h), false, 1, maxExpansions);
    }

    /**
     * Runs a search.
     *
     * @param task the task
     * @param method the algorithm
     * @param evaluators one or more heuristics (A* uses the first)
     * @param preferred whether to use preferred-operator open lists (GBFS and lazy GBFS)
     * @param weight {@code w} for weighted A*
     * @param maxExpansions the expansion budget
     * @return the result
     */
    public static Result search(StripsTask task, Method method, List<Evaluator> evaluators, boolean preferred,
            double weight, int maxExpansions) {
        long t0 = System.nanoTime();
        Result r = switch (method) {
            case ASTAR -> astar(task, evaluators.get(0), 1, maxExpansions);
            case WEIGHTED_ASTAR -> astar(task, evaluators.get(0), weight, maxExpansions);
            case GBFS -> greedy(task, evaluators, preferred, false, maxExpansions);
            case LAZY_GBFS -> greedy(task, evaluators, preferred, true, maxExpansions);
        };
        return new Result(r.plan(), r.found(), r.cost(), r.expanded(), r.evaluated(), r.generated(),
            (System.nanoTime() - t0) / 1e6);
    }

    /** The node store shared by all algorithms. */
    private static final class Nodes {
        final List<BitSet> states = new ArrayList<>();
        final List<Double> g = new ArrayList<>();
        final List<Integer> parent = new ArrayList<>();
        final List<Integer> via = new ArrayList<>();
        final List<Object[]> data = new ArrayList<>();
        final Map<BitSet, Integer> index = new HashMap<>();

        int add(BitSet s, double cost, int p, int a, Object[] d) {
            int id = states.size();
            states.add(s);
            g.add(cost);
            parent.add(p);
            via.add(a);
            data.add(d);
            index.put(s, id);
            return id;
        }

        List<StripsTask.Action> plan(StripsTask task, int node) {
            List<StripsTask.Action> out = new ArrayList<>();
            for (int c = node; parent.get(c) >= 0; c = parent.get(c)) {
                out.add(task.actions().get(via.get(c)));
            }
            Collections.reverse(out);
            return out;
        }
    }

    private static Object[] rootData(List<Evaluator> ev, BitSet s) {
        Object[] d = new Object[ev.size()];
        for (int i = 0; i < d.length; i++) {
            d[i] = ev.get(i).initialData(s);
        }
        return d;
    }

    private static Object[] childData(List<Evaluator> ev, Object[] pd, BitSet p, StripsTask.Action a, BitSet c) {
        Object[] d = new Object[ev.size()];
        for (int i = 0; i < d.length; i++) {
            d[i] = ev.get(i).progress(pd[i], p, a, c);
        }
        return d;
    }

    private static Result astar(StripsTask task, Evaluator h, double w, int maxExpansions) {
        List<Evaluator> ev = List.of(h);
        Nodes n = new Nodes();
        BitSet s0 = task.initialState();
        Object[] d0 = rootData(ev, s0);
        int root = n.add(s0, 0, -1, -1, d0);
        List<Double> hs = new ArrayList<>();
        double h0 = h.evaluate(s0, d0[0]).h();
        hs.add(h0);
        int evaluated = 1;
        int generated = 1;
        int expanded = 0;
        long order = 0;
        if (Double.isInfinite(h0)) {
            return new Result(List.of(), false, Double.POSITIVE_INFINITY, 0, evaluated, generated, 0);
        }
        PriorityQueue<Entry> open = new PriorityQueue<>(ORDER);
        open.add(new Entry(w * h0, h0, order++, root, -1));
        BitSet closedAt = new BitSet();
        List<Double> closedG = new ArrayList<>();
        closedG.add(Double.POSITIVE_INFINITY);
        while (!open.isEmpty() && expanded < maxExpansions) {
            Entry e = open.poll();
            int c = e.node();
            double gc = n.g.get(c);
            if (e.key() > gc + w * hs.get(c) + 1e-9) {
                continue;
            }
            if (closedAt.get(c) && closedG.get(c) <= gc) {
                continue;
            }
            closedAt.set(c);
            closedG.set(c, gc);
            expanded++;
            BitSet s = n.states.get(c);
            if (task.isGoal(s)) {
                return new Result(n.plan(task, c), true, gc, expanded, evaluated, generated, 0);
            }
            for (StripsTask.Action a : task.applicable(s)) {
                BitSet t = a.apply(s);
                double ng = gc + a.cost();
                Integer id = n.index.get(t);
                generated++;
                if (id == null) {
                    Object[] dt = childData(ev, n.data.get(c), s, a, t);
                    id = n.add(t, ng, c, a.id(), dt);
                    double ht = h.evaluate(t, dt[0]).h();
                    evaluated++;
                    hs.add(ht);
                    closedG.add(Double.POSITIVE_INFINITY);
                    if (Double.isInfinite(ht)) {
                        continue;
                    }
                    open.add(new Entry(ng + w * ht, ht, order++, id, -1));
                } else if (ng < n.g.get(id) - 1e-12) {
                    n.g.set(id, ng);
                    n.parent.set(id, c);
                    n.via.set(id, a.id());
                    double ht = hs.get(id);
                    if (Double.isInfinite(ht)) {
                        continue;
                    }
                    open.add(new Entry(ng + w * ht, ht, order++, id, -1));
                }
            }
        }
        return new Result(List.of(), false, Double.POSITIVE_INFINITY, expanded, evaluated, generated, 0);
    }

    private static Result greedy(StripsTask task, List<Evaluator> ev, boolean usePreferred, boolean lazy,
            int maxExpansions) {
        int k = ev.size();
        int queues = usePreferred ? 2 * k : k;
        List<PriorityQueue<Entry>> open = new ArrayList<>();
        for (int q = 0; q < queues; q++) {
            open.add(new PriorityQueue<>(ORDER));
        }
        int[] priority = new int[queues];
        double[] best = new double[k];
        java.util.Arrays.fill(best, Double.POSITIVE_INFINITY);
        Nodes n = new Nodes();
        BitSet s0 = task.initialState();
        Object[] d0 = rootData(ev, s0);
        int root = n.add(s0, 0, -1, -1, d0);
        BitSet closed = new BitSet();
        int evaluated = 0;
        int generated = 1;
        int expanded = 0;
        long order = 0;
        // per node: heuristic values and preferred-ness (eager)
        List<double[]> hv = new ArrayList<>();
        Evaluation[] e0 = new Evaluation[k];
        double[] h0 = new double[k];
        for (int i = 0; i < k; i++) {
            e0[i] = ev.get(i).evaluate(s0, d0[i]);
            h0[i] = e0[i].h();
            evaluated++;
        }
        hv.add(h0);
        List<BitSet> prefs = new ArrayList<>();
        BitSet p0 = new BitSet();
        for (Evaluation e : e0) {
            for (int p : e.preferred()) {
                p0.set(p);
            }
        }
        prefs.add(p0);
        if (Double.isInfinite(h0[0])) {
            return new Result(List.of(), false, Double.POSITIVE_INFINITY, 0, evaluated, generated, 0);
        }
        if (lazy) {
            // lazy entries hold (parent node, action); the root is a pseudo entry
            open.get(0).add(new Entry(0, 0, order++, root, -1));
        } else {
            for (int i = 0; i < k; i++) {
                open.get(i).add(new Entry(h0[i], 0, order++, root, -1));
            }
        }
        while (expanded < maxExpansions) {
            int q = -1;
            for (int i = 0; i < queues; i++) {
                if (!open.get(i).isEmpty() && (q < 0 || priority[i] < priority[q])) {
                    q = i;
                }
            }
            if (q < 0) {
                break;
            }
            priority[q]++;
            Entry e = open.get(q).poll();
            int node;
            BitSet s;
            double[] hs;
            Evaluation[] evs = null;
            if (lazy) {
                if (e.action() < 0) {
                    node = e.node();
                    s = n.states.get(node);
                    evs = e0;
                    hs = h0;
                } else {
                    BitSet ps = n.states.get(e.node());
                    StripsTask.Action a = task.actions().get(e.action());
                    BitSet t = a.apply(ps);
                    Integer id = n.index.get(t);
                    if (id != null && closed.get(id)) {
                        continue;
                    }
                    double ng = n.g.get(e.node()) + a.cost();
                    if (id == null) {
                        id = n.add(t, ng, e.node(), a.id(), childData(ev, n.data.get(e.node()), ps, a, t));
                        hv.add(null);
                    }
                    node = id;
                    s = t;
                    evs = new Evaluation[k];
                    hs = new double[k];
                    for (int i = 0; i < k; i++) {
                        evs[i] = ev.get(i).evaluate(s, n.data.get(node)[i]);
                        hs[i] = evs[i].h();
                        evaluated++;
                    }
                    hv.set(node, hs);
                    if (Double.isInfinite(hs[0])) {
                        closed.set(node);
                        continue;
                    }
                }
            } else {
                node = e.node();
                if (closed.get(node)) {
                    continue;
                }
                s = n.states.get(node);
                hs = hv.get(node);
            }
            closed.set(node);
            expanded++;
            if (task.isGoal(s)) {
                return new Result(n.plan(task, node), true, n.g.get(node), expanded, evaluated, generated, 0);
            }
            boolean progress = false;
            for (int i = 0; i < k; i++) {
                if (hs[i] < best[i]) {
                    best[i] = hs[i];
                    progress = true;
                }
            }
            if (progress && usePreferred) {
                for (int i = k; i < queues; i++) {
                    priority[i] -= 1000;
                }
            }
            List<StripsTask.Action> succ = task.applicable(s);
            if (lazy) {
                BitSet pref = new BitSet();
                for (int i = 0; i < k; i++) {
                    for (int p : evs[i].preferred()) {
                        pref.set(p);
                    }
                }
                for (StripsTask.Action a : succ) {
                    boolean isPref = pref.get(a.id());
                    for (int i = 0; i < k; i++) {
                        // preferred successors go first among equal keys
                        open.get(i).add(new Entry(hs[i], isPref ? 0 : 1, order++, node, a.id()));
                        if (usePreferred && isPref) {
                            open.get(k + i).add(new Entry(hs[i], 0, order++, node, a.id()));
                        }
                    }
                    generated++;
                }
            } else {
                BitSet pref = prefs.get(node);
                for (StripsTask.Action a : succ) {
                    BitSet t = a.apply(s);
                    generated++;
                    if (n.index.containsKey(t)) {
                        continue;
                    }
                    Object[] dt = childData(ev, n.data.get(node), s, a, t);
                    int id = n.add(t, n.g.get(node) + a.cost(), node, a.id(), dt);
                    double[] ht = new double[k];
                    BitSet pt = new BitSet();
                    for (int i = 0; i < k; i++) {
                        Evaluation et = ev.get(i).evaluate(t, dt[i]);
                        ht[i] = et.h();
                        if (usePreferred) {
                            for (int p : et.preferred()) {
                                pt.set(p);
                            }
                        }
                        evaluated++;
                    }
                    hv.add(ht);
                    prefs.add(pt);
                    if (Double.isInfinite(ht[0])) {
                        closed.set(id);
                        continue;
                    }
                    for (int i = 0; i < k; i++) {
                        open.get(i).add(new Entry(ht[i], 0, order++, id, -1));
                        if (usePreferred && pref.get(a.id())) {
                            open.get(k + i).add(new Entry(ht[i], 0, order++, id, -1));
                        }
                    }
                }
            }
        }
        return new Result(List.of(), false, Double.POSITIVE_INFINITY, expanded, evaluated, generated, 0);
    }
}
