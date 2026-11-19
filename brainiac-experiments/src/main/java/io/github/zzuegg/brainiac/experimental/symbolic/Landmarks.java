package io.github.zzuegg.brainiac.experimental.symbolic;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/**
 * Landmark heuristics for {@link StripsTask}s. A landmark is a fact (or a set of actions) that
 * every plan must make true (or use).
 *
 * <ul>
 *   <li>{@link #lmCut}: the LM-cut heuristic (Helmert &amp; Domshlak, "Landmarks, critical paths
 *       and abstractions: what's the difference anyway?", ICAPS 2009). It computes h_max, builds
 *       the justification graph in which every action links its costliest precondition (its
 *       precondition choice function) to its adds, takes the cut between the facts reachable
 *       from the state without entering the zero-cost goal zone and that zone, records it as a
 *       disjunctive action landmark of cost {@code m} (the cheapest action of the cut), reduces
 *       the cut's costs by {@code m} and repeats until h_max of the goal is 0. The sum of the
 *       {@code m}s is admissible and dominates h_max.</li>
 *   <li>{@link #generate}: fact landmarks found by back-chaining from the goals (Richter,
 *       Helmert &amp; Westphal, AAAI 2008, simplified): for a landmark {@code L} not true
 *       initially, the achievers that can be applied before {@code L} (their preconditions are
 *       relaxed-reachable without any achiever of {@code L}) are collected and the facts all of
 *       them require become landmarks, greedy-necessarily ordered before {@code L}.</li>
 *   <li>{@link #landmarkCount}: LAMA's path-dependent landmark-count heuristic (Richter &amp;
 *       Westphal, JAIR 2010): landmarks not yet accepted along the path, plus accepted ones that
 *       are required again (false goal landmarks, and false landmarks that must hold before an
 *       unaccepted one). Inadmissible, but very informative for satisficing search. Its
 *       preferred operators are the applicable achievers of unaccepted landmarks.</li>
 * </ul>
 */
public final class Landmarks {

    private final StripsTask task;
    private final int[] landmarks;
    private final int[][] predecessors;
    private final boolean[] goalLandmark;
    private final int orderings;

    private Landmarks(StripsTask task, int[] landmarks, int[][] predecessors, boolean[] goalLandmark,
            int orderings) {
        this.task = task;
        this.landmarks = landmarks;
        this.predecessors = predecessors;
        this.goalLandmark = goalLandmark;
        this.orderings = orderings;
    }

    /** @return the number of fact landmarks */
    public int size() {
        return landmarks.length;
    }

    /** @return the number of greedy-necessary orderings */
    public int orderings() {
        return orderings;
    }

    /**
     * @param i a landmark index
     * @return its fact
     */
    public int fact(int i) {
        return landmarks[i];
    }

    /**
     * @param i a landmark index
     * @return the landmarks ordered greedy-necessarily before it (do not modify)
     */
    public int[] predecessors(int i) {
        return predecessors[i];
    }

    // ---- LM-cut ------------------------------------------------------------------------------

    /**
     * The LM-cut heuristic as an evaluator.
     *
     * @param task the task
     * @return the evaluator
     */
    public static StateSpaceSearch.Evaluator lmCut(StripsTask task) {
        return (s, d) -> StateSpaceSearch.Evaluation.of(lmCutValue(task, s, null));
    }

    /**
     * Computes LM-cut.
     *
     * @param task the task
     * @param state the state
     * @param cuts if not null, receives the action landmarks (one array of action ids per cut)
     * @return the estimate, infinity for a relaxed dead end
     */
    public static double lmCutValue(StripsTask task, BitSet state, List<int[]> cuts) {
        int na = task.actions().size();
        int nf = task.facts();
        double[] cost = new double[na];
        for (StripsTask.Action a : task.actions()) {
            cost[a.id()] = a.cost();
        }
        int[] goal = task.goal();
        double h = 0;
        int[] pcf = new int[na];
        for (int round = 0; round < 10_000; round++) {
            RelaxationHeuristics.Exploration x = RelaxationHeuristics.explore(task, state, cost, true);
            double hg = 0;
            int goalPcf = -1;
            for (int g : goal) {
                double c = x.factCost()[g];
                if (Double.isInfinite(c)) {
                    return Double.POSITIVE_INFINITY;
                }
                if (goalPcf < 0 || c > hg) {
                    hg = c;
                    goalPcf = g;
                }
            }
            if (hg <= 1e-12) {
                break;
            }
            for (StripsTask.Action a : task.actions()) {
                int best = -1;
                double bc = -1;
                if (Double.isFinite(x.actionValue()[a.id()])) {
                    for (int p : a.pre()) {
                        double c = x.factCost()[p];
                        if (c > bc) {
                            bc = c;
                            best = p;
                        }
                    }
                    pcf[a.id()] = best; // -1: no precondition, hangs off the state
                } else {
                    pcf[a.id()] = -2; // unreachable
                }
            }
            // goal zone: facts that reach the goal through zero-cost justification edges
            BitSet zone = new BitSet(nf);
            ArrayDeque<Integer> q = new ArrayDeque<>();
            zone.set(goalPcf);
            q.add(goalPcf);
            while (!q.isEmpty()) {
                int f = q.poll();
                for (int a : task.achievers(f)) {
                    int p = pcf[a];
                    if (p >= 0 && cost[a] <= 1e-12 && !zone.get(p)) {
                        zone.set(p);
                        q.add(p);
                    }
                }
            }
            // forward from the state without entering the goal zone
            BitSet before = new BitSet(nf);
            BitSet cut = new BitSet(na);
            for (int f = state.nextSetBit(0); f >= 0; f = state.nextSetBit(f + 1)) {
                before.set(f);
                q.add(f);
            }
            for (StripsTask.Action a : task.actions()) {
                if (pcf[a.id()] == -1) {
                    reach(a, zone, before, cut, q);
                }
            }
            while (!q.isEmpty()) {
                int f = q.poll();
                for (int a : task.consumers(f)) {
                    if (pcf[a] == f) {
                        reach(task.actions().get(a), zone, before, cut, q);
                    }
                }
            }
            double m = Double.POSITIVE_INFINITY;
            for (int a = cut.nextSetBit(0); a >= 0; a = cut.nextSetBit(a + 1)) {
                m = Math.min(m, cost[a]);
            }
            if (cut.isEmpty() || Double.isInfinite(m)) {
                break;
            }
            h += m;
            for (int a = cut.nextSetBit(0); a >= 0; a = cut.nextSetBit(a + 1)) {
                cost[a] -= m;
            }
            if (cuts != null) {
                cuts.add(cut.stream().toArray());
            }
        }
        return h;
    }

    private static void reach(StripsTask.Action a, BitSet zone, BitSet before, BitSet cut, ArrayDeque<Integer> q) {
        for (int f : a.add()) {
            if (zone.get(f)) {
                cut.set(a.id());
            } else if (!before.get(f)) {
                before.set(f);
                q.add(f);
            }
        }
    }

    // ---- fact landmarks ------------------------------------------------------------------------

    /**
     * Finds fact landmarks and greedy-necessary orderings by back-chaining from the goals.
     *
     * @param task the task
     * @return the landmark graph
     */
    public static Landmarks generate(StripsTask task) {
        BitSet init = task.initialState();
        List<Integer> lms = new ArrayList<>();
        int[] lmIndex = new int[task.facts()];
        java.util.Arrays.fill(lmIndex, -1);
        List<List<Integer>> preds = new ArrayList<>();
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        int[] goal = task.goal();
        for (int g : goal) {
            if (lmIndex[g] < 0) {
                lmIndex[g] = lms.size();
                lms.add(g);
                preds.add(new ArrayList<>());
                queue.add(g);
            }
        }
        int orderings = 0;
        while (!queue.isEmpty()) {
            int l = queue.poll();
            if (init.get(l)) {
                continue;
            }
            BitSet reach = reachableWithout(task, init, l);
            BitSet shared = null;
            for (int a : task.achievers(l)) {
                StripsTask.Action act = task.actions().get(a);
                boolean ok = true;
                for (int p : act.pre()) {
                    if (!reach.get(p)) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) {
                    continue;
                }
                BitSet pre = new BitSet();
                for (int p : act.pre()) {
                    pre.set(p);
                }
                if (shared == null) {
                    shared = pre;
                } else {
                    shared.and(pre);
                }
            }
            if (shared == null) {
                continue;
            }
            for (int p = shared.nextSetBit(0); p >= 0; p = shared.nextSetBit(p + 1)) {
                if (lmIndex[p] < 0) {
                    lmIndex[p] = lms.size();
                    lms.add(p);
                    preds.add(new ArrayList<>());
                    queue.add(p);
                }
                if (!preds.get(lmIndex[l]).contains(lmIndex[p])) {
                    preds.get(lmIndex[l]).add(lmIndex[p]);
                    orderings++;
                }
            }
        }
        int[] facts = lms.stream().mapToInt(Integer::intValue).toArray();
        int[][] pr = new int[facts.length][];
        boolean[] isGoal = new boolean[facts.length];
        for (int i = 0; i < facts.length; i++) {
            pr[i] = preds.get(i).stream().mapToInt(Integer::intValue).toArray();
        }
        for (int g : goal) {
            isGoal[lmIndex[g]] = true;
        }
        return new Landmarks(task, facts, pr, isGoal, orderings);
    }

    private static BitSet reachableWithout(StripsTask task, BitSet init, int excluded) {
        BitSet reach = (BitSet) init.clone();
        int[] remaining = new int[task.actions().size()];
        ArrayDeque<Integer> q = new ArrayDeque<>();
        boolean[] fired = new boolean[remaining.length];
        for (StripsTask.Action a : task.actions()) {
            remaining[a.id()] = a.pre().length;
        }
        for (int f = init.nextSetBit(0); f >= 0; f = init.nextSetBit(f + 1)) {
            q.add(f);
        }
        List<StripsTask.Action> start = new ArrayList<>();
        for (StripsTask.Action a : task.actions()) {
            if (a.pre().length == 0) {
                start.add(a);
            }
        }
        for (StripsTask.Action a : start) {
            fireIfAllowed(a, excluded, reach, q, fired);
        }
        while (!q.isEmpty()) {
            int f = q.poll();
            for (int a : task.consumers(f)) {
                if (--remaining[a] == 0) {
                    fireIfAllowed(task.actions().get(a), excluded, reach, q, fired);
                }
            }
        }
        return reach;
    }

    private static void fireIfAllowed(StripsTask.Action a, int excluded, BitSet reach, ArrayDeque<Integer> q,
            boolean[] fired) {
        if (fired[a.id()]) {
            return;
        }
        for (int f : a.add()) {
            if (f == excluded) {
                return;
            }
        }
        fired[a.id()] = true;
        for (int f : a.add()) {
            if (!reach.get(f)) {
                reach.set(f);
                q.add(f);
            }
        }
    }

    // ---- landmark count -------------------------------------------------------------------------

    /**
     * LAMA's landmark-count heuristic over these landmarks (path dependent).
     *
     * @return the evaluator
     */
    public StateSpaceSearch.Evaluator landmarkCount() {
        return new StateSpaceSearch.Evaluator() {
            @Override
            public Object initialData(BitSet init) {
                BitSet acc = new BitSet();
                for (int i = 0; i < landmarks.length; i++) {
                    if (init.get(landmarks[i]) && predecessors[i].length == 0) {
                        acc.set(i);
                    }
                }
                return acc;
            }

            @Override
            public Object progress(Object parentData, BitSet parent, StripsTask.Action action, BitSet child) {
                BitSet before = (BitSet) parentData;
                BitSet acc = (BitSet) before.clone();
                for (int i = 0; i < landmarks.length; i++) {
                    if (!acc.get(i) && child.get(landmarks[i])) {
                        boolean ok = true;
                        for (int p : predecessors[i]) {
                            if (!before.get(p)) {
                                ok = false;
                                break;
                            }
                        }
                        if (ok) {
                            acc.set(i);
                        }
                    }
                }
                return acc;
            }

            @Override
            public StateSpaceSearch.Evaluation evaluate(BitSet state, Object data) {
                BitSet acc = (BitSet) data;
                int h = 0;
                BitSet needed = new BitSet();
                for (int i = 0; i < landmarks.length; i++) {
                    if (!acc.get(i)) {
                        h++;
                        needed.set(landmarks[i]);
                        for (int p : predecessors[i]) {
                            if (acc.get(p) && !state.get(landmarks[p])) {
                                if (!needed.get(landmarks[p])) {
                                    h++;
                                    needed.set(landmarks[p]);
                                }
                            }
                        }
                    } else if (goalLandmark[i] && !state.get(landmarks[i])) {
                        h++;
                        needed.set(landmarks[i]);
                    }
                }
                if (h == 0 && !task.isGoal(state)) {
                    h = 1;
                }
                List<Integer> pref = new ArrayList<>();
                for (StripsTask.Action a : task.applicable(state)) {
                    for (int f : a.add()) {
                        if (needed.get(f) && !state.get(f)) {
                            pref.add(a.id());
                            break;
                        }
                    }
                }
                return new StateSpaceSearch.Evaluation(h, pref.stream().mapToInt(Integer::intValue).toArray());
            }
        };
    }
}
