package io.github.zzuegg.brainiac.experimental.symbolic;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Greedy best-first search in partial space (Wang &amp; Trevizan, "Graph learning for planning:
 * the story thus far and open challenges" line of work; partial-space search and action-set
 * heuristics, AAAI 2025, <a href="https://arxiv.org/abs/2504.20318">arXiv:2504.20318</a>).
 *
 * <p>Instead of generating every successor of a state, a node of partial space is a state with a
 * partially instantiated action: first the action schema is chosen, then one argument at a time.
 * Each such node stands for the set of applicable ground actions consistent with it and is scored
 * by an action-set heuristic {@code h(s, A)} ({@link RankingHeuristic#trainActionSets}, WL
 * features of the state's graph with the action set as extra nodes). A node whose set holds one
 * action becomes the successor state, which enters the queue with its full applicable set. Poor
 * schemas or argument values are thus pruned before their ground successors are generated, which
 * pays off when the branching factor is high (many objects per schema).
 *
 * <p>This implementation computes the applicable ground actions of an expanded state to
 * enumerate the partial nodes (a lifted planner would match preconditions instead); the measured
 * cost is the number of generated (heuristically evaluated) nodes.
 */
public final class PartialSpaceSearch {

    private PartialSpaceSearch() {
    }

    /**
     * The result of a search.
     *
     * @param plan the plan (empty if none)
     * @param found whether a goal was reached
     * @param cost the plan cost
     * @param expanded nodes popped and expanded
     * @param generated nodes generated and evaluated
     * @param millis wall time
     */
    public record Result(List<StripsTask.Action> plan, boolean found, double cost, int expanded, int generated,
            double millis) {
    }

    /**
     * Splits the actions consistent with {@code star}'s prefix of length {@code level} (level -1:
     * nothing fixed) by their next decision (the schema at level -1, else argument
     * {@code level}); the group containing {@code star} comes first.
     *
     * @param actions applicable actions
     * @param star the reference action
     * @param level the prefix length, -1 for the schema decision
     * @return the groups
     */
    public static List<List<StripsTask.Action>> split(List<StripsTask.Action> actions, StripsTask.Action star,
            int level) {
        Map<Integer, List<StripsTask.Action>> groups = new LinkedHashMap<>();
        for (StripsTask.Action a : actions) {
            if (level >= 0) {
                if (a.schema() != star.schema()) {
                    continue;
                }
                boolean ok = true;
                for (int i = 0; i < level; i++) {
                    if (a.args()[i] != star.args()[i]) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) {
                    continue;
                }
            }
            int key = level < 0 ? a.schema() : a.args()[level];
            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(a);
        }
        List<List<StripsTask.Action>> out = new ArrayList<>();
        int starKey = level < 0 ? star.schema() : star.args()[level];
        if (groups.containsKey(starKey)) {
            out.add(groups.remove(starKey));
        }
        out.addAll(groups.values());
        return out;
    }

    private record Node(int state, int level, List<StripsTask.Action> actions) {
    }

    private record Entry(double key, long order, Node node) {
    }

    /**
     * Runs greedy best-first search in partial space.
     *
     * @param task the task
     * @param h the action-set ranker
     * @param maxExpansions the expansion budget
     * @return the result
     */
    public static Result search(StripsTask task, RankingHeuristic h, int maxExpansions) {
        long t0 = System.nanoTime();
        List<BitSet> states = new ArrayList<>();
        List<Integer> parent = new ArrayList<>();
        List<StripsTask.Action> via = new ArrayList<>();
        Map<BitSet, Integer> index = new HashMap<>();
        BitSet s0 = task.initialState();
        states.add(s0);
        parent.add(-1);
        via.add(null);
        index.put(s0, 0);
        PriorityQueue<Entry> open = new PriorityQueue<>((a, b) -> a.key() != b.key() ? Double.compare(a.key(), b.key())
            : Long.compare(a.order(), b.order()));
        long order = 0;
        List<StripsTask.Action> app0 = task.applicable(s0);
        open.add(new Entry(h.score(task, s0, app0), order++, new Node(0, -1, app0)));
        int generated = 1;
        int expanded = 0;
        while (!open.isEmpty() && expanded < maxExpansions) {
            Node n = open.poll().node();
            expanded++;
            BitSet s = states.get(n.state());
            if (n.level() == -1 && task.isGoal(s)) {
                List<StripsTask.Action> plan = new ArrayList<>();
                for (int c = n.state(); parent.get(c) >= 0; c = parent.get(c)) {
                    plan.add(via.get(c));
                }
                Collections.reverse(plan);
                return new Result(plan, true, task.validate(plan), expanded, generated, (System.nanoTime() - t0) / 1e6);
            }
            if (n.actions().isEmpty()) {
                continue;
            }
            StripsTask.Action ref = n.actions().get(0);
            List<List<StripsTask.Action>> groups = split(n.actions(), ref, n.level());
            for (List<StripsTask.Action> g : groups) {
                if (g.size() == 1) {
                    StripsTask.Action a = g.get(0);
                    BitSet t = a.apply(s);
                    if (index.containsKey(t)) {
                        continue;
                    }
                    int id = states.size();
                    states.add(t);
                    parent.add(n.state());
                    via.add(a);
                    index.put(t, id);
                    List<StripsTask.Action> app = task.applicable(t);
                    double key = task.isGoal(t) ? Double.NEGATIVE_INFINITY : h.score(task, t, app);
                    open.add(new Entry(key, order++, new Node(id, -1, app)));
                } else {
                    open.add(new Entry(h.score(task, s, g), order++, new Node(n.state(), n.level() + 1, g)));
                }
                generated++;
            }
        }
        return new Result(List.of(), false, Double.POSITIVE_INFINITY, expanded, generated,
            (System.nanoTime() - t0) / 1e6);
    }
}
