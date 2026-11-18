package io.github.zzuegg.brainiac.experimental.symbolic;

import java.util.Arrays;
import java.util.BitSet;
import java.util.PriorityQueue;

/**
 * Delete-relaxation heuristics for {@link StripsTask}s (Bonet &amp; Geffner, "Planning as heuristic
 * search", AIJ 2001; Hoffmann &amp; Nebel, "The FF planning system", JAIR 2001). In the delete
 * relaxation actions never remove facts, so the cost of reaching each fact can be computed by a
 * generalised Dijkstra over facts and actions:
 *
 * <ul>
 *   <li>{@link Kind#H_MAX}: an action costs its own cost plus the costliest precondition; the
 *       goal costs its costliest fact. Admissible, but weak.</li>
 *   <li>{@link Kind#H_ADD}: preconditions and goals are summed. Informative but inadmissible,
 *       since it double-counts shared sub-plans.</li>
 *   <li>{@link Kind#H_FF}: a relaxed plan is extracted backwards from the goals along the h_add
 *       best supporters and its cost is the estimate; the relaxed plan's actions that are
 *       applicable in the state are the preferred operators (FF's helpful actions).</li>
 *   <li>{@link Kind#GOAL_COUNT} and {@link Kind#BLIND}: the unsatisfied goals, and 0 off the goal
 *       (the baselines).</li>
 * </ul>
 * Unreachable goals give infinity (a dead end).
 */
public final class RelaxationHeuristics implements StateSpaceSearch.Evaluator {

    /** The heuristic. */
    public enum Kind {
        /** 0 at goals, otherwise the cheapest action cost. */
        BLIND,
        /** The number of unsatisfied goal facts. */
        GOAL_COUNT,
        /** h_max. */
        H_MAX,
        /** h_add. */
        H_ADD,
        /** The FF relaxed-plan heuristic with preferred operators. */
        H_FF
    }

    private final StripsTask task;
    private final Kind kind;
    private final int[] goal;
    private final double minCost;

    /**
     * Creates the heuristic.
     *
     * @param task the task
     * @param kind which heuristic
     */
    public RelaxationHeuristics(StripsTask task, Kind kind) {
        this.task = task;
        this.kind = kind;
        this.goal = task.goal();
        double m = Double.POSITIVE_INFINITY;
        for (StripsTask.Action a : task.actions()) {
            m = Math.min(m, a.cost());
        }
        this.minCost = task.actions().isEmpty() ? 0 : m;
    }

    /** @return the kind */
    public Kind kind() {
        return kind;
    }

    @Override
    public StateSpaceSearch.Evaluation evaluate(BitSet state, Object data) {
        return evaluate(state);
    }

    /**
     * Evaluates a state.
     *
     * @param state the state
     * @return the estimate and (for h_FF) the preferred operators
     */
    public StateSpaceSearch.Evaluation evaluate(BitSet state) {
        switch (kind) {
            case BLIND:
                return StateSpaceSearch.Evaluation.of(task.isGoal(state) ? 0 : minCost);
            case GOAL_COUNT: {
                int c = 0;
                for (int g : goal) {
                    if (!state.get(g)) {
                        c++;
                    }
                }
                return StateSpaceSearch.Evaluation.of(c);
            }
            default:
                break;
        }
        double[] actionCost = new double[task.actions().size()];
        for (StripsTask.Action a : task.actions()) {
            actionCost[a.id()] = a.cost();
        }
        Exploration x = explore(task, state, actionCost, kind == Kind.H_MAX);
        double h = 0;
        for (int g : goal) {
            double c = x.factCost[g];
            if (Double.isInfinite(c)) {
                return StateSpaceSearch.Evaluation.of(Double.POSITIVE_INFINITY);
            }
            h = kind == Kind.H_MAX ? Math.max(h, c) : h + c;
        }
        if (kind != Kind.H_FF) {
            return StateSpaceSearch.Evaluation.of(h);
        }
        BitSet marked = new BitSet();
        BitSet seen = new BitSet();
        int[] stack = new int[task.facts()];
        int top = 0;
        for (int g : goal) {
            if (!state.get(g) && !seen.get(g)) {
                seen.set(g);
                stack[top++] = g;
            }
        }
        double ff = 0;
        while (top > 0) {
            int f = stack[--top];
            int a = x.supporter[f];
            if (a < 0 || marked.get(a)) {
                continue;
            }
            marked.set(a);
            ff += actionCost[a];
            for (int p : task.actions().get(a).pre()) {
                if (!state.get(p) && !seen.get(p)) {
                    seen.set(p);
                    stack[top++] = p;
                }
            }
        }
        int[] preferred = marked.stream().filter(a -> task.actions().get(a).applicable(state)).toArray();
        return new StateSpaceSearch.Evaluation(ff, preferred);
    }

    /**
     * The result of a relaxed exploration.
     *
     * @param factCost the h_max or h_add cost of every fact
     * @param supporter the best supporter of every fact, -1 for facts of the state or unreached
     * @param actionValue the cost of every action's precondition set (max or sum), infinite if
     *     never enabled
     */
    public record Exploration(double[] factCost, int[] supporter, double[] actionValue) {
    }

    /**
     * Generalised Dijkstra over the relaxed task.
     *
     * @param task the task
     * @param state the state
     * @param actionCost the action costs to use (LM-cut changes them)
     * @param max whether preconditions combine by maximum (h_max) or sum (h_add)
     * @return the exploration
     */
    public static Exploration explore(StripsTask task, BitSet state, double[] actionCost, boolean max) {
        int nf = task.facts();
        int na = task.actions().size();
        double[] cost = new double[nf];
        Arrays.fill(cost, Double.POSITIVE_INFINITY);
        int[] supporter = new int[nf];
        Arrays.fill(supporter, -1);
        int[] remaining = new int[na];
        double[] value = new double[na];
        Arrays.fill(value, Double.POSITIVE_INFINITY);
        PriorityQueue<double[]> pq = new PriorityQueue<>((a, b) -> Double.compare(a[0], b[0]));
        for (int f = state.nextSetBit(0); f >= 0; f = state.nextSetBit(f + 1)) {
            cost[f] = 0;
            pq.add(new double[] {0, f});
        }
        for (StripsTask.Action a : task.actions()) {
            remaining[a.id()] = a.pre().length;
            if (a.pre().length == 0) {
                value[a.id()] = 0;
                fire(a, actionCost[a.id()], cost, supporter, pq);
            } else {
                value[a.id()] = 0;
            }
        }
        boolean[] done = new boolean[nf];
        while (!pq.isEmpty()) {
            double[] e = pq.poll();
            int f = (int) e[1];
            if (done[f] || e[0] > cost[f]) {
                continue;
            }
            done[f] = true;
            for (int a : task.consumers(f)) {
                value[a] = max ? Math.max(value[a], cost[f]) : value[a] + cost[f];
                if (--remaining[a] == 0) {
                    fire(task.actions().get(a), value[a] + actionCost[a], cost, supporter, pq);
                }
            }
        }
        for (int a = 0; a < na; a++) {
            if (remaining[a] > 0) {
                value[a] = Double.POSITIVE_INFINITY;
            }
        }
        return new Exploration(cost, supporter, value);
    }

    private static void fire(StripsTask.Action a, double c, double[] cost, int[] supporter,
            PriorityQueue<double[]> pq) {
        for (int f : a.add()) {
            if (c < cost[f]) {
                cost[f] = c;
                supporter[f] = a.id();
                pq.add(new double[] {c, f});
            }
        }
    }
}
