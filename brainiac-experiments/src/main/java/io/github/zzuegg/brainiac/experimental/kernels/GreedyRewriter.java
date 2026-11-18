package io.github.zzuegg.brainiac.experimental.kernels;

import java.util.ArrayList;
import java.util.List;

/**
 * The baseline to equality saturation: destructive rewriting in phases, as hand-written
 * fusion passes do. Each phase applies one rule (with constant folding) to the current graph,
 * keeps the result only if {@link TensorExpressionGraph#cost} drops, and then forgets the
 * alternatives. Phases run in a fixed order, once or until nothing improves, so a rule that
 * becomes applicable only after a later rule has fired is missed by a single pass. This is the
 * phase-ordering problem.
 */
public final class GreedyRewriter {

    private final List<EGraph.Rule> phases;
    private final boolean untilFixpoint;
    private final List<String> fired = new ArrayList<>();

    /**
     * @param phases the rules, in order
     * @param untilFixpoint repeat the sequence until no phase improves the cost
     */
    public GreedyRewriter(List<EGraph.Rule> phases, boolean untilFixpoint) {
        this.phases = List.copyOf(phases);
        this.untilFixpoint = untilFixpoint;
    }

    /**
     * @param g the graph
     * @return the rewritten graph
     */
    public TensorExpressionGraph rewrite(TensorExpressionGraph g) {
        TensorExpressionGraph current = g;
        boolean improved = true;
        for (int round = 0; round < 10 && improved; round++) {
            improved = false;
            for (EGraph.Rule rule : phases) {
                EGraph e = EGraph.of(current);
                e.saturate(List.of(rule, EGraph.constantFolding()), 1, 100_000);
                TensorExpressionGraph next = e.extract();
                if (next.cost() < current.cost() - 1e-9) {
                    current = next;
                    fired.add(rule.name());
                    improved = true;
                }
            }
            if (!untilFixpoint) {
                break;
            }
        }
        return current;
    }

    /** @return the phases that improved the graph, in order */
    public List<String> fired() {
        return List.copyOf(fired);
    }

    @Override
    public String toString() {
        return "GreedyRewriter(" + phases.size() + " phases" + (untilFixpoint ? ", to fixpoint" : "") + ")";
    }
}
