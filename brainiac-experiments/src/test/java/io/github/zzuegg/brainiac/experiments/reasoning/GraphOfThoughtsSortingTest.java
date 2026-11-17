package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.GraphOfThoughts;
import java.util.List;
import org.junit.jupiter.api.Test;

class GraphOfThoughtsSortingTest {

    @Test
    void graphRecordsAggregationParentsAndCountsCalls() {
        GraphOfThoughts<Integer> g = GraphOfThoughts.of(x -> -Math.abs(x - 10));
        GraphOfThoughts.Thought<Integer> root = g.root(0);
        List<GraphOfThoughts.Thought<Integer>> parts = g.split(root, x -> List.of(3, 4));
        GraphOfThoughts.Thought<Integer> sum = g.aggregate(parts, 2, xs -> xs.stream().mapToInt(i -> i).sum());
        assertEquals(List.of(parts.get(0).id(), parts.get(1).id()), sum.parents());
        GraphOfThoughts.Thought<Integer> fixed = g.refine(sum, 5, 0, x -> x + 1);
        assertEquals(10, fixed.content());
        assertEquals(1 + 2 + 3, g.calls(), "split, two merges, three refinements");
    }

    @Test
    void graphOfThoughtsSortsBetterThanTreeOfThoughtsAtEqualCalls() {
        GraphOfThoughtsSorting.Result r = GraphOfThoughtsSorting.run(60, 32, 0.006, 24, 1);
        System.out.printf("sorting 32 digits: errors GoT %.2f vs ToT %.2f vs IO %.2f; exact GoT %.2f vs ToT %.2f vs "
                + "IO %.2f; calls GoT %.1f ToT %.1f%n", r.gotErrors(), r.totErrors(), r.ioErrors(), r.gotSolved(),
                r.totSolved(), r.ioSolved(), r.gotCalls(), r.totCalls());
        assertTrue(r.gotCalls() <= r.totCalls() + 0.5, r.toString());
        assertTrue(r.gotErrors() < r.totErrors() * 0.5, r.toString());
        assertTrue(r.gotSolved() > r.totSolved() + 0.2, r.toString());
        assertTrue(r.totErrors() < r.ioErrors(), r.toString());
    }
}
