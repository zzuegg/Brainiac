package io.github.zzuegg.brainiac.experiments.parsing;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ParsingShootoutConstituencyTest {

    @Test
    void chartParserYieldsTreesAndBeatsRightBranching() {
        Map<String, double[]> r = ParsingShootout.constituency(ParsingShootoutDependencyTest.BUDGET, 1);
        String line = ParsingShootoutDependencyTest.line("parsing constituency F1, exact, formed, PP, F1 long, ms", r);
        System.out.println(line);
        assertTrue(r.get("chart BiLSTM margin")[2] == 1.0, line);
        assertTrue(r.get("chart BiLSTM margin")[0] > r.get("right-branching")[0], line);
        assertTrue(r.get("treebank PCFG")[0] > r.get("right-branching")[0], line);
    }
}
