package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SyntheticCorpus.Kind;
import org.junit.jupiter.api.Test;

import java.util.Map;

class RetrievalShootoutTest {

    static String table(String title, Map<String, Map<Kind, RetrievalShootout.Metrics>> r) {
        StringBuilder s = new StringBuilder(title + " R@1/R@10/MRR@10 lexical | paraphrase | multi-aspect:");
        r.forEach((name, m) -> {
            s.append("\n  ").append(String.format("%-28s", name));
            for (Kind k : Kind.values()) {
                RetrievalShootout.Metrics x = m.get(k);
                s.append(String.format(" %.2f/%.2f/%.2f |", x.recall1(), x.recall10(), x.mrr10()));
            }
        });
        return s.toString();
    }

    @Test
    void bm25WinsOnRareTermsAndDenseOnParaphrases() {
        Map<String, Map<Kind, RetrievalShootout.Metrics>> r = new RetrievalShootout(1).lexicalVersusDense(300);
        System.out.println(table("BM25 vs DPR", r));
        assertTrue(r.get("BM25").get(Kind.LEXICAL).mrr10() > r.get("DPR").get(Kind.LEXICAL).mrr10(), r.toString());
        assertTrue(r.get("DPR").get(Kind.PARAPHRASE).mrr10() > r.get("BM25").get(Kind.PARAPHRASE).mrr10() + 0.2,
                r.toString());
        assertTrue(r.get("DPR").get(Kind.PARAPHRASE).mrr10()
                > r.get("untrained encoder").get(Kind.PARAPHRASE).mrr10() + 0.2, r.toString());
    }
}
