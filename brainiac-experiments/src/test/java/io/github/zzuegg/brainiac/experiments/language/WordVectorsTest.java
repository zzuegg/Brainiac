package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.StaticWordEmbeddings;
import io.github.zzuegg.brainiac.experimental.llm.StaticWordEmbeddings.Kind;

import org.junit.jupiter.api.Test;

class WordVectorsTest {

    @Test
    void staticVectorsLearnThePlantedStructure() {
        SyntheticGrammar g = new SyntheticGrammar();
        StringBuilder line = new StringBuilder("Word vectors analogy/similarity/OOV/downstream:");
        double[][] r = new double[Kind.values().length][];
        for (Kind k : Kind.values()) {
            StaticWordEmbeddings e = WordVectors.train(k, g, 1);
            r[k.ordinal()] = new double[] {WordVectors.analogies(e, g), WordVectors.similarity(e, g),
                WordVectors.oov(e, g), WordVectors.downstream(e, g, 1)};
            line.append(String.format(" %s %.3f/%.3f/%.3f/%.3f;", k, r[k.ordinal()][0], r[k.ordinal()][1],
                    r[k.ordinal()][2], r[k.ordinal()][3]));
        }
        double random = WordVectors.downstream(null, g, 1);
        line.append(String.format(" random-init downstream %.3f", random));
        System.out.println(line);
        double[] sgns = r[Kind.SGNS.ordinal()];
        double[] fast = r[Kind.FASTTEXT.ordinal()];
        assertTrue(sgns[0] > 0.3, "SGNS solves planted analogies");
        assertTrue(sgns[1] > 0.2, "cosine tracks the gold similarity");
        assertTrue(fast[2] > sgns[2] + 0.2, "fastText's n-grams give unseen words useful vectors");
        assertTrue(sgns[3] > random + 0.2, "warm-starting the embedding transfers to unseen stems");
    }
}
