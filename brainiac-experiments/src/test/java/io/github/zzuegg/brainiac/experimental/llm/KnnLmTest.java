package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Embedding;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KnnLmTest {

    static final int V = 12;

    static Model lm() {
        return Model.sequential().input(6).add(Embedding.of(V, 16)).add(TransformerBlock.of(2).causal(true).rotary(
                10000)).layer(LayerNorm.of(16)).dense(V, Activation.SOFTMAX).seed(3).build();
    }

    static int[] text(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = r.nextInt(V);
        }
        return t;
    }

    @Test
    void retrievalAddsKnowledgeWithoutTraining(@TempDir Path dir) throws Exception {
        Model model = lm();
        int[] doc = text(40, 1);
        KnnLm knn = KnnLm.create(16, V).search(4, 1);
        knn.addText(model, doc);
        assertEquals(39, knn.size());
        double[] ppl = knn.perplexity(model, doc, 0.7f);
        assertTrue(ppl[0] > 6, "an untrained model is near uniform: " + ppl[0]);
        assertTrue(ppl[1] < 2, "with the document in the datastore: " + ppl[1]);
        // generation follows the stored text from a prompt taken out of it
        int[] gen = knn.generate(model, Arrays.copyOfRange(doc, 10, 16), 8, 0.9f);
        assertArrayEquals(Arrays.copyOfRange(doc, 16, 24), Arrays.copyOfRange(gen, 6, 14));
        // the datastore persists
        knn.save(dir.resolve("knn.bin"));
        KnnLm loaded = KnnLm.load(dir.resolve("knn.bin"), V).search(4, 1);
        assertEquals(ppl[1], loaded.perplexity(model, doc, 0.7f)[1], 1e-6);
        // an unrelated text is not helped much: the mixture only costs the lambda share
        double[] other = knn.perplexity(model, text(40, 2), 0.2f);
        assertTrue(other[1] < other[0] * 1.3, Arrays.toString(other));
    }
}
