package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;

class EmbeddingSearchTest {

    @Test
    void ivfPqTradesRecallForBytes() {
        Map<String, double[]> r = EmbeddingSearch.ann(8000, 32, 100, 1);
        StringBuilder line = new StringBuilder("ANN recall@10 / QPS / bytes per vector:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.0f/%.0f;", k, v[0], v[1], v[2])));
        System.out.println(line);
        assertTrue(r.get("IVF-PQ probe 16")[0] > 0.5, line.toString());
        assertTrue(r.get("IVF-PQ probe 16")[2] < r.get("flat (exact)")[2] / 10, line.toString());
        assertTrue(r.get("IVF-PQ anisotropic probe 16")[0] > r.get("IVF-PQ probe 16")[0], line.toString());
    }

    @Test
    void matryoshkaPrefixesAreGoodEmbeddings() {
        Map<String, double[]> r = EmbeddingSearch.matryoshka(500, 1);
        StringBuilder line = new StringBuilder("Matryoshka recall@10 at d=4/8/16/32:");
        r.forEach((k, v) -> line.append(" ").append(k).append(" ").append(Arrays.toString(
                Arrays.stream(v).map(x -> Math.round(x * 1000) / 1000.0).toArray())));
        System.out.println(line);
        assertTrue(r.get("Matryoshka")[1] > r.get("plain truncated")[1] + 0.05, line.toString());
        assertTrue(r.get("adaptive Matryoshka (4 -> 32)")[0] > r.get("adaptive plain (4 -> 32)")[0], line.toString());
        assertTrue(r.get("Matryoshka")[3] > r.get("plain truncated")[3] - 0.05, line.toString());
    }
}
