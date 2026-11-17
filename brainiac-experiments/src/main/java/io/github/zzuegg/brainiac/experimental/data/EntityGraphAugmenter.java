package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.IntPredicate;
import java.util.random.RandomGenerator;

/**
 * EntiGraph synthetic continued pretraining (Yang et al. 2024, "Synthetic continued
 * pretraining", <a href="https://arxiv.org/abs/2409.07431">arXiv:2409.07431</a>, ICLR 2025):
 * a model learns little from a small corpus read many times, because each fact appears in
 * one form only. EntiGraph rewrites the corpus into a much larger, more diverse one: it
 * extracts the entities of every document, then for each pair (and optionally each triple)
 * of entities in the same document asks a generator to write new text about how they relate,
 * given the source document. The synthetic text states the same knowledge in many forms
 * (both directions, combined with other facts), so continued pretraining on it makes facts
 * retrievable from questions phrased in ways the source never used.
 *
 * <p>The entity extractor is a predicate over token ids here, and the generator a
 * {@link PairWriter} (an LLM prompted with the document and the pair in the paper; any
 * function in a toy).
 */
public final class EntityGraphAugmenter {

    /** Writes text about the relation between entities, from a source document. */
    @FunctionalInterface
    public interface PairWriter {

        /**
         * @param document the source document
         * @param entities two (or three) entities that occur in it
         * @param random the source of randomness
         * @return a synthetic document about them, empty if there is nothing to say
         */
        int[] write(int[] document, int[] entities, RandomGenerator random);
    }

    private final IntPredicate isEntity;
    private final PairWriter writer;

    /**
     * @param isEntity which tokens are entities
     * @param writer the generator of synthetic text
     */
    public EntityGraphAugmenter(IntPredicate isEntity, PairWriter writer) {
        this.isEntity = isEntity;
        this.writer = writer;
    }

    /**
     * @param document a document
     * @return its distinct entities, in order of appearance
     */
    public int[] entities(int[] document) {
        Set<Integer> seen = new LinkedHashSet<>();
        for (int t : document) {
            if (isEntity.test(t)) {
                seen.add(t);
            }
        }
        return seen.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Builds the synthetic corpus.
     *
     * @param corpus the source documents
     * @param samples generations per entity pair (ordered both ways) or triple
     * @param triples whether to also write about triples of entities
     * @param random the source of randomness
     * @return the synthetic documents
     */
    public List<int[]> augment(List<int[]> corpus, int samples, boolean triples, RandomGenerator random) {
        List<int[]> out = new ArrayList<>();
        for (int[] doc : corpus) {
            int[] e = entities(doc);
            for (int a = 0; a < e.length; a++) {
                for (int b = 0; b < e.length; b++) {
                    if (a == b) {
                        continue;
                    }
                    for (int s = 0; s < samples; s++) {
                        add(out, writer.write(doc, new int[] {e[a], e[b]}, random));
                    }
                    if (!triples) {
                        continue;
                    }
                    for (int c = 0; c < e.length; c++) {
                        if (c != a && c != b) {
                            for (int s = 0; s < samples; s++) {
                                add(out, writer.write(doc, new int[] {e[a], e[b], e[c]}, random));
                            }
                        }
                    }
                }
            }
        }
        return out;
    }

    private static void add(List<int[]> out, int[] doc) {
        if (doc != null && doc.length > 0) {
            out.add(doc);
        }
    }
}
