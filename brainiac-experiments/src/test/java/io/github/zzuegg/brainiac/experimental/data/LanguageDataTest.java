package io.github.zzuegg.brainiac.experimental.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.BytePairTokenizer;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class LanguageDataTest {

    private static final String TEXT;

    static {
        String[] subjects = {"the cat", "the dog", "a bird", "my friend", "the teacher"};
        String[] verbs = {"sat on", "looked at", "ran to", "talked about"};
        String[] objects = {"the mat", "the house", "the park", "the end of the road"};
        StringBuilder b = new StringBuilder();
        SplittableRandom r = new SplittableRandom(1);
        for (int i = 0; i < 300; i++) {
            b.append(subjects[r.nextInt(subjects.length)]).append(' ').append(verbs[r.nextInt(verbs.length)])
                    .append(' ').append(objects[r.nextInt(objects.length)]).append(" in order to rest. ");
        }
        TEXT = b.toString();
    }

    @Test
    void bpeDropoutVariesSegmentationButNotText() {
        BytePairTokenizer bpe = BytePairTokenizer.train(TEXT, 320);
        BpeDropoutTokenizer drop = new BpeDropoutTokenizer(bpe, 0.3, 1);
        String s = "the teacher talked about the end of the road.";
        assertArrayEquals(bpe.encode(s), drop.encode(s, 0, new SplittableRandom(0)));
        int[] a = drop.encode(s);
        int[] b = drop.encode(s);
        assertEquals(s, drop.decode(a));
        assertEquals(s, drop.decode(b));
        assertTrue(a.length > bpe.encode(s).length || b.length > bpe.encode(s).length, "dropout splits more");
        assertTrue(!Arrays.equals(a, b) || !Arrays.equals(a, drop.encode(s)), "segmentations vary");
    }

    @Test
    void superBpeSpendsTheSameVocabularyOnFewerTokens() {
        BytePairTokenizer bpe = BytePairTokenizer.train(TEXT, 360);
        SuperBpeTokenizer sup = SuperBpeTokenizer.train(TEXT, 320, 360);
        // plain BPE runs out of in-word merges on this small corpus; SuperBPE keeps going across words
        assertTrue(bpe.size() <= sup.size() && sup.size() <= 360, bpe.size() + " vs " + sup.size());
        String held = "my friend ran to the park in order to rest. the dog sat on the mat in order to rest.";
        assertEquals(held, sup.decode(sup.encode(held)));
        int plain = bpe.encode(held).length;
        int sw = sup.encode(held).length;
        assertTrue(sw < plain * 0.8, "superbpe " + sw + " vs bpe " + plain);
        assertTrue(sup.superwordTokens(held) > 0);
    }

    @Test
    void minHashFindsNearDuplicates() {
        List<String> docs = new ArrayList<>();
        docs.add("the quick brown fox jumps over the lazy dog near the river bank today");
        docs.add("the quick brown fox jumps over the lazy dog near the river bank");
        docs.add("completely different text about language models and their training data");
        docs.add("the quick brown fox jumped over the lazy dog near the river bank today");
        docs.add("another unrelated sentence on tokenizers, packing and data mixtures");
        MinHashDeduplicator d = new MinHashDeduplicator(5, 16, 4, 7);
        double est = MinHashDeduplicator.similarity(d.signature(docs.get(0)), d.signature(docs.get(3)));
        double exact = MinHashDeduplicator.jaccard(docs.get(0), docs.get(3), 5);
        assertEquals(exact, est, 0.15);
        List<String> kept = d.deduplicate(docs, 0.6);
        assertEquals(3, kept.size(), kept.toString());
        assertTrue(kept.contains(docs.get(0)) && kept.contains(docs.get(2)) && kept.contains(docs.get(4)));
    }

    @Test
    void packingWithDocumentMasksGivesTheUnpackedLoss() {
        SplittableRandom r = new SplittableRandom(4);
        List<int[]> docs = new ArrayList<>();
        for (int i = 0; i < 9; i++) {
            int[] d = new int[3 + r.nextInt(6)];
            for (int t = 0; t < d.length; t++) {
                d[t] = r.nextInt(10);
            }
            docs.add(d);
        }
        SequencePacker.Packed packed = SequencePacker.pack(docs, 16, 0);
        SequencePacker.Packed padded = SequencePacker.padded(docs, 16, 0);
        assertTrue(packed.efficiency() > padded.efficiency() + 0.3, packed.efficiency() + " vs " + padded.efficiency());
        HookedDecoder m = HookedDecoder.create(HookedDecoder.Settings.tiny(10), 2);
        float lp = HookedDecoder.loss(m.forward(packed.tokens(), packed.positions(), packed.segments(), false).logits(),
                packed.targets()).value().item();
        float lu = HookedDecoder.loss(m.forward(padded.tokens(), padded.positions(), padded.segments(), false).logits(),
                padded.targets()).value().item();
        assertEquals(lu, lp, 1e-4);
        HookedDecoder.Run naive = m.forward(packed.tokens());
        assertNotEquals(lu, HookedDecoder.loss(naive.logits(), packed.targets()).value().item(), 1e-4,
                "without masks, documents leak into each other");
    }

    @Test
    void mixturesScheduleAndReweight() {
        double[] w = DataMixture.temperature(new long[] {900, 100}, 1);
        assertEquals(0.9, w[0], 1e-9);
        double[] hot = DataMixture.temperature(new long[] {900, 100}, 3);
        assertTrue(hot[1] > 0.3);
        DataMixture s = DataMixture.schedule(new long[] {0, 100}, new double[][] {{1, 0}, {0, 1}});
        assertEquals(0.5, s.weights(50)[1], 1e-9);
        assertArrayEquals(new int[] {6, 2}, DataMixture.constant(3, 1).allocate(0, 8));
        int ones = 0;
        SplittableRandom r = new SplittableRandom(1);
        for (int i = 0; i < 2000; i++) {
            ones += s.sample(75, r);
        }
        assertEquals(0.75, ones / 2000.0, 0.04);
        DataMixture online = DataMixture.constant(1, 1);
        double[] after = online.updateWithExcessLoss(0, new double[] {2.0, 1.0}, new double[] {1.0, 1.0}, 1, 1e-3);
        assertTrue(after[0] > 0.7, "the domain with excess loss gains weight");
        SequenceLengthWarmup warm = new SequenceLengthWarmup(4, 32, 100, 4);
        assertEquals(4, warm.length(0));
        assertEquals(16, warm.length(45));
        assertEquals(32, warm.length(100));
        assertTrue(warm.warmupCost() < 0.6);
        assertEquals(4, warm.crop(new int[][] {new int[32]}, 0)[0].length);
    }
}
