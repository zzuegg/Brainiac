package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.compress.HuffmanCoder;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class WeightSharingTest {

    @Test
    void huffmanRoundTripsAndBeatsFixedWidthOnSkewedSymbols() {
        SplittableRandom r = new SplittableRandom(1);
        int[] symbols = new int[2000];
        for (int i = 0; i < symbols.length; i++) {
            double u = r.nextDouble();
            symbols[i] = u < 0.6 ? 0 : u < 0.85 ? 1 : 2 + r.nextInt(14);
        }
        HuffmanCoder h = HuffmanCoder.fit(symbols, 16);
        assertArrayEquals(symbols, h.decode(h.encode(symbols), symbols.length));
        assertTrue(h.encodedBits(symbols) < 4L * symbols.length * 0.6, "skewed symbols compress: "
                + h.encodedBits(symbols));
    }

    @Test
    void deepCompressionAgainstRoundToNearestAndGptq() {
        List<CompactRow> rows = CompactModels.weightSharing(new int[] {3, 4}, 0.7, 40, 1);
        StringBuilder line = new StringBuilder("8.5.9 deep compression, digits acc @bits (x compression):");
        for (CompactRow r : rows) {
            line.append(String.format(" %s %.3f @%.0f (%.1fx);", r.method(), r.metric(), r.cost(), r.size()));
        }
        System.out.println(line);
        for (int i = 2; i < rows.size(); i += 5) {
            CompactRow dc = rows.get(i + 3);
            CompactRow huff = rows.get(i + 4);
            assertTrue(huff.cost() < dc.cost(), "Huffman coding saves bits: " + huff + " vs " + dc);
            assertTrue(huff.size() > 10, "deep compression compresses more than 10x: " + huff);
            assertTrue(huff.metric() > 0.4, "and still classifies: " + huff);
        }
    }
}
