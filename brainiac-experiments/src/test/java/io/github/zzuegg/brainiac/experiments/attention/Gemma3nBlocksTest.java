package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class Gemma3nBlocksTest {

    @Test
    void matformerSubModelsBeatExtractingFromAPlainModel() {
        List<Gemma3nBlocks.WidthRow> rows = Gemma3nBlocks.matformer(16, 64, 200, 1, new int[] {16});
        System.out.print(Gemma3nBlocks.widthTable(rows));
        Gemma3nBlocks.WidthRow quarter = rows.get(1);
        Gemma3nBlocks.WidthRow full = rows.get(3);
        System.out.printf("val loss at FFN width 16 of 64: MatFormer %.4f vs scratch %.4f vs truncated %.4f vs "
                + "magnitude-pruned %.4f; at 64: MatFormer %.4f vs plain %.4f%n", quarter.nested(), quarter.scratch(),
                quarter.truncated(), quarter.pruned(), full.nested(), full.truncated());
        assertTrue(quarter.nested() < quarter.truncated());
        assertTrue(quarter.nested() < quarter.pruned());
        assertTrue(quarter.nested() < quarter.scratch() + 0.15);
        assertTrue(full.nested() < full.truncated() + 0.15);
    }
}
