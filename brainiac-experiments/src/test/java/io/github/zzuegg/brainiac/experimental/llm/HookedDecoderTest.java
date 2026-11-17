package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.llm.LanguageModel;

import org.junit.jupiter.api.Test;

class HookedDecoderTest {

    @Test
    void learnsCountingAndStepsMatchFullForward() {
        HookedDecoder m = TinyLm.trained();
        float loss = TinyLm.meanLoss(m, 12, 99);
        assertTrue(loss < 1.0, "loss " + loss);
        int[] seq = {3, 5, 7, 9};
        LanguageModel.State s = m.newState(1);
        m.next(s, new int[][] {{3, 5}}, null);
        float[] stepped = m.next(s, new int[][] {{7, 9}}, null).toArray();
        assertArrayEquals(m.lastLogits(seq), stepped, 1e-4f);
        assertEquals(11, LogitMath.argmax(stepped));
    }

    @Test
    void documentMaskSeparatesPackedSequences() {
        HookedDecoder m = TinyLm.trained();
        int[][] packed = {{1, 2, 3, 7, 9, 11}};
        int[][] seg = {{0, 0, 0, 1, 1, 1}};
        int[][] pos = {{0, 1, 2, 0, 1, 2}};
        float[] joint = m.forward(packed, pos, seg, false).logits().value().slice(1, 5, 6).toArray();
        float[] alone = m.lastLogits(new int[] {7, 9, 11});
        assertArrayEquals(alone, joint, 1e-4f);
    }
}
