package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.experimental.data.FimTransform;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FillInTheMiddleTest {

    @Test
    void transformsDocuments() {
        int[] doc = {1, 2, 3, 4, 5};
        assertArrayEquals(new int[] {10, 1, 11, 4, 5, 12, 2, 3},
                FimTransform.of(10, 11, 12, FimTransform.Mode.PSM).apply(doc, 1, 3));
        assertArrayEquals(new int[] {10, 11, 4, 5, 12, 1, 2, 3},
                FimTransform.of(10, 11, 12, FimTransform.Mode.SPM).apply(doc, 1, 3));
    }

    @Test
    void fimIsFree() {
        FillInTheMiddle e = FillInTheMiddle.create();
        HookedDecoder plain = e.train(0, 900, 1);
        HookedDecoder fim = e.train(0.5, 900, 1);
        FillInTheMiddle.Score p = e.evaluate(plain, false, 7);
        FillInTheMiddle.Score f = e.evaluate(fim, true, 7);
        System.out.printf("fill-in-the-middle: infill token accuracy %.3f (exact %.3f) vs plain causal LM %.3f "
                + "(exact %.3f); left-to-right loss %.3f vs %.3f%n", f.infillTokenAccuracy(), f.infillExact(),
                p.infillTokenAccuracy(), p.infillExact(), f.leftToRightLoss(), p.leftToRightLoss());
        assertTrue(f.infillTokenAccuracy() > 0.8, "the FIM model infills from the suffix");
        assertTrue(p.infillTokenAccuracy() < 0.3, "a plain model cannot");
        assertTrue(f.leftToRightLoss() < p.leftToRightLoss() + 0.15, "FIM costs little left-to-right");
    }
}
