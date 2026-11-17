package io.github.zzuegg.brainiac.experiments.generative;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class BlockDiffusionTextTest {

    @Test
    void blockDiffusionSitsBetweenAutoregressionAndMaskedDiffusion() {
        Map<String, BlockDiffusionText.Outcome> r = BlockDiffusionText.run(300, 1);
        StringBuilder line = new StringBuilder("Grammar sentences, perplexity/tokens per step/valid:");
        r.forEach((k, v) -> line.append(String.format(" %s %.2f/%.2f/%.2f;", k, v.perplexity(), v.tokensPerStep(),
                v.valid())));
        System.out.println(line);
        BlockDiffusionText.Outcome ar = r.get("autoregressive (B=1)");
        BlockDiffusionText.Outcome bd3 = r.get("BD3-LM (B=4)");
        BlockDiffusionText.Outcome mdlm = r.get("masked diffusion (B=16)");
        assertTrue(bd3.perplexity() < mdlm.perplexity(), "a tighter bound than full masked diffusion: " + line);
        assertTrue(bd3.tokensPerStep() > 1.5 * ar.tokensPerStep(), "several tokens per pass: " + line);
        assertTrue(bd3.valid() > 0.5, line.toString());
    }
}
