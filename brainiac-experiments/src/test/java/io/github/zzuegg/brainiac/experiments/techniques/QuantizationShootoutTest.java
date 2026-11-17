package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class QuantizationShootoutTest {

    @Test
    void hessianAwareAndLearnedQuantizationBeatRoundToNearest() {
        Map<String, Float> acc = QuantizationShootout.run(2, 1);
        System.out.println(acc);
        float rtn = acc.get("rtn");
        assertTrue(acc.get("float") > 0.8f, "the float model learned: " + acc);
        assertTrue(acc.get("gptq") > rtn, "GPTQ beats RTN at 2 bits: " + acc);
        assertTrue(acc.get("lsq-qat") > rtn, "LSQ beats RTN at 2 bits: " + acc);
        assertTrue(acc.get("fp8-e4m3") > acc.get("float") - 0.03f, "FP8 E4M3 is nearly lossless: " + acc);
    }
}
