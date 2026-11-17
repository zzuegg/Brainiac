package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;

import org.junit.jupiter.api.Test;

class PeftShootoutTest {

    private static HookedDecoder base;

    static synchronized HookedDecoder base() {
        if (base == null) {
            base = CountingTasks.pretrained(150, 3);
        }
        return base;
    }

    private static PeftShootout.Result adapts(PeftShootout.Method m) {
        PeftShootout.Result r = PeftShootout.run(base(), m, 40, 9);
        assertTrue(r.after() < 0.75f * r.before(), m + ": " + r.before() + " -> " + r.after());
        return r;
    }

    private static long loraValues() {
        return PeftShootout.run(base(), PeftShootout.Method.LORA, 1, 9).trainable();
    }

    @Test
    void lowRankMethodsAdaptWithFewValues() {
        long full = adapts(PeftShootout.Method.FULL).trainable();
        PeftShootout.Result lora = adapts(PeftShootout.Method.LORA);
        assertTrue(lora.trainable() < full / 2);
    }

    @Test
    void loraVariantsAdaptToo() {
        PeftShootout.Result plus = adapts(PeftShootout.Method.LORA_PLUS);
        PeftShootout.Result pissa = adapts(PeftShootout.Method.PISSA);
        assertTrue(plus.after() < 0.5f && pissa.after() < 0.5f);
    }

    @Test
    void vectorMethodsAdaptWithFewerValuesStill() {
        long lora = loraValues();
        assertTrue(adapts(PeftShootout.Method.VERA).trainable() < lora);
        assertTrue(adapts(PeftShootout.Method.IA3).trainable() < lora);
        adapts(PeftShootout.Method.BITFIT);
        adapts(PeftShootout.Method.HOULSBY);
    }

    @Test
    void promptMethodsAdaptThroughTheInputAlone() {
        long lora = loraValues();
        assertTrue(adapts(PeftShootout.Method.PROMPT_TUNING).trainable() < lora);
        adapts(PeftShootout.Method.P_TUNING);
        adapts(PeftShootout.Method.PREFIX_TUNING);
    }
}
