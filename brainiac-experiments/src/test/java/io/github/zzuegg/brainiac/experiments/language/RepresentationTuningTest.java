package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;

import org.junit.jupiter.api.Test;

class RepresentationTuningTest {

    @Test
    void loreftAdaptsWithFewerValuesThanLora() {
        HookedDecoder base = PeftShootoutTest.base();
        RepresentationTuning.Result reft = RepresentationTuning.reft(base, 40, 9, 4);
        PeftShootout.Result lora = PeftShootout.run(base, PeftShootout.Method.LORA, 40, 9);
        PeftShootout.Result ia3 = PeftShootout.run(base, PeftShootout.Method.IA3, 40, 9);
        PeftShootout.Result bitfit = PeftShootout.run(base, PeftShootout.Method.BITFIT, 40, 9);
        System.out.printf("LoReFT rank 4: loss %.3f -> %.3f with %d values; LoRA %.3f with %d; IA3 %.3f with %d; "
                + "BitFit %.3f with %d%n", reft.before(), reft.after(), reft.trainable(), lora.after(),
                lora.trainable(), ia3.after(), ia3.trainable(), bitfit.after(), bitfit.trainable());
        assertTrue(reft.after() < 0.75f * reft.before());
        assertTrue(reft.trainable() < lora.trainable() / 4);
    }

    @Test
    void lisaTrainsEveryLayerWithLoraSizedOptimizerState() {
        HookedDecoder base = RepresentationTuning.deepBase(150, 5);
        RepresentationTuning.Result full = RepresentationTuning.lisa(base, "full", 0, 0, 40, 9);
        RepresentationTuning.Result lora = RepresentationTuning.lisa(base, "lora", 0, 0, 40, 9);
        RepresentationTuning.Result lisa = RepresentationTuning.lisa(base, "lisa", 1, 5, 40, 9);
        System.out.printf("LISA (1 of 4 blocks, period 5): loss %.3f -> %.3f, peak optimizer state %d; full %.3f, %d; "
                + "LoRA %.3f, %d%n", lisa.before(), lisa.after(), lisa.peakState(), full.after(), full.peakState(),
                lora.after(), lora.peakState());
        assertTrue(lisa.after() < 0.75f * lisa.before());
        assertTrue(lisa.peakState() < 0.6 * full.peakState());
        assertTrue(lisa.after() < 1.5f * full.after() + 0.05f);
    }
}
