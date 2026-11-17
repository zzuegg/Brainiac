package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertTrue;

class TalkingDigitsTest {

    @Test
    void recognizesSpokenDigitStrings() {
        TalkingDigits td = TalkingDigits.of(32, 1);
        SplittableRandom r = new SplittableRandom(2);
        for (int step = 0; step < 300; step++) {
            List<int[]> digits = new ArrayList<>();
            List<float[]> waves = TalkingDigits.utterances(16, 3, r, digits);
            td.trainStep(waves, digits);
        }
        List<int[]> digits = new ArrayList<>();
        List<float[]> waves = TalkingDigits.utterances(60, 3, new SplittableRandom(99), digits);
        int[][] heard = td.recognize(waves);
        int correct = 0;
        for (int i = 0; i < heard.length; i++) {
            correct += Arrays.equals(heard[i], digits.get(i)) ? 1 : 0;
        }
        double accuracy = correct / 60.0;
        System.out.printf("talking digits: whole strings recognized %.3f%n", accuracy);
        assertTrue(accuracy > 0.7, "whole strings recognized: " + accuracy);
    }
}
