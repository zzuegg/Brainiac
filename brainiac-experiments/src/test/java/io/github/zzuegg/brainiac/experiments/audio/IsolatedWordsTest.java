package io.github.zzuegg.brainiac.experiments.audio;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class IsolatedWordsTest {

    @Test
    void dtwTemplatesBeatTheLearnedRecogniserWithFewExamples() {
        double[] one = SpokenDigitRecognition.isolatedWords(1, 20, 300, 1);
        double[] five = SpokenDigitRecognition.isolatedWords(5, 20, 300, 1);
        double[] noisy = SpokenDigitRecognition.isolatedWords(5, 5, 300, 1);
        System.out.printf("word accuracy (DTW+MFCC / MLP+CTC bands / MLP+CTC MFCC): 1 shot 20 dB %.2f/%.2f/%.2f, "
                + "5 shots 20 dB %.2f/%.2f/%.2f, 5 shots 5 dB %.2f/%.2f/%.2f%n", one[0], one[1], one[2], five[0],
                five[1], five[2], noisy[0], noisy[1], noisy[2]);
        assertTrue(one[0] > one[1], "one template per word beats the MLP trained on ten utterances");
        assertTrue(five[0] > 0.8, "DTW with five templates: " + five[0]);
    }
}
