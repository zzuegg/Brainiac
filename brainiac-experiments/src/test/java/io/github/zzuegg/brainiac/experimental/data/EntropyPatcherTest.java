package io.github.zzuegg.brainiac.experimental.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class EntropyPatcherTest {

    private static String corpus(long seed, int sentences) {
        String[] subjects = {"the cat", "a dog", "the bird", "my friend"};
        String[] verbs = {"sat on", "looked at", "jumped over", "walked to"};
        String[] objects = {"the mat", "a log", "the house", "my garden"};
        SplittableRandom r = new SplittableRandom(seed);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < sentences; i++) {
            sb.append(subjects[r.nextInt(4)]).append(' ').append(verbs[r.nextInt(4)]).append(' ')
                    .append(objects[r.nextInt(4)]).append(". ");
        }
        return sb.toString();
    }

    @Test
    void entropyIsHighWhereTheTextIsUnpredictable() {
        ByteNgramModel model = ByteNgramModel.train(corpus(1, 400), 4);
        byte[] text = corpus(2, 20).getBytes(StandardCharsets.UTF_8);
        double[] h = model.entropies(text);
        double afterPeriod = 0;
        double inside = 0;
        int a = 0;
        int b = 0;
        for (int i = 2; i < text.length; i++) {
            if (text[i - 2] == '.' && text[i - 1] == ' ') {
                afterPeriod += h[i];
                a++;
            } else if (Character.isLetter(text[i - 1]) && Character.isLetter(text[i])) {
                inside += h[i];
                b++;
            }
        }
        assertTrue(afterPeriod / a > 1.0 && inside / b < 0.5, "start of sentence " + afterPeriod / a
                + " bits, inside words " + inside / b + " bits");
        double[] p = model.distribution(text, 10);
        double sum = 0;
        for (double q : p) {
            sum += q;
        }
        assertEquals(1, sum, 1e-9);
    }

    @Test
    void patchesStartAtUnpredictableWordsAndCompressTheText() {
        ByteNgramModel model = ByteNgramModel.train(corpus(3, 400), 4);
        String text = corpus(4, 10);
        for (EntropyPatcher.Rule rule : EntropyPatcher.Rule.values()) {
            EntropyPatcher patcher = new EntropyPatcher(model, rule, 0.8, 16);
            List<String> patches = patcher.patches(text);
            assertEquals(text, String.join("", patches));
            double meanLength = text.length() / (double) patches.size();
            assertTrue(meanLength > 4, rule + ": mean patch length " + meanLength + " " + patches);
            int atWordStart = 0;
            for (String p : patches) {
                char c = p.charAt(0);
                atWordStart += c != ' ' && c != '.' ? 1 : 0;
            }
            assertTrue(atWordStart >= patches.size() * 0.8, rule + ": " + patches);
            int[] ids = patcher.patchIds(text.getBytes(StandardCharsets.UTF_8));
            assertEquals(patches.size() - 1, ids[ids.length - 1]);
        }
    }
}
