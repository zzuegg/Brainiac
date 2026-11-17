package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.experimental.nn.HdcClassifier;
import io.github.zzuegg.brainiac.experimental.nn.HdcItemMemory;
import io.github.zzuegg.brainiac.experimental.tensor.Hypervector;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Two classic hyperdimensional-computing demos on bit-packed {@code Hypervector}s.
 *
 * <ul>
 * <li>{@link #dollarOfMexico}: Kanerva's "What is the dollar of Mexico?". Countries are
 *     records {@code NAME*usa + CAPITAL*washington + CURRENCY*dollar} (bind = XOR, + = bundle);
 *     {@code mexico_record * usa_record * dollar} unbinds to a noisy {@code peso}, which the
 *     item memory cleans up. Analogy by algebra, no training.</li>
 * <li>{@link #languageId}: texts are bundles of letter trigrams, each trigram the binding of
 *     permuted letter vectors (position by rotation). With one training text per language,
 *     an {@code HdcClassifier} recognizes the language of new texts.</li>
 * </ul>
 */
public final class HyperdimensionalDemo {
    private HyperdimensionalDemo() {
    }

    /**
     * Answers "what is the dollar of Mexico?" by unbinding records.
     *
     * @param dims the vector size
     * @param seed the seed
     * @return the symbol the item memory returns, which should be {@code "peso"}
     */
    public static String dollarOfMexico(int dims, long seed) {
        HdcItemMemory<String> memory = new HdcItemMemory<>(dims, seed);
        Hypervector name = memory.get("NAME");
        Hypervector capital = memory.get("CAPITAL");
        Hypervector currency = memory.get("CURRENCY");
        Hypervector usa = Hypervector.bundle(List.of(name.bind(memory.get("usa")),
                capital.bind(memory.get("washington")), currency.bind(memory.get("dollar"))));
        Hypervector mexico = Hypervector.bundle(List.of(name.bind(memory.get("mexico")),
                capital.bind(memory.get("mexico city")), currency.bind(memory.get("peso"))));
        Hypervector answer = mexico.bind(usa).bind(memory.get("dollar"));
        return memory.cleanup(answer);
    }

    /**
     * Encodes a text as the bundle of its letter trigrams.
     *
     * @param text the text
     * @param letters the letter memory
     * @return the text vector
     */
    public static Hypervector encode(String text, HdcItemMemory<Character> letters) {
        Hypervector.Accumulator acc = new Hypervector.Accumulator(letters.dims());
        for (int i = 0; i + 2 < text.length(); i++) {
            Hypervector g = letters.get(text.charAt(i)).permute(2)
                    .bind(letters.get(text.charAt(i + 1)).permute(1))
                    .bind(letters.get(text.charAt(i + 2)));
            acc.add(g, 1);
        }
        return acc.toVector();
    }

    /**
     * Synthetic "languages": each a random Markov chain over the same letters, with strong
     * preferences for which letter follows which.
     *
     * @param languages how many languages
     * @param seed the seed
     * @return {@code [language][from][to]} transition probabilities
     */
    public static double[][][] languages(int languages, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = 12;
        double[][][] t = new double[languages][n][n];
        for (int l = 0; l < languages; l++) {
            for (int a = 0; a < n; a++) {
                double sum = 0;
                for (int b = 0; b < n; b++) {
                    t[l][a][b] = Math.pow(r.nextDouble(), 4);
                    sum += t[l][a][b];
                }
                for (int b = 0; b < n; b++) {
                    t[l][a][b] /= sum;
                }
            }
        }
        return t;
    }

    /**
     * Writes a text in a language.
     *
     * @param chain the language's transitions
     * @param length the text length
     * @param random the random source
     * @return the text
     */
    public static String text(double[][] chain, int length, SplittableRandom random) {
        StringBuilder sb = new StringBuilder();
        int c = random.nextInt(chain.length);
        for (int i = 0; i < length; i++) {
            sb.append((char) ('a' + c));
            double u = random.nextDouble();
            int next = chain.length - 1;
            for (int b = 0; b < chain.length; b++) {
                u -= chain[c][b];
                if (u <= 0) {
                    next = b;
                    break;
                }
            }
            c = next;
        }
        return sb.toString();
    }

    /**
     * One-shot language identification: train on one text per language, test on new ones.
     *
     * @param languages how many languages
     * @param trainLength the length of the single training text
     * @param tests test texts per language
     * @param testLength their length
     * @param dims the vector size
     * @param seed the seed
     * @return the test accuracy
     */
    public static double languageId(int languages, int trainLength, int tests, int testLength, int dims, long seed) {
        double[][][] chains = languages(languages, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        HdcItemMemory<Character> letters = new HdcItemMemory<>(dims, seed + 2);
        HdcClassifier classifier = new HdcClassifier(languages, dims);
        for (int l = 0; l < languages; l++) {
            classifier.add(encode(text(chains[l], trainLength, r), letters), l);
        }
        List<Integer> right = new ArrayList<>();
        for (int l = 0; l < languages; l++) {
            for (int i = 0; i < tests; i++) {
                right.add(classifier.predict(encode(text(chains[l], testLength, r), letters)) == l ? 1 : 0);
            }
        }
        return right.stream().mapToInt(Integer::intValue).average().orElse(0);
    }
}
