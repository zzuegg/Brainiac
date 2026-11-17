package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.data.BytePairTokenizer;
import io.github.zzuegg.brainiac.data.Tokenizer;
import io.github.zzuegg.brainiac.experimental.data.TokenizerGenome;
import io.github.zzuegg.brainiac.experimental.data.TokenizerScores;
import io.github.zzuegg.brainiac.experimental.evolve.TokenizerSearch;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * X19, evolved tokenizers: vocabulary size and merges are evolved against the held-out
 * bits per character of a small (bigram) language model, and compared with the raw byte
 * level and with greedy byte-pair encoding at several hand-picked sizes.
 *
 * <p>The text comes from a toy grammar over a Zipf-distributed vocabulary, so frequent
 * words, shared stems and suffixes give merges something to find.
 */
public final class EvolvedTokenizers {

    private static final String[] STEMS = {"walk", "talk", "jump", "play", "look", "work", "call", "turn", "help",
        "move", "open", "start", "learn", "paint", "climb"};
    private static final String[] SUFFIXES = {"", "s", "ed", "ing", "er"};
    private static final String[] NOUNS = {"cat", "dog", "bird", "child", "robot", "teacher", "farmer", "river",
        "mountain", "garden", "window", "letter"};
    private static final String[] SMALL = {"the", "a", "and", "then", "with", "near", "over"};

    /**
     * The outcome.
     *
     * @param evolvedBits held-out bits per character of the evolved tokenizer
     * @param evolvedVocabulary its vocabulary size
     * @param byteBits bits per character with raw bytes
     * @param greedyBits bits per character of greedy BPE, by vocabulary size
     */
    public record Outcome(double evolvedBits, int evolvedVocabulary, double byteBits, Map<Integer, Double> greedyBits) {
    }

    private EvolvedTokenizers() {
    }

    /**
     * Generates text from the toy grammar.
     *
     * @param sentences the number of sentences
     * @param seed the seed
     * @return the text
     */
    public static String corpus(int sentences, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        StringBuilder sb = new StringBuilder();
        for (int s = 0; s < sentences; s++) {
            sb.append(zipf(SMALL, r)).append(' ').append(zipf(NOUNS, r)).append(' ')
                    .append(zipf(STEMS, r)).append(SUFFIXES[r.nextInt(SUFFIXES.length)]).append(' ')
                    .append(zipf(SMALL, r)).append(' ').append(zipf(NOUNS, r)).append(r.nextInt(4) == 0 ? "s" : "")
                    .append(". ");
        }
        return sb.toString();
    }

    private static String zipf(String[] words, SplittableRandom r) {
        double total = 0;
        for (int i = 0; i < words.length; i++) {
            total += 1.0 / (i + 1);
        }
        double u = r.nextDouble() * total;
        for (int i = 0; i < words.length; i++) {
            u -= 1.0 / (i + 1);
            if (u <= 0) {
                return words[i];
            }
        }
        return words[words.length - 1];
    }

    /**
     * Runs the comparison.
     *
     * @param generations generations of the search
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int generations, long seed) {
        String train = corpus(400, seed);
        String valid = corpus(150, seed + 1);
        String test = corpus(150, seed + 2);
        // the search sees validation text; the final numbers are on unseen test text
        TokenizerSearch search = TokenizerSearch.of(train,
                t -> TokenizerScores.bigramBitsPerCharacter(t, train, valid)).population(12, 2, 150).seed(seed);
        TokenizerSearch.Scored best = search.run(generations);
        Tokenizer evolved = best.genome().toTokenizer();
        Map<Integer, Double> greedy = new LinkedHashMap<>();
        for (int merges : List.of(16, 48, 100, 150)) {
            greedy.put(256 + merges,
                    TokenizerScores.bigramBitsPerCharacter(BytePairTokenizer.train(train, 256 + merges), train, test));
        }
        Tokenizer bytes = new TokenizerGenome(List.of(), 0).toTokenizer();
        return new Outcome(TokenizerScores.bigramBitsPerCharacter(evolved, train, test), evolved.size(),
                TokenizerScores.bigramBitsPerCharacter(bytes, train, test), greedy);
    }
}
