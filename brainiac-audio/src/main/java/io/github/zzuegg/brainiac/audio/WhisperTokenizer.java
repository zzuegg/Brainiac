package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.PretrainedTokenizer;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;
import java.util.TreeSet;

/**
 * Whisper's tokenizer: the byte-level BPE of a Hugging Face Whisper folder plus the
 * special tokens that steer the decoder: start of transcript, one token per language, the
 * task (transcribe or translate), "no timestamps" and the 1501 timestamp tokens
 * {@code <|0.00|>} to {@code <|30.00|>}.
 */
public final class WhisperTokenizer {

    /** Language codes in Whisper's order (their tokens follow start-of-transcript), with names. */
    private static final String[][] LANGUAGES = {
        {"en", "english"}, {"zh", "chinese"}, {"de", "german"}, {"es", "spanish"}, {"ru", "russian"},
        {"ko", "korean"}, {"fr", "french"}, {"ja", "japanese"}, {"pt", "portuguese"}, {"tr", "turkish"},
        {"pl", "polish"}, {"ca", "catalan"}, {"nl", "dutch"}, {"ar", "arabic"}, {"sv", "swedish"},
        {"it", "italian"}, {"id", "indonesian"}, {"hi", "hindi"}, {"fi", "finnish"}, {"vi", "vietnamese"},
        {"he", "hebrew"}, {"uk", "ukrainian"}, {"el", "greek"}, {"ms", "malay"}, {"cs", "czech"},
        {"ro", "romanian"}, {"da", "danish"}, {"hu", "hungarian"}, {"ta", "tamil"}, {"no", "norwegian"},
        {"th", "thai"}, {"ur", "urdu"}, {"hr", "croatian"}, {"bg", "bulgarian"}, {"lt", "lithuanian"},
        {"la", "latin"}, {"mi", "maori"}, {"ml", "malayalam"}, {"cy", "welsh"}, {"sk", "slovak"},
        {"te", "telugu"}, {"fa", "persian"}, {"lv", "latvian"}, {"bn", "bengali"}, {"sr", "serbian"},
        {"az", "azerbaijani"}, {"sl", "slovenian"}, {"kn", "kannada"}, {"et", "estonian"}, {"mk", "macedonian"},
        {"br", "breton"}, {"eu", "basque"}, {"is", "icelandic"}, {"hy", "armenian"}, {"ne", "nepali"},
        {"mn", "mongolian"}, {"bs", "bosnian"}, {"kk", "kazakh"}, {"sq", "albanian"}, {"sw", "swahili"},
        {"gl", "galician"}, {"mr", "marathi"}, {"pa", "punjabi"}, {"si", "sinhala"}, {"km", "khmer"},
        {"sn", "shona"}, {"yo", "yoruba"}, {"so", "somali"}, {"af", "afrikaans"}, {"oc", "occitan"},
        {"ka", "georgian"}, {"be", "belarusian"}, {"tg", "tajik"}, {"sd", "sindhi"}, {"gu", "gujarati"},
        {"am", "amharic"}, {"yi", "yiddish"}, {"lo", "lao"}, {"uz", "uzbek"}, {"fo", "faroese"},
        {"ht", "haitian creole"}, {"ps", "pashto"}, {"tk", "turkmen"}, {"nn", "nynorsk"}, {"mt", "maltese"},
        {"sa", "sanskrit"}, {"lb", "luxembourgish"}, {"my", "myanmar"}, {"bo", "tibetan"}, {"tl", "tagalog"},
        {"mg", "malagasy"}, {"as", "assamese"}, {"tt", "tatar"}, {"haw", "hawaiian"}, {"ln", "lingala"},
        {"ha", "hausa"}, {"ba", "bashkir"}, {"jw", "javanese"}, {"su", "sundanese"}, {"yue", "cantonese"},
    };

    private final PretrainedTokenizer text;
    private final int eot;
    private final int sot;
    private final int translate;
    private final int transcribe;
    private final int startOfPrevious;
    private final int noSpeech;
    private final int noTimestamps;
    private final int timestampBegin;
    /** Language code to token, in Whisper's order, only those the vocabulary has. */
    private final Map<String, Integer> languages = new LinkedHashMap<>();

    /**
     * Wraps a text tokenizer.
     *
     * @param text the BPE tokenizer of a Whisper folder
     * @param vocabSize the model's vocabulary, which bounds the timestamp tokens
     */
    public WhisperTokenizer(PretrainedTokenizer text, int vocabSize) {
        this.text = text;
        this.eot = need("<|endoftext|>");
        this.sot = need("<|startoftranscript|>");
        boolean multilingual = vocabSize >= 51865;
        int nLanguages = vocabSize >= 51866 ? 100 : 99;
        if (multilingual) {
            for (int i = 0; i < nLanguages; i++) {
                String code = LANGUAGES[i][0];
                OptionalInt id = text.id("<|" + code + "|>");
                languages.put(code, id.isPresent() ? id.getAsInt() : sot + 1 + i);
            }
        }
        // the special tokens follow the languages; tokenizer.json names them, older vocabularies may not
        int after = sot + 1 + (multilingual ? nLanguages : 0);
        this.translate = text.id("<|translate|>").orElse(after);
        this.transcribe = text.id("<|transcribe|>").orElse(after + 1);
        this.startOfPrevious = text.id("<|startofprev|>").orElse(after + 3);
        int ns = text.id("<|nospeech|>").orElse(-1);
        this.noSpeech = ns >= 0 ? ns : text.id("<|nocaptions|>").orElse(after + 4);
        this.noTimestamps = text.id("<|notimestamps|>").orElse(after + 5);
        this.timestampBegin = noTimestamps + 1;
    }

    /**
     * Loads the tokenizer of a Hugging Face Whisper folder.
     *
     * @param dir the folder ({@code tokenizer.json}, or {@code vocab.json} with {@code merges.txt})
     * @param vocabSize the model's vocabulary
     * @return the tokenizer
     */
    public static WhisperTokenizer load(Path dir, int vocabSize) {
        return new WhisperTokenizer(PretrainedTokenizer.load(dir), vocabSize);
    }

    private int need(String token) {
        return text.id(token).orElseThrow(() -> new IllegalArgumentException(
                "the tokenizer has no " + token + " token; is this a Whisper tokenizer?"));
    }

    /**
     * Returns the text tokenizer.
     *
     * @return the BPE tokenizer
     */
    public PretrainedTokenizer text() {
        return text;
    }

    /**
     * Encodes text without special tokens.
     *
     * @param s the text
     * @return its tokens
     */
    public int[] encode(String s) {
        return text.encode(s, false);
    }

    /**
     * Decodes text tokens, dropping special and timestamp tokens.
     *
     * @param ids the tokens
     * @return the text
     */
    public String decode(int[] ids) {
        int n = 0;
        int[] kept = new int[ids.length];
        for (int id : ids) {
            if (id < eot) {
                kept[n++] = id;
            }
        }
        return text.decode(java.util.Arrays.copyOf(kept, n), true);
    }

    /**
     * The end-of-text token.
     *
     * @return its id
     */
    public int eot() {
        return eot;
    }

    /**
     * The start-of-transcript token.
     *
     * @return its id
     */
    public int sot() {
        return sot;
    }

    /**
     * The task token for translation to English.
     *
     * @return its id
     */
    public int translate() {
        return translate;
    }

    /**
     * The task token for transcription.
     *
     * @return its id
     */
    public int transcribe() {
        return transcribe;
    }

    /**
     * The token before a prompt of earlier text.
     *
     * @return its id
     */
    public int startOfPrevious() {
        return startOfPrevious;
    }

    /**
     * The token whose probability after start-of-transcript says there is no speech.
     *
     * @return its id
     */
    public int noSpeech() {
        return noSpeech;
    }

    /**
     * The token that asks for text without timestamps.
     *
     * @return its id
     */
    public int noTimestamps() {
        return noTimestamps;
    }

    /**
     * The first timestamp token, {@code <|0.00|>}; each next one is 20 ms later.
     *
     * @return its id
     */
    public int timestampBegin() {
        return timestampBegin;
    }

    /**
     * Returns whether the model knows several languages.
     *
     * @return {@code false} for English-only models
     */
    public boolean multilingual() {
        return !languages.isEmpty();
    }

    /**
     * The language codes the model knows, in its order.
     *
     * @return codes such as {@code en} and {@code de}; empty for English-only models
     */
    public List<String> languageCodes() {
        return new ArrayList<>(languages.keySet());
    }

    /**
     * The token of a language.
     *
     * @param language a code ({@code de}) or an English name ({@code german})
     * @return its token
     * @throws IllegalArgumentException if the model does not know it
     */
    public int languageToken(String language) {
        String code = languageCode(language);
        Integer id = languages.get(code);
        if (id == null) {
            throw new IllegalArgumentException(multilingual()
                    ? "Whisper does not know the language '" + language + "'; use a code such as en, de or fr"
                    : "this Whisper model is English-only; it cannot take language '" + language + "'");
        }
        return id;
    }

    /**
     * Returns the language of a token.
     *
     * @param token a language token
     * @return its code, or {@code null} if it is not a language token
     */
    public String languageOf(int token) {
        for (Map.Entry<String, Integer> e : languages.entrySet()) {
            if (e.getValue() == token) {
                return e.getKey();
            }
        }
        return null;
    }

    /**
     * The tokens of all languages the model knows.
     *
     * @return their ids
     */
    public int[] languageTokens() {
        return languages.values().stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Normalizes a language to its code.
     *
     * @param language a code or an English name, any case
     * @return the code
     */
    public static String languageCode(String language) {
        String l = language.trim().toLowerCase(java.util.Locale.ROOT);
        for (String[] e : LANGUAGES) {
            if (e[0].equals(l) || e[1].equals(l)) {
                return e[0];
            }
        }
        return l;
    }

    /**
     * The English name of a language code.
     *
     * @param code a code such as {@code de}
     * @return its name such as {@code german}, or the code if unknown
     */
    public static String languageName(String code) {
        for (String[] e : LANGUAGES) {
            if (e[0].equals(code)) {
                return e[1];
            }
        }
        return code;
    }

    /**
     * Tokens that are never speech: symbols and music notes, as OpenAI's
     * {@code non_speech_tokens}, used when a model folder lists no {@code suppress_tokens}.
     *
     * @return their ids, sorted
     */
    public int[] nonSpeechTokens() {
        List<String> symbols = new ArrayList<>();
        for (char c : "\"#()*+/:;<=>@[\\]^_`{|}~「」『』".toCharArray()) {
            symbols.add(String.valueOf(c));
        }
        symbols.addAll(List.of("<<", ">>", "<<<", ">>>", "--", "---", "-(", "-[", "('", "(\"", "((", "))", "(((",
                ")))", "[[", "]]", "{{", "}}", "♪♪", "♪♪♪"));
        String misc = "♩♪♫♬♭♮♯";
        for (char c : misc.toCharArray()) {
            symbols.add(String.valueOf(c));
        }
        Set<Integer> out = new TreeSet<>();
        out.add(encode(" -")[0]);
        out.add(encode(" '")[0]);
        for (String s : symbols) {
            for (String v : new String[] {s, " " + s}) {
                int[] t = encode(v);
                if (t.length == 1 || t.length > 0 && misc.contains(s)) {
                    out.add(t[0]);
                }
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }
}
