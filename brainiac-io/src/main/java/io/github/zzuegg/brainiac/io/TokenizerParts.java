package io.github.zzuegg.brainiac.io;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The stages of a Hugging Face {@code tokenizer.json}: normalizers, pre-tokenizers, models
 * and decoders, each built from its JSON node as the {@code tokenizers} library does.
 */
final class TokenizerParts {

    private TokenizerParts() {
    }

    /** GPT-2's pre-tokenizer pattern, used by the ByteLevel pre-tokenizer. */
    static final String GPT2_PATTERN =
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    /** Llama 3's pre-tokenizer pattern (also most newer byte-level models). */
    static final String LLAMA3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}"
            + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    /**
     * GPT-4o's (tiktoken's o200k) pre-tokenizer pattern, which gpt-oss uses: words split
     * before capitals, contractions kept on the word, up to three digits.
     */
    static final String GPT4O_PATTERN = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
            + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
            + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*"
            + "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+"
            + "|\\s+(?!\\S)|\\s+";
    /** Qwen 2's pre-tokenizer pattern: Llama 3's with single digits. */
    static final String QWEN2_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
            + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    /**
     * DeepSeek LLM's (and DeepSeek-V2's) pre-tokenizer: splits applied one after another, as
     * llama.cpp's {@code deepseek-llm} (the letter class is the tokenizer's own list).
     */
    static final List<String> DEEPSEEK_LLM_PATTERNS = List.of(
            "[\r\n]",
            "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
            "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
            "\\s+$",
            "[一-龥ࠀ-一가-퟿]+",
            "\\p{N}+");
    /** DeepSeek-V3's pre-tokenizer, as llama.cpp's {@code deepseek-v3}. */
    static final List<String> DEEPSEEK_V3_PATTERNS = List.of(
            "\\p{N}{1,3}",
            "[一-龥぀-ゟ゠-ヿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+");
    /** Kimi K2's pre-tokenizer (its tiktoken pattern; {@code \p{Han}} runs split off first). */
    static final String KIMI_K2_PATTERN = "[\\p{IsHan}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]*"
            + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
            + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]+"
            + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
            + "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    static Pattern compile(String regex) {
        return Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
    }

    // ---------------------------------------------------------------- byte-level alphabet

    /** GPT-2's map from each byte to a printable character, and back. */
    static final char[] BYTE_TO_CHAR = new char[256];
    static final Map<Character, Integer> CHAR_TO_BYTE = new HashMap<>();

    static {
        int n = 0;
        for (int b = 0; b < 256; b++) {
            boolean printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            BYTE_TO_CHAR[b] = printable ? (char) b : (char) (256 + n++);
            CHAR_TO_BYTE.put(BYTE_TO_CHAR[b], b);
        }
    }

    static String toByteLevel(String s) {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        StringBuilder sb = new StringBuilder(bytes.length);
        for (byte b : bytes) {
            sb.append(BYTE_TO_CHAR[b & 0xFF]);
        }
        return sb.toString();
    }

    /** Decodes UTF-8, replacing each invalid byte with U+FFFD as Rust's lossy decoder does. */
    static String utf8(byte[] bytes) {
        try {
            return StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes)).toString();
        } catch (CharacterCodingException e) {
            return new String(bytes, StandardCharsets.UTF_8);
        }
    }

    // ---------------------------------------------------------------- normalizers

    static UnaryOperator<String> normalizer(Object node) {
        if (node == null) {
            return s -> s;
        }
        Map<String, Object> n = Json.object(node);
        String type = Json.string(n, "type", "");
        return switch (type) {
            case "NFC" -> s -> Normalizer.normalize(s, Normalizer.Form.NFC);
            case "NFD" -> s -> Normalizer.normalize(s, Normalizer.Form.NFD);
            case "NFKC", "Precompiled" -> s -> Normalizer.normalize(s, Normalizer.Form.NFKC);
            case "NFKD" -> s -> Normalizer.normalize(s, Normalizer.Form.NFKD);
            case "Lowercase" -> s -> s.toLowerCase(Locale.ROOT);
            case "StripAccents" -> TokenizerParts::stripAccents;
            case "Strip" -> {
                boolean left = !Boolean.FALSE.equals(n.get("left"));
                boolean right = !Boolean.FALSE.equals(n.get("right"));
                yield s -> left && right ? s.strip() : left ? s.stripLeading() : right ? s.stripTrailing() : s;
            }
            case "Replace" -> {
                Pattern p = pattern(n.get("pattern"));
                String content = Matcher.quoteReplacement(Json.string(n, "content", ""));
                yield s -> p.matcher(s).replaceAll(content);
            }
            case "Prepend" -> {
                String prepend = Json.string(n, "prepend", "");
                yield s -> s.isEmpty() ? s : prepend + s;
            }
            case "Sequence" -> {
                List<UnaryOperator<String>> parts = new ArrayList<>();
                for (Object o : Json.array(n.get("normalizers"))) {
                    parts.add(normalizer(o));
                }
                yield s -> {
                    for (UnaryOperator<String> p : parts) {
                        s = p.apply(s);
                    }
                    return s;
                };
            }
            case "BertNormalizer" -> bert(n);
            default -> throw new IllegalArgumentException("unsupported normalizer " + type);
        };
    }

    private static String stripAccents(String s) {
        return s.codePoints().filter(c -> Character.getType(c) != Character.NON_SPACING_MARK)
                .collect(StringBuilder::new, StringBuilder::appendCodePoint, StringBuilder::append).toString();
    }

    private static UnaryOperator<String> bert(Map<String, Object> n) {
        boolean clean = !Boolean.FALSE.equals(n.get("clean_text"));
        boolean chinese = !Boolean.FALSE.equals(n.get("handle_chinese_chars"));
        boolean lower = !Boolean.FALSE.equals(n.get("lowercase"));
        Object strip = n.get("strip_accents");
        boolean accents = strip == null ? lower : Boolean.TRUE.equals(strip);
        return s -> {
            StringBuilder sb = new StringBuilder();
            s.codePoints().forEach(c -> {
                if (clean) {
                    if (c == 0 || c == 0xFFFD || (Character.isISOControl(c) && c != '\t' && c != '\n' && c != '\r')
                            || Character.getType(c) == Character.FORMAT) {
                        return;
                    }
                    if (c == '\t' || c == '\n' || c == '\r' || Character.isWhitespace(c) || Character.isSpaceChar(c)) {
                        sb.append(' ');
                        return;
                    }
                }
                if (chinese && isChinese(c)) {
                    sb.append(' ').appendCodePoint(c).append(' ');
                } else {
                    sb.appendCodePoint(c);
                }
            });
            String out = sb.toString();
            if (accents) {
                out = stripAccents(Normalizer.normalize(out, Normalizer.Form.NFD));
            }
            return lower ? out.toLowerCase(Locale.ROOT) : out;
        };
    }

    private static boolean isChinese(int c) {
        return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x20000 && c <= 0x2A6DF)
                || (c >= 0x2A700 && c <= 0x2B73F) || (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B820 && c <= 0x2CEAF)
                || (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);
    }

    private static Pattern pattern(Object node) {
        Map<String, Object> p = Json.object(node);
        if (p.get("Regex") instanceof String r) {
            return compile(r);
        }
        return Pattern.compile(Pattern.quote(Json.string(p, "String", "")));
    }

    // ---------------------------------------------------------------- pre-tokenizers

    /** Splits text into words; {@code first} says whether this is the start of the input. */
    interface PreTokenizer {
        List<String> split(String text, boolean first);
    }

    static PreTokenizer preTokenizer(Object node) {
        if (node == null) {
            return (s, first) -> List.of(s);
        }
        Map<String, Object> n = Json.object(node);
        String type = Json.string(n, "type", "");
        return switch (type) {
            case "Sequence" -> {
                List<PreTokenizer> parts = new ArrayList<>();
                for (Object o : Json.array(n.get("pretokenizers"))) {
                    parts.add(preTokenizer(o));
                }
                yield (s, first) -> {
                    List<String> words = List.of(s);
                    for (PreTokenizer p : parts) {
                        List<String> next = new ArrayList<>();
                        for (String w : words) {
                            next.addAll(p.split(w, first));
                        }
                        words = next;
                    }
                    return words;
                };
            }
            case "ByteLevel" -> {
                boolean prefix = Boolean.TRUE.equals(n.get("add_prefix_space"));
                boolean regex = !Boolean.FALSE.equals(n.get("use_regex"));
                Pattern p = compile(GPT2_PATTERN);
                yield (s, first) -> {
                    if (prefix && !s.startsWith(" ")) {
                        s = " " + s;
                    }
                    List<String> words = regex ? split(p, s, "Isolated", false) : List.of(s);
                    List<String> out = new ArrayList<>(words.size());
                    for (String w : words) {
                        out.add(toByteLevel(w));
                    }
                    return out;
                };
            }
            case "Split" -> {
                Pattern p = pattern(n.get("pattern"));
                String behavior = Json.string(n, "behavior", "Isolated");
                boolean invert = Boolean.TRUE.equals(n.get("invert"));
                yield (s, first) -> split(p, s, behavior, invert);
            }
            case "Metaspace" -> {
                String rep = Json.string(n, "replacement", "▁");
                String scheme = metaspaceScheme(n);
                boolean splitWords = !Boolean.FALSE.equals(n.get("split"));
                yield (s, first) -> {
                    String t = s.replace(" ", rep);
                    if (!t.startsWith(rep) && (scheme.equals("always") || (scheme.equals("first") && first))) {
                        t = rep + t;
                    }
                    return splitWords ? split(Pattern.compile(Pattern.quote(rep)), t, "MergedWithNext", false)
                            : List.of(t);
                };
            }
            case "Whitespace" -> {
                Pattern p = compile("\\w+|[^\\w\\s]+");
                yield (s, first) -> split(p, s, "Removed", true);
            }
            case "WhitespaceSplit" -> {
                Pattern p = compile("\\s+");
                yield (s, first) -> split(p, s, "Removed", false);
            }
            case "BertPreTokenizer" -> {
                Pattern ws = compile("\\s+");
                Pattern punct = compile("[\\p{P}\\p{S}&&[\\x21-\\x2F\\x3A-\\x40\\x5B-\\x60\\x7B-\\x7E]]|\\p{P}");
                yield (s, first) -> {
                    List<String> out = new ArrayList<>();
                    for (String w : split(ws, s, "Removed", false)) {
                        out.addAll(split(punct, w, "Isolated", false));
                    }
                    return out;
                };
            }
            case "Punctuation" -> {
                Pattern punct = compile("[\\x21-\\x2F\\x3A-\\x40\\x5B-\\x60\\x7B-\\x7E]|\\p{P}");
                String behavior = Json.string(n, "behavior", "Isolated");
                yield (s, first) -> split(punct, s, behavior, false);
            }
            case "Digits" -> {
                Pattern p = compile(Boolean.TRUE.equals(n.get("individual_digits")) ? "\\p{N}" : "\\p{N}+");
                yield (s, first) -> split(p, s, "Isolated", false);
            }
            default -> throw new IllegalArgumentException("unsupported pre-tokenizer " + type);
        };
    }

    static String metaspaceScheme(Map<String, Object> n) {
        Object scheme = n.get("prepend_scheme");
        if (scheme instanceof String s) {
            return s;
        }
        return Boolean.FALSE.equals(n.get("add_prefix_space")) ? "never" : "always";
    }

    /** Splits like the tokenizers library's {@code SplitDelimiterBehavior}. */
    static List<String> split(Pattern p, String s, String behavior, boolean invert) {
        // pieces: (text, isMatch)
        List<String> texts = new ArrayList<>();
        List<Boolean> matched = new ArrayList<>();
        Matcher m = p.matcher(s);
        int at = 0;
        while (m.find()) {
            if (m.end() == m.start()) {
                continue;
            }
            if (m.start() > at) {
                texts.add(s.substring(at, m.start()));
                matched.add(invert);
            }
            texts.add(m.group());
            matched.add(!invert);
            at = m.end();
        }
        if (at < s.length()) {
            texts.add(s.substring(at));
            matched.add(invert);
        }
        List<String> out = new ArrayList<>();
        switch (behavior) {
            case "Removed" -> {
                for (int i = 0; i < texts.size(); i++) {
                    if (!matched.get(i)) {
                        out.add(texts.get(i));
                    }
                }
            }
            case "Isolated" -> out.addAll(texts);
            case "MergedWithPrevious" -> {
                for (int i = 0; i < texts.size(); i++) {
                    if (matched.get(i) && !out.isEmpty() && !matched.get(i - 1)) {
                        out.set(out.size() - 1, out.getLast() + texts.get(i));
                    } else {
                        out.add(texts.get(i));
                    }
                }
            }
            case "MergedWithNext" -> {
                String pending = null;
                for (int i = 0; i < texts.size(); i++) {
                    if (matched.get(i)) {
                        if (pending != null) {
                            out.add(pending);
                        }
                        pending = texts.get(i);
                    } else {
                        out.add(pending == null ? texts.get(i) : pending + texts.get(i));
                        pending = null;
                    }
                }
                if (pending != null) {
                    out.add(pending);
                }
            }
            case "Contiguous" -> {
                for (int i = 0; i < texts.size(); i++) {
                    if (matched.get(i) && i > 0 && matched.get(i - 1)) {
                        out.set(out.size() - 1, out.getLast() + texts.get(i));
                    } else {
                        out.add(texts.get(i));
                    }
                }
            }
            default -> throw new IllegalArgumentException("unsupported split behavior " + behavior);
        }
        return out;
    }

    // ---------------------------------------------------------------- models

    /** Turns one pre-tokenized word into ids. */
    interface Model {
        void encode(String word, List<Integer> out);
    }

    /** Ids of {@code <0xXX>} byte tokens, or null if any is missing. */
    static int[] byteTokens(Map<String, Integer> vocab) {
        int[] ids = new int[256];
        for (int b = 0; b < 256; b++) {
            Integer id = vocab.get(String.format("<0x%02X>", b));
            if (id == null) {
                return null;
            }
            ids[b] = id;
        }
        return ids;
    }

    /** Byte-pair encoding by merge rank, as the tokenizers library and tiktoken do. */
    static final class Bpe implements Model {
        private final Map<String, Integer> vocab;
        private final Map<String, Integer> ranks;
        private final Integer unk;
        private final String prefix;
        private final String suffix;
        private final boolean fuseUnk;
        private final int[] bytes;
        private final boolean ignoreMerges;
        private final Map<String, int[]> cache = new java.util.concurrent.ConcurrentHashMap<>();

        Bpe(Map<String, Integer> vocab, List<String[]> merges, String unkToken, String prefix, String suffix,
                boolean fuseUnk, boolean byteFallback, boolean ignoreMerges) {
            this.vocab = vocab;
            this.ranks = new HashMap<>();
            for (int i = 0; i < merges.size(); i++) {
                ranks.putIfAbsent(merges.get(i)[0] + "\u0000" + merges.get(i)[1], i);
            }
            this.unk = unkToken == null ? null : vocab.get(unkToken);
            this.prefix = prefix == null ? "" : prefix;
            this.suffix = suffix == null ? "" : suffix;
            this.fuseUnk = fuseUnk;
            this.bytes = byteFallback ? byteTokens(vocab) : null;
            this.ignoreMerges = ignoreMerges;
        }

        @Override
        public void encode(String word, List<Integer> out) {
            int[] ids = cache.get(word);
            if (ids == null) {
                ids = compute(word);
                if (cache.size() < 100_000) {
                    cache.put(word, ids);
                }
            }
            for (int id : ids) {
                out.add(id);
            }
        }

        private int[] compute(String word) {
            List<Integer> out = new ArrayList<>();
            if (ignoreMerges && vocab.containsKey(word)) {
                return new int[] {vocab.get(word)};
            }
            List<String> symbols = new ArrayList<>();
            int n = word.codePointCount(0, word.length());
            int i = 0;
            for (int at = 0; at < word.length(); i++) {
                int cp = word.codePointAt(at);
                String s = new String(Character.toChars(cp));
                at += Character.charCount(cp);
                if (i > 0) {
                    s = prefix + s;
                }
                if (i == n - 1) {
                    s = s + suffix;
                }
                symbols.add(s);
            }
            // merge the pair of lowest rank, leftmost first, until none is left
            while (symbols.size() > 1) {
                int best = Integer.MAX_VALUE;
                int pos = -1;
                for (int k = 0; k + 1 < symbols.size(); k++) {
                    Integer r = ranks.get(symbols.get(k) + "\u0000" + symbols.get(k + 1));
                    if (r != null && r < best) {
                        best = r;
                        pos = k;
                    }
                }
                if (pos < 0) {
                    break;
                }
                String b = symbols.get(pos + 1);
                if (!prefix.isEmpty() && b.startsWith(prefix)) {
                    b = b.substring(prefix.length());
                }
                symbols.set(pos, symbols.get(pos) + b);
                symbols.remove(pos + 1);
            }
            boolean lastUnk = false;
            for (String s : symbols) {
                Integer id = vocab.get(s);
                if (id != null) {
                    out.add(id);
                    lastUnk = false;
                } else if (bytes != null) {
                    for (byte b : s.getBytes(StandardCharsets.UTF_8)) {
                        out.add(bytes[b & 0xFF]);
                    }
                    lastUnk = false;
                } else if (unk != null) {
                    if (!(fuseUnk && lastUnk)) {
                        out.add(unk);
                    }
                    lastUnk = true;
                } else {
                    throw new IllegalArgumentException("no token for \"" + s + "\" and no unknown token");
                }
            }
            return out.stream().mapToInt(Integer::intValue).toArray();
        }
    }

    /** Greedy longest-match word pieces, as BERT does. */
    static final class WordPiece implements Model {
        private final Map<String, Integer> vocab;
        private final int unk;
        private final String prefix;
        private final int maxChars;

        WordPiece(Map<String, Integer> vocab, String unkToken, String prefix, int maxChars) {
            this.vocab = vocab;
            Integer u = vocab.get(unkToken);
            if (u == null) {
                throw new IllegalArgumentException("unknown token " + unkToken + " is not in the vocabulary");
            }
            this.unk = u;
            this.prefix = prefix;
            this.maxChars = maxChars;
        }

        @Override
        public void encode(String word, List<Integer> out) {
            if (word.codePointCount(0, word.length()) > maxChars) {
                out.add(unk);
                return;
            }
            List<Integer> ids = new ArrayList<>();
            int start = 0;
            while (start < word.length()) {
                int end = word.length();
                Integer found = null;
                while (start < end) {
                    String piece = word.substring(start, end);
                    if (start > 0) {
                        piece = prefix + piece;
                    }
                    found = vocab.get(piece);
                    if (found != null) {
                        break;
                    }
                    end = word.offsetByCodePoints(end, -1);
                }
                if (found == null) {
                    out.add(unk);
                    return;
                }
                ids.add(found);
                start = end;
            }
            out.addAll(ids);
        }
    }

    /** The most likely segmentation under piece log-probabilities (Viterbi), as SentencePiece does. */
    static final class Unigram implements Model {
        private final Map<String, Integer> vocab = new HashMap<>();
        private final double[] scores;
        private final int unk;
        private final double unkScore;
        private final int[] bytes;
        private final int maxLength;
        private final boolean fuseUnk;

        Unigram(List<String> pieces, double[] scores, int unk, boolean byteFallback, boolean fuseUnk) {
            this.scores = scores;
            int longest = 1;
            double min = 0;
            for (int i = 0; i < pieces.size(); i++) {
                vocab.putIfAbsent(pieces.get(i), i);
                longest = Math.max(longest, pieces.get(i).length());
                min = Math.min(min, scores[i]);
            }
            this.maxLength = longest;
            this.unk = unk;
            this.unkScore = min - 10;
            this.bytes = byteFallback ? byteTokens(vocab) : null;
            this.fuseUnk = fuseUnk;
        }

        Unigram excluding(java.util.Set<Integer> ids) {
            for (int id : ids) {
                vocab.values().remove(id);
            }
            return this;
        }

        @Override
        public void encode(String word, List<Integer> out) {
            int n = word.length();
            double[] best = new double[n + 1];
            int[] from = new int[n + 1];
            int[] piece = new int[n + 1];
            java.util.Arrays.fill(best, Double.NEGATIVE_INFINITY);
            best[0] = 0;
            for (int i = 0; i < n; i++) {
                if (best[i] == Double.NEGATIVE_INFINITY) {
                    continue;
                }
                boolean single = false;
                int charEnd = word.offsetByCodePoints(i, 1);
                for (int end = i + 1; end <= Math.min(n, i + maxLength); end++) {
                    Integer id = vocab.get(word.substring(i, end));
                    if (id != null) {
                        if (end == charEnd) {
                            single = true;
                        }
                        double s = best[i] + scores[id];
                        if (s > best[end]) {
                            best[end] = s;
                            from[end] = i;
                            piece[end] = id;
                        }
                    }
                }
                if (!single) {
                    double s = best[i] + unkScore;
                    if (s > best[charEnd]) {
                        best[charEnd] = s;
                        from[charEnd] = i;
                        piece[charEnd] = -1;
                    }
                }
            }
            List<int[]> path = new ArrayList<>();
            for (int at = n; at > 0; at = from[at]) {
                path.add(new int[] {from[at], at, piece[at]});
            }
            boolean lastUnk = false;
            for (int k = path.size() - 1; k >= 0; k--) {
                int[] p = path.get(k);
                if (p[2] >= 0) {
                    out.add(p[2]);
                    lastUnk = false;
                } else if (bytes != null) {
                    for (byte b : word.substring(p[0], p[1]).getBytes(StandardCharsets.UTF_8)) {
                        out.add(bytes[b & 0xFF]);
                    }
                    lastUnk = false;
                } else {
                    if (!(fuseUnk && lastUnk)) {
                        out.add(unk);
                    }
                    lastUnk = true;
                }
            }
        }
    }

    /**
     * SentencePiece's byte-pair encoding: repeatedly merges the adjacent pair whose union is
     * the piece with the highest score (llama.cpp's "SPM" tokenizer works the same way).
     */
    static final class ScoreBpe implements Model {
        private final Map<String, Integer> vocab;
        private final double[] scores;
        private final int unk;
        private final int[] bytes;

        ScoreBpe(Map<String, Integer> vocab, double[] scores, int unk, boolean byteFallback) {
            this.vocab = vocab;
            this.scores = scores;
            this.unk = unk;
            this.bytes = byteFallback ? byteTokens(vocab) : null;
        }

        @Override
        public void encode(String word, List<Integer> out) {
            List<String> sym = new ArrayList<>();
            for (int at = 0; at < word.length(); ) {
                int cp = word.codePointAt(at);
                sym.add(new String(Character.toChars(cp)));
                at += Character.charCount(cp);
            }
            // a linked list of symbols with a queue of candidate merges, best score first
            int n = sym.size();
            String[] text = sym.toArray(new String[0]);
            int[] prev = new int[n];
            int[] next = new int[n];
            for (int i = 0; i < n; i++) {
                prev[i] = i - 1;
                next[i] = i + 1 < n ? i + 1 : -1;
            }
            record Cand(int left, int right, double score, String merged) {
            }
            PriorityQueue<Cand> queue = new PriorityQueue<>((a, b) -> a.score != b.score
                    ? Double.compare(b.score, a.score) : Integer.compare(a.left, b.left));
            java.util.function.BiConsumer<Integer, Integer> offer = (l, r) -> {
                if (l < 0 || r < 0) {
                    return;
                }
                String m = text[l] + text[r];
                Integer id = vocab.get(m);
                if (id != null) {
                    queue.add(new Cand(l, r, scores[id], m));
                }
            };
            for (int i = 0; i + 1 < n; i++) {
                offer.accept(i, i + 1);
            }
            while (!queue.isEmpty()) {
                Cand c = queue.poll();
                if (text[c.left] == null || text[c.right] == null || next[c.left] != c.right
                        || !(text[c.left] + text[c.right]).equals(c.merged)) {
                    continue;   // stale
                }
                text[c.left] = c.merged;
                text[c.right] = null;
                next[c.left] = next[c.right];
                if (next[c.right] >= 0) {
                    prev[next[c.right]] = c.left;
                }
                offer.accept(prev[c.left], c.left);
                offer.accept(c.left, next[c.left]);
            }
            for (int i = 0; i >= 0 && n > 0; i = next[i]) {
                Integer id = vocab.get(text[i]);
                if (id != null) {
                    out.add(id);
                } else if (bytes != null) {
                    for (byte b : text[i].getBytes(StandardCharsets.UTF_8)) {
                        out.add(bytes[b & 0xFF]);
                    }
                } else {
                    out.add(unk);
                }
            }
        }
    }

    /** One id per whole word. */
    record WordLevel(Map<String, Integer> vocab, int unk) implements Model {
        @Override
        public void encode(String word, List<Integer> out) {
            out.add(vocab.getOrDefault(word, unk));
        }
    }

    // ---------------------------------------------------------------- decoders

    /** Turns tokens back into text pieces, as the tokenizers library's decoders do. */
    interface Decoder {
        List<String> decode(List<String> tokens);
    }

    static Decoder decoder(Object node) {
        if (node == null) {
            return tokens -> tokens;
        }
        Map<String, Object> n = Json.object(node);
        String type = Json.string(n, "type", "");
        return switch (type) {
            case "Sequence" -> {
                List<Decoder> parts = new ArrayList<>();
                for (Object o : Json.array(n.get("decoders"))) {
                    parts.add(decoder(o));
                }
                yield tokens -> {
                    for (Decoder d : parts) {
                        tokens = d.decode(tokens);
                    }
                    return tokens;
                };
            }
            case "ByteLevel" -> tokens -> {
                ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                StringBuilder sb = new StringBuilder();
                for (String t : tokens) {
                    for (int i = 0; i < t.length(); i++) {
                        Integer b = CHAR_TO_BYTE.get(t.charAt(i));
                        if (b != null) {
                            bytes.write(b);
                        } else {
                            sb.append(utf8(bytes.toByteArray()));
                            bytes.reset();
                            sb.append(t.charAt(i));
                        }
                    }
                }
                sb.append(utf8(bytes.toByteArray()));
                return List.of(sb.toString());
            };
            case "Metaspace" -> {
                String rep = Json.string(n, "replacement", "▁");
                boolean strip = !metaspaceScheme(n).equals("never");
                yield tokens -> {
                    List<String> out = new ArrayList<>();
                    for (int i = 0; i < tokens.size(); i++) {
                        String t = tokens.get(i).replace(rep, " ");
                        if (i == 0 && strip && t.startsWith(" ")) {
                            t = t.substring(1);
                        }
                        out.add(t);
                    }
                    return out;
                };
            }
            case "Replace" -> {
                Pattern p = pattern(n.get("pattern"));
                String content = Matcher.quoteReplacement(Json.string(n, "content", ""));
                yield tokens -> tokens.stream().map(t -> p.matcher(t).replaceAll(content)).toList();
            }
            case "ByteFallback" -> tokens -> {
                List<String> out = new ArrayList<>();
                ByteArrayOutputStream pending = new ByteArrayOutputStream();
                for (String t : tokens) {
                    if (t.length() == 6 && t.startsWith("<0x") && t.endsWith(">")) {
                        pending.write(Integer.parseInt(t.substring(3, 5), 16));
                        continue;
                    }
                    flushBytes(pending, out);
                    out.add(t);
                }
                flushBytes(pending, out);
                return out;
            };
            case "Fuse" -> tokens -> List.of(String.join("", tokens));
            case "Strip" -> {
                String content = Json.string(n, "content", " ");
                int start = (int) Json.integer(n, "start", 0);
                int stop = (int) Json.integer(n, "stop", 0);
                yield tokens -> tokens.stream().map(t -> {
                    for (int i = 0; i < start && t.startsWith(content); i++) {
                        t = t.substring(content.length());
                    }
                    for (int i = 0; i < stop && t.endsWith(content); i++) {
                        t = t.substring(0, t.length() - content.length());
                    }
                    return t;
                }).toList();
            }
            case "WordPiece" -> {
                String prefix = Json.string(n, "prefix", "##");
                boolean cleanup = !Boolean.FALSE.equals(n.get("cleanup"));
                // like the tokenizers library, the cleanup runs on each token on its own
                yield tokens -> {
                    List<String> out = new ArrayList<>();
                    for (int i = 0; i < tokens.size(); i++) {
                        String t = tokens.get(i);
                        if (i > 0) {
                            t = t.startsWith(prefix) ? t.substring(prefix.length()) : " " + t;
                        }
                        if (cleanup) {
                            t = t.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",")
                                    .replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m")
                                    .replace(" do not", " don't").replace(" 's", "'s").replace(" 've", "'ve")
                                    .replace(" 're", "'re");
                        }
                        out.add(t);
                    }
                    return out;
                };
            }
            case "BPEDecoder" -> {
                String suffix = Json.string(n, "suffix", "</w>");
                yield tokens -> {
                    List<String> out = new ArrayList<>();
                    for (int i = 0; i < tokens.size(); i++) {
                        out.add(tokens.get(i).replace(suffix, i == tokens.size() - 1 ? "" : " "));
                    }
                    return out;
                };
            }
            default -> throw new IllegalArgumentException("unsupported decoder " + type);
        };
    }

    private static void flushBytes(ByteArrayOutputStream pending, List<String> out) {
        if (pending.size() == 0) {
            return;
        }
        byte[] b = pending.toByteArray();
        pending.reset();
        try {
            out.add(StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(b)).toString());
        } catch (CharacterCodingException e) {
            for (int i = 0; i < b.length; i++) {
                out.add("�");
            }
        }
    }
}
