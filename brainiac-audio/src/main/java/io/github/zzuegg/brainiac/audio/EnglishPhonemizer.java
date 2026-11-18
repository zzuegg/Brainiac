package io.github.zzuegg.brainiac.audio;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

/**
 * English text to phonemes, as Kokoro reads them: a port of misaki's English G2P
 * (hexgrad/misaki, Apache 2.0) with its gold and silver pronunciation dictionaries for
 * American and British English (about 180,000 words each), stress rules, plural, past and
 * {@code -ing} forms of known words, numbers, ordinals, years, decimals and currency, and
 * context such as "the" before a vowel. Words in no dictionary are split into known parts or
 * read with letter-to-sound rules. Part-of-speech tags come from a few context rules instead
 * of a statistical tagger, so heteronyms such as "read" or "record" can be read the other way.
 *
 * <pre>{@code
 * String ps = EnglishPhonemizer.american().phonemize("It's 5:30, and the café is open.");
 * }</pre>
 */
public final class EnglishPhonemizer {

    private static final String PRIMARY = "ˈ";
    private static final String SECONDARY = "ˌ";
    private static final String STRESSES = "ˌˈ";
    private static final String VOWELS = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
    private static final String CONSONANTS = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
    private static final String DIPHTHONGS = "AIOQWYʤʧ";
    private static final String PUNCTS = ";:,.!?—…\"“”";
    private static final String NON_QUOTE_PUNCTS = ";:,.!?—…";
    private static final String SUBTOKEN_JUNKS = "',-._‘’/";
    private static final String US_TAUS = "AIOWYiuæɑəɛɪɹʊʌ";
    private static final Set<String> ORDINALS = Set.of("st", "nd", "rd", "th");
    private static final Map<String, String> SYMBOLS = Map.of("%", "percent", "&", "and", "+", "plus", "@", "at");
    private static final Map<String, String[]> CURRENCIES = Map.of("$", new String[] {"dollar", "cent"},
            "£", new String[] {"pound", "pence"}, "€", new String[] {"euro", "cent"});
    private static final Pattern SUBTOKEN = Pattern.compile("^['‘’]+|\\p{Lu}(?=\\p{Lu}\\p{Ll})|(?:^-)?(?:\\d?[,.]?\\d)+"
            + "|[-_]+|['‘’]{2,}|\\p{L}*?(?:['‘’]\\p{L})*?\\p{Ll}(?=\\p{Lu})|\\p{L}+(?:['‘’]\\p{L})*|[^-_\\p{L}'‘’\\d]"
            + "|['‘’]+$");
    private static final Pattern CHUNK = Pattern.compile("(\\S+)(\\s*)");
    private static final String LEADING = "\"“‘([{¿¡«";
    private static final String TRAILING = "\"”’)]}.,;:!?…»";
    private static final Set<String> VERB_CUES = Set.of("to", "will", "would", "can", "could", "shall", "should", "may",
            "might", "must", "do", "does", "did", "don't", "didn't", "doesn't", "won't", "can't", "i", "you", "we",
            "they", "please", "let's");
    private static final Set<String> PAST_CUES = Set.of("was", "were", "been", "had", "has", "have", "is", "are",
            "being", "got", "gets", "get");
    private static final Set<String> NOUN_CUES = Set.of("the", "a", "an", "this", "that", "these", "those", "my", "your",
            "his", "her", "its", "our", "their", "some", "any", "no", "every", "each", "of", "new", "old");

    private static volatile EnglishPhonemizer american;
    private static volatile EnglishPhonemizer british;

    private final boolean isBritish;
    private final Map<String, Object> golds;
    private final Map<String, Object> silvers;

    private EnglishPhonemizer(boolean british) {
        this.isBritish = british;
        String prefix = british ? "gb" : "us";
        this.golds = grow(read(prefix + "_gold.tsv.gz"));
        this.silvers = grow(read(prefix + "_silver.tsv.gz"));
    }

    /**
     * American English, as Kokoro's {@code a} voices speak it.
     *
     * @return the shared phonemizer (its dictionaries load on first use)
     */
    public static EnglishPhonemizer american() {
        EnglishPhonemizer p = american;
        if (p == null) {
            synchronized (EnglishPhonemizer.class) {
                if (american == null) {
                    american = new EnglishPhonemizer(false);
                }
                p = american;
            }
        }
        return p;
    }

    /**
     * British English, as Kokoro's {@code b} voices speak it.
     *
     * @return the shared phonemizer
     */
    public static EnglishPhonemizer british() {
        EnglishPhonemizer p = british;
        if (p == null) {
            synchronized (EnglishPhonemizer.class) {
                if (british == null) {
                    british = new EnglishPhonemizer(true);
                }
                p = british;
            }
        }
        return p;
    }

    /** @return whether this reads British English */
    public boolean isBritish() {
        return isBritish;
    }

    private static Map<String, Object> read(String name) {
        Map<String, Object> out = new HashMap<>(200_000);
        try (InputStream in = EnglishPhonemizer.class.getResourceAsStream("g2p/" + name)) {
            if (in == null) {
                throw new IllegalStateException("missing dictionary g2p/" + name);
            }
            BufferedReader r = new BufferedReader(new InputStreamReader(new GZIPInputStream(in),
                    StandardCharsets.UTF_8));
            String line;
            while ((line = r.readLine()) != null) {
                String[] f = line.split("\t", -1);
                if (f.length == 2 && !f[1].contains("=")) {
                    out.put(f[0], f[1]);
                } else {
                    Map<String, String> tags = new HashMap<>();
                    for (int i = 1; i < f.length; i++) {
                        int eq = f[i].indexOf('=');
                        String v = f[i].substring(eq + 1);
                        tags.put(f[i].substring(0, eq), v.isEmpty() ? null : v);
                    }
                    out.put(f[0], tags);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return out;
    }

    /** Adds the capitalized form of lowercase words and the lowercase form of capitalized ones. */
    private static Map<String, Object> grow(Map<String, Object> d) {
        Map<String, Object> e = new HashMap<>(d.size() * 2);
        for (Map.Entry<String, Object> kv : d.entrySet()) {
            String k = kv.getKey();
            if (k.length() < 2) {
                continue;
            }
            if (k.equals(lower(k))) {
                String cap = capitalize(k);
                if (!k.equals(cap)) {
                    e.put(cap, kv.getValue());
                }
            } else if (k.equals(capitalize(lower(k)))) {
                e.put(lower(k), kv.getValue());
            }
        }
        e.putAll(d);
        return e;
    }

    // ------------------------------------------------------------------ public API

    /**
     * Turns text into phonemes: words separated by spaces, punctuation kept.
     *
     * @param text English text
     * @return the phonemes, as Kokoro's vocabulary spells them
     */
    public String phonemize(String text) {
        StringBuilder out = new StringBuilder();
        for (Word w : words(text)) {
            out.append(w.phonemes()).append(w.whitespace());
        }
        return out.toString().strip();
    }

    /**
     * One word or punctuation mark of the input and its phonemes.
     *
     * @param text the text
     * @param phonemes its phonemes ("" for silent symbols)
     * @param whitespace the whitespace that followed it ("" or " ")
     */
    public record Word(String text, String phonemes, String whitespace) {
    }

    /**
     * Turns text into phonemes word by word, for splitting long text at punctuation.
     *
     * @param text English text
     * @return the words in order
     */
    public List<Word> words(String text) {
        List<Object> tokens = tokenize(text);
        Context ctx = new Context(null, false);
        for (int i = tokens.size() - 1; i >= 0; i--) {
            Object o = tokens.get(i);
            if (o instanceof Tok t) {
                ctx = context(ctx, t.phonemes, t);
                continue;
            }
            @SuppressWarnings("unchecked")
            List<Tok> w = (List<Tok>) o;
            tagGroup(tokens, i, w);
            int left = 0;
            int right = w.size();
            boolean fallback = false;
            while (left < right) {
                boolean fixed = false;
                for (int j = left; j < right; j++) {
                    fixed |= w.get(j).phonemes != null;
                }
                Tok t = fixed ? null : merge(w.subList(left, right), null);
                Found f = t == null ? null : lexicon(t, ctx);
                if (f != null) {
                    w.get(left).phonemes = f.ps;
                    for (int j = left + 1; j < right; j++) {
                        w.get(j).phonemes = "";
                    }
                    ctx = context(ctx, f.ps, t);
                    right = left;
                    left = 0;
                } else if (left + 1 < right) {
                    left++;
                } else {
                    right--;
                    Tok last = w.get(right);
                    if (last.phonemes == null) {
                        if (allIn(last.text, SUBTOKEN_JUNKS)) {
                            last.phonemes = "";
                        } else {
                            fallback = true;
                            break;
                        }
                    }
                    left = 0;
                }
            }
            if (fallback) {
                Tok t = merge(w, null);
                w.get(0).phonemes = LetterToSound.read(t.text, this);
                for (int j = 1; j < w.size(); j++) {
                    w.get(j).phonemes = "";
                }
                ctx = context(ctx, w.get(0).phonemes, t);
            } else {
                resolve(w);
            }
        }
        List<Word> out = new ArrayList<>();
        for (Object o : tokens) {
            Tok t;
            if (o instanceof Tok single) {
                t = single;
            } else {
                @SuppressWarnings("unchecked")
                List<Tok> w = (List<Tok>) o;
                t = merge(w, "");
            }
            String ps = t.phonemes == null ? "" : t.phonemes;
            if (!isBritish) {
                ps = ps.replace('ɾ', 'T');
            }
            out.add(new Word(t.text, ps, t.ws.isEmpty() ? "" : " "));
        }
        return out;
    }

    // ------------------------------------------------------------------ tokens

    private static final class Tok {
        String text;
        String tag;
        String ws;
        boolean head = true;
        String phonemes;
        String currency;
        boolean prespace;

        Tok(String text, String ws) {
            this.text = text;
            this.ws = ws;
        }
    }

    private record Context(Boolean futureVowel, boolean futureTo) {
    }

    private record Found(String ps, int rating) {
    }

    /** Splits text into punctuation tokens and word groups (lists of subtokens with no space between). */
    private List<Object> tokenize(String text) {
        text = Normalizer.normalize(text, Normalizer.Form.NFKC).replace('‘', '\'').replace('’', '\'');
        List<Object> out = new ArrayList<>();
        Matcher m = CHUNK.matcher(text.strip());
        String currency = null;
        while (m.find()) {
            String chunk = m.group(1);
            String ws = m.group(2).isEmpty() ? "" : " ";
            List<String> pieces = new ArrayList<>();
            int a = 0;
            int b = chunk.length();
            List<String> lead = new ArrayList<>();
            List<String> trail = new ArrayList<>();
            while (a < b && LEADING.indexOf(chunk.charAt(a)) >= 0) {
                lead.add(String.valueOf(chunk.charAt(a++)));
            }
            while (b > a && TRAILING.indexOf(chunk.charAt(b - 1)) >= 0
                    && !(chunk.charAt(b - 1) == '.' && b - a > 1 && (isAbbreviation(chunk.substring(a, b))
                    || golds.containsKey(chunk.substring(a, b))))) {
                trail.add(0, String.valueOf(chunk.charAt(--b)));
            }
            pieces.addAll(lead);
            if (a < b) {
                pieces.add(chunk.substring(a, b));
            }
            pieces.addAll(trail);
            for (int p = 0; p < pieces.size(); p++) {
                String piece = pieces.get(p);
                String pws = p == pieces.size() - 1 ? ws : "";
                if (isPunctuation(piece)) {
                    Tok t = new Tok(piece, pws);
                    t.phonemes = piece.equals("\"") ? (p < lead.size() ? "“" : "”") : punctuation(piece);
                    t.tag = ".";
                    out.add(t);
                    currency = null;
                    continue;
                }
                List<Tok> group = new ArrayList<>();
                Matcher s = SUBTOKEN.matcher(piece);
                while (s.find()) {
                    String sub = s.group();
                    if (sub.isEmpty()) {
                        continue;
                    }
                    if (CURRENCIES.containsKey(sub)) {
                        // a currency sign stands alone (silent) and the amount starts a new word
                        currency = sub;
                        if (!group.isEmpty()) {
                            out.add(group);
                            group = new ArrayList<>();
                        }
                        Tok c = new Tok(sub, "");
                        c.phonemes = "";
                        out.add(c);
                        continue;
                    }
                    Tok t = new Tok(sub, "");
                    if (currency != null) {
                        if (Character.isDigit(sub.charAt(sub.length() - 1))) {
                            t.currency = currency;
                        } else {
                            currency = null;
                        }
                    }
                    if (!group.isEmpty()) {
                        t.head = false;
                    }
                    group.add(t);
                }
                if (group.isEmpty()) {
                    if (!out.isEmpty() && out.get(out.size() - 1) instanceof Tok last) {
                        last.ws = pws;
                    }
                    continue;
                }
                group.get(group.size() - 1).ws = pws;
                if (group.stream().allMatch(t -> t.phonemes != null)) {
                    out.addAll(group);
                } else {
                    out.add(group);
                }
            }
        }
        return out;
    }

    private static boolean isAbbreviation(String s) {
        // "e.g." and "U.S." keep their dots
        return s.matches("(\\p{L}{1,2}\\.)+\\p{L}{0,2}\\.?") && s.indexOf('.') < s.length() - 1;
    }

    private static boolean isPunctuation(String s) {
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isLetterOrDigit(c) || "$£€%&+@/'_".indexOf(c) >= 0) {
                return false;
            }
        }
        return true;
    }

    private static String punctuation(String s) {
        if (s.equals("-") || s.equals("–") || s.equals("--") || s.equals("—")) {
            return "—";
        }
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (PUNCTS.indexOf(c) >= 0 || c == '(' || c == ')') {
                out.append(c);
            } else if (c == '[' || c == '{') {
                out.append('(');
            } else if (c == ']' || c == '}') {
                out.append(')');
            } else if (c == '«') {
                out.append('“');
            } else if (c == '»') {
                out.append('”');
            }
        }
        return out.toString();
    }

    /** Guesses a coarse tag from the previous word, for dictionary entries that differ by part of speech. */
    private static void tagGroup(List<Object> tokens, int index, List<Tok> group) {
        String prev = null;
        for (int i = index - 1; i >= 0; i--) {
            Object o = tokens.get(i);
            if (o instanceof Tok t) {
                if (t.tag != null && t.tag.equals(".")) {
                    break;
                }
                continue;
            }
            @SuppressWarnings("unchecked")
            List<Tok> w = (List<Tok>) o;
            StringBuilder sb = new StringBuilder();
            for (Tok t : w) {
                sb.append(t.text);
            }
            prev = lower(sb.toString());
            break;
        }
        String tag = null;
        String first0 = lower(group.get(0).text);
        if (prev != null && (first0.equals("am") || first0.equals("pm")) && group.size() == 1
                && prev.chars().anyMatch(Character::isDigit)) {
            tag = "NN";
        } else if (prev != null && PAST_CUES.contains(prev)) {
            tag = "VBN";
        } else if (prev != null && VERB_CUES.contains(prev)) {
            tag = "VB";
        } else if (prev != null && NOUN_CUES.contains(prev)) {
            tag = "NN";
        }
        String first = group.get(0).text;
        Object after = index + 1 < tokens.size() ? tokens.get(index + 1) : null;
        if ((first0.equals("that") || first0.equals("this")) && group.size() == 1
                && (after == null || (after instanceof Tok p && ".".equals(p.tag)))) {
            tag = "DT";     // a demonstrative pronoun ends the clause: "what's that?"
        } else if (first.equals("I") && group.size() == 1) {
            tag = "PRP";
        } else if (first.equals("A") && group.size() == 1) {
            Object next = index + 1 < tokens.size() ? tokens.get(index + 1) : null;
            if (next instanceof List<?> l && !l.isEmpty() && l.get(0) instanceof Tok n
                    && !n.text.equals(n.text.toUpperCase(Locale.ROOT))) {
                tag = "DT";
            }
        }
        for (Tok t : group) {
            t.tag = tag;
        }
    }

    private static Tok merge(List<Tok> tokens, String unk) {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < tokens.size(); i++) {
            text.append(tokens.get(i).text);
            if (i < tokens.size() - 1) {
                text.append(tokens.get(i).ws);
            }
        }
        Tok last = tokens.get(tokens.size() - 1);
        Tok t = new Tok(text.toString(), last.ws);
        t.head = tokens.get(0).head;
        t.tag = tokens.get(0).tag;
        for (Tok x : tokens) {
            if (x.currency != null) {
                t.currency = x.currency;
            }
        }
        if (unk != null) {
            StringBuilder ps = new StringBuilder();
            for (Tok x : tokens) {
                if (x.prespace && !ps.isEmpty() && !Character.isWhitespace(ps.charAt(ps.length() - 1))
                        && x.phonemes != null && !x.phonemes.isEmpty()) {
                    ps.append(' ');
                }
                ps.append(x.phonemes == null ? unk : x.phonemes);
            }
            t.phonemes = ps.toString();
        }
        return t;
    }

    private static Context context(Context ctx, String ps, Tok token) {
        Boolean vowel = ctx.futureVowel();
        if (ps != null && !ps.isEmpty()) {
            for (int i = 0; i < ps.length(); i++) {
                char c = ps.charAt(i);
                if (VOWELS.indexOf(c) >= 0 || CONSONANTS.indexOf(c) >= 0 || NON_QUOTE_PUNCTS.indexOf(c) >= 0) {
                    vowel = NON_QUOTE_PUNCTS.indexOf(c) >= 0 ? null : VOWELS.indexOf(c) >= 0;
                    break;
                }
            }
        }
        String t = token.text;
        boolean isTo = t.equals("to") || t.equals("To") || (t.equals("TO") && "TO".equals(token.tag));
        return new Context(vowel, isTo);
    }

    /** Joins the phonemes of a word's parts, softening all but the strongest stresses. */
    private static void resolve(List<Tok> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tokens.size(); i++) {
            sb.append(tokens.get(i).text);
            if (i < tokens.size() - 1) {
                sb.append(tokens.get(i).ws);
            }
        }
        String text = sb.toString();
        Set<Integer> classes = new java.util.HashSet<>();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (SUBTOKEN_JUNKS.indexOf(c) < 0) {
                classes.add(Character.isLetter(c) ? 0 : Character.isDigit(c) ? 1 : 2);
            }
        }
        boolean prespace = text.contains(" ") || text.contains("/") || classes.size() > 1;
        for (int i = 0; i < tokens.size(); i++) {
            Tok t = tokens.get(i);
            if (t.phonemes == null) {
                if (i == tokens.size() - 1 && t.text.length() == 1 && NON_QUOTE_PUNCTS.contains(t.text)) {
                    t.phonemes = t.text;
                } else if (allIn(t.text, SUBTOKEN_JUNKS)) {
                    t.phonemes = "";
                }
            } else if (i > 0) {
                t.prespace = prespace;
            }
        }
        if (prespace) {
            return;
        }
        List<int[]> idx = new ArrayList<>();   // {hasPrimary, weight, index}
        for (int i = 0; i < tokens.size(); i++) {
            String ps = tokens.get(i).phonemes;
            if (ps != null && !ps.isEmpty()) {
                int weight = 0;
                for (int k = 0; k < ps.length(); k++) {
                    weight += DIPHTHONGS.indexOf(ps.charAt(k)) >= 0 ? 2 : 1;
                }
                idx.add(new int[] {ps.contains(PRIMARY) ? 1 : 0, weight, i});
            }
        }
        if (idx.size() == 2 && tokens.get(idx.get(0)[2]).text.length() == 1) {
            Tok t = tokens.get(idx.get(1)[2]);
            t.phonemes = stress(t.phonemes, -0.5);
            return;
        }
        int primaries = 0;
        for (int[] x : idx) {
            primaries += x[0];
        }
        if (idx.size() < 2 || primaries <= (idx.size() + 1) / 2) {
            return;
        }
        idx.sort((x, y) -> x[0] != y[0] ? Integer.compare(x[0], y[0])
                : x[1] != y[1] ? Integer.compare(x[1], y[1]) : Integer.compare(x[2], y[2]));
        for (int k = 0; k < idx.size() / 2; k++) {
            Tok t = tokens.get(idx.get(k)[2]);
            t.phonemes = stress(t.phonemes, -0.5);
        }
    }

    // ------------------------------------------------------------------ lexicon

    private Found lexicon(Tok t, Context ctx) {
        String word = Normalizer.normalize(t.text, Normalizer.Form.NFKC);
        Double stress = word.equals(lower(word)) ? null : word.equals(upper(word)) ? 2.0 : 0.5;
        Found f = getWord(word, t.tag, stress, ctx);
        if (f != null) {
            return new Found(appendCurrency(f.ps, t.currency), f.rating);
        }
        if (isNumber(word, t.head)) {
            return number(word, t.currency, t.head);
        }
        if (!lexiconChars(word)) {
            return null;
        }
        if (!word.equals(lower(word)) && (word.equals(upper(word))
                || word.substring(1).equals(lower(word.substring(1))))) {
            f = getWord(lower(word), t.tag, stress, ctx);
            if (f != null) {
                return new Found(appendCurrency(f.ps, t.currency), f.rating);
            }
        }
        return null;
    }

    private static boolean lexiconChars(String w) {
        for (int i = 0; i < w.length(); i++) {
            char c = w.charAt(i);
            if (!(c == '\'' || c == '-' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) {
                return false;
            }
        }
        return true;
    }

    private Found nnp(String word) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (!Character.isLetter(c)) {
                continue;
            }
            Object v = golds.get(upper(String.valueOf(c)));
            if (!(v instanceof String s)) {
                return null;
            }
            sb.append(s);
        }
        String ps = stress(sb.toString(), 0);
        int cut = ps.lastIndexOf(SECONDARY);
        if (cut >= 0) {
            ps = ps.substring(0, cut) + PRIMARY + ps.substring(cut + 1);
        }
        return new Found(ps, 3);
    }

    private Found specialCase(String word, String tag, Double stress, Context ctx) {
        if (SYMBOLS.containsKey(word)) {
            return lookup(SYMBOLS.get(word), null, null, ctx);
        }
        String stripped = word.replaceAll("^\\.+|\\.+$", "");
        if (stripped.contains(".") && isAlpha(word.replace(".", "")) && maxPart(word) < 3) {
            return nnp(word);
        }
        if (word.equals("a") || (word.equals("A") && "DT".equals(tag))) {
            return new Found("ɐ", 4);
        }
        if (word.equals("am") || word.equals("Am") || word.equals("AM")) {
            if (tag != null && tag.startsWith("NN")) {
                return nnp(word);
            }
            if (ctx.futureVowel() == null || !word.equals("am") || (stress != null && stress > 0)) {
                return new Found((String) golds.get("am"), 4);
            }
            return new Found("ɐm", 4);
        }
        if (word.equals("an") || word.equals("An") || word.equals("AN")) {
            if (word.equals("AN") && tag != null && tag.startsWith("NN")) {
                return nnp(word);
            }
            return new Found("ɐn", 4);
        }
        if (word.equals("I") && "PRP".equals(tag)) {
            return new Found(SECONDARY + "I", 4);
        }
        if (word.equals("to") || word.equals("To") || (word.equals("TO") && "TO".equals(tag))) {
            Boolean v = ctx.futureVowel();
            return new Found(v == null ? (String) golds.get("to") : v ? "tʊ" : "tə", 4);
        }
        if (word.equals("the") || word.equals("The") || (word.equals("THE") && "DT".equals(tag))) {
            return new Found(Boolean.TRUE.equals(ctx.futureVowel()) ? "ði" : "ðə", 4);
        }
        if (word.matches("(?i)vs\\.?")) {
            return lookup("versus", null, null, ctx);
        }
        if (word.equals("used") || word.equals("Used") || word.equals("USED")) {
            @SuppressWarnings("unchecked")
            Map<String, String> used = (Map<String, String>) golds.get("used");
            return new Found(ctx.futureTo() ? used.get("VBD") : used.get("DEFAULT"), 4);
        }
        return null;
    }

    private static int maxPart(String word) {
        int max = 0;
        for (String p : word.split("\\.")) {
            max = Math.max(max, p.length());
        }
        return max;
    }

    private static String parentTag(String tag) {
        if (tag == null) {
            return null;
        }
        if (tag.startsWith("VB")) {
            return "VERB";
        }
        if (tag.startsWith("NN")) {
            return "NOUN";
        }
        if (tag.startsWith("ADV") || tag.startsWith("RB")) {
            return "ADV";
        }
        if (tag.startsWith("ADJ") || tag.startsWith("JJ")) {
            return "ADJ";
        }
        return tag;
    }

    boolean isKnown(String word) {
        if (golds.containsKey(word) || SYMBOLS.containsKey(word) || silvers.containsKey(word)) {
            return true;
        }
        if (!isAlpha(word) || !lexiconChars(word)) {
            return false;
        }
        if (word.length() == 1) {
            return true;
        }
        if (word.equals(upper(word)) && golds.containsKey(lower(word))) {
            return true;
        }
        return word.substring(1).equals(upper(word.substring(1)));
    }

    private Found lookup(String word, String tag, Double stress, Context ctx) {
        boolean isNnp = false;
        if (word.equals(upper(word)) && !golds.containsKey(word)) {
            word = lower(word);
            isNnp = "NNP".equals(tag);
        }
        Object ps = golds.get(word);
        int rating = 4;
        if (ps == null && !isNnp) {
            ps = silvers.get(word);
            rating = 3;
        }
        String s;
        if (ps instanceof Map<?, ?> m) {
            String key = tag;
            if (ctx != null && ctx.futureVowel() == null && m.containsKey("None")) {
                key = "None";
            } else if (!m.containsKey(key)) {
                key = parentTag(tag);
            }
            s = (String) (m.containsKey(key) ? m.get(key) : m.get("DEFAULT"));
        } else {
            s = (String) ps;
        }
        if (s == null || (isNnp && !s.contains(PRIMARY))) {
            Found f = nnp(word);
            if (f != null) {
                return f;
            }
        }
        return s == null ? null : new Found(stress(s, stress), rating);
    }

    // lookup can return null phonemes, as Python's (None, None)
    private Found lookupOrNull(String word, String tag, Double stress, Context ctx) {
        return lookup(word, tag, stress, ctx);
    }

    private String plural(String stem) {
        if (stem == null || stem.isEmpty()) {
            return null;
        }
        char c = stem.charAt(stem.length() - 1);
        if ("ptkfθ".indexOf(c) >= 0) {
            return stem + "s";
        }
        if ("szʃʒʧʤ".indexOf(c) >= 0) {
            return stem + (isBritish ? "ɪ" : "ᵻ") + "z";
        }
        return stem + "z";
    }

    private Found stemS(String word, String tag, Double stress, Context ctx) {
        String stem;
        if (word.length() > 2 && word.endsWith("s") && !word.endsWith("ss")
                && isKnown(word.substring(0, word.length() - 1))) {
            stem = word.substring(0, word.length() - 1);
        } else if ((word.endsWith("'s") || (word.length() > 4 && word.endsWith("es")))
                && isKnown(word.substring(0, word.length() - 2))) {
            stem = word.substring(0, word.length() - 2);
        } else if (word.length() > 4 && word.endsWith("ies") && isKnown(word.substring(0, word.length() - 3) + "y")) {
            stem = word.substring(0, word.length() - 3) + "y";
        } else {
            return null;
        }
        Found f = lookupOrNull(stem, tag, stress, ctx);
        return f == null ? null : new Found(plural(f.ps), f.rating);
    }

    private String past(String stem) {
        if (stem == null || stem.isEmpty()) {
            return null;
        }
        char c = stem.charAt(stem.length() - 1);
        if ("pkfθʃsʧ".indexOf(c) >= 0) {
            return stem + "t";
        }
        if (c == 'd') {
            return stem + (isBritish ? "ɪ" : "ᵻ") + "d";
        }
        if (c != 't') {
            return stem + "d";
        }
        if (isBritish || stem.length() < 2) {
            return stem + "ɪd";
        }
        if (US_TAUS.indexOf(stem.charAt(stem.length() - 2)) >= 0) {
            return stem.substring(0, stem.length() - 1) + "ɾᵻd";
        }
        return stem + "ᵻd";
    }

    private Found stemEd(String word, String tag, Double stress, Context ctx) {
        String stem;
        if (word.endsWith("d") && !word.endsWith("dd") && isKnown(word.substring(0, word.length() - 1))) {
            stem = word.substring(0, word.length() - 1);
        } else if (word.endsWith("ed") && !word.endsWith("eed") && isKnown(word.substring(0, word.length() - 2))) {
            stem = word.substring(0, word.length() - 2);
        } else {
            return null;
        }
        Found f = lookupOrNull(stem, tag, stress, ctx);
        return f == null ? null : new Found(past(f.ps), f.rating);
    }

    private String ing(String stem) {
        if (stem == null || stem.isEmpty()) {
            return null;
        }
        if (isBritish) {
            char c = stem.charAt(stem.length() - 1);
            if (c == 'ə' || c == 'ː') {
                return null;
            }
        } else if (stem.length() > 1 && stem.charAt(stem.length() - 1) == 't'
                && US_TAUS.indexOf(stem.charAt(stem.length() - 2)) >= 0) {
            return stem.substring(0, stem.length() - 1) + "ɾɪŋ";
        }
        return stem + "ɪŋ";
    }

    private static final Pattern DOUBLED_ING = Pattern.compile("([bcdgklmnprstvxz])\\1ing$|cking$");

    private Found stemIng(String word, String tag, Double stress, Context ctx) {
        String stem;
        if (word.endsWith("ing") && isKnown(word.substring(0, word.length() - 3))) {
            stem = word.substring(0, word.length() - 3);
        } else if (word.endsWith("ing") && isKnown(word.substring(0, word.length() - 3) + "e")) {
            stem = word.substring(0, word.length() - 3) + "e";
        } else if (DOUBLED_ING.matcher(word).find() && isKnown(word.substring(0, word.length() - 4))) {
            stem = word.substring(0, word.length() - 4);
        } else {
            return null;
        }
        Found f = lookupOrNull(stem, tag, stress, ctx);
        if (f == null) {
            return null;
        }
        String ps = ing(f.ps);
        return ps == null ? null : new Found(ps, f.rating);
    }

    private Found getWord(String word, String tag, Double stress, Context ctx) {
        Found f = specialCase(word, tag, stress, ctx);
        if (f != null && f.ps != null) {
            return f;
        }
        if (isKnown(word)) {
            return lookup(word, tag, stress, ctx);
        }
        if (word.endsWith("s'") && isKnown(word.substring(0, word.length() - 2) + "'s")) {
            return lookup(word.substring(0, word.length() - 2) + "'s", tag, stress, ctx);
        }
        if (word.endsWith("'") && isKnown(word.substring(0, word.length() - 1))) {
            return lookup(word.substring(0, word.length() - 1), tag, stress, ctx);
        }
        f = stemS(word, tag, stress, ctx);
        if (f != null && f.ps != null) {
            return f;
        }
        f = stemEd(word, tag, stress, ctx);
        if (f != null && f.ps != null) {
            return f;
        }
        f = stemIng(word, tag, stress == null ? Double.valueOf(0.5) : stress, ctx);
        return f != null && f.ps != null ? f : null;
    }

    // ------------------------------------------------------------------ numbers

    private static boolean isNumber(String word, boolean head) {
        boolean digit = false;
        for (int i = 0; i < word.length(); i++) {
            digit |= Character.isDigit(word.charAt(i));
        }
        if (!digit) {
            return false;
        }
        for (String s : new String[] {"ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"}) {
            if (word.endsWith(s)) {
                word = word.substring(0, word.length() - s.length());
                break;
            }
        }
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (!(Character.isDigit(c) || c == ',' || c == '.' || (head && i == 0 && c == '-'))) {
                return false;
            }
        }
        return true;
    }

    private static boolean isCurrency(String word) {
        if (!word.contains(".")) {
            return true;
        }
        if (word.indexOf('.') != word.lastIndexOf('.')) {
            return false;
        }
        return word.substring(word.indexOf('.') + 1).length() < 3;
    }

    private Found number(String word, String currency, boolean head) {
        Matcher sm = Pattern.compile("[a-z']+$").matcher(word);
        String suffix = sm.find() ? sm.group() : null;
        if (suffix != null) {
            word = word.substring(0, word.length() - suffix.length());
        }
        List<Found> result = new ArrayList<>();
        if (word.startsWith("-")) {
            result.add(lookup("minus", null, null, null));
            word = word.substring(1);
        }
        if (!head && !word.contains(".")) {
            String num = word.replace(",", "");
            if (num.isEmpty()) {
                return null;
            }
            if (num.charAt(0) == '0' || num.length() > 3) {
                for (char c : num.toCharArray()) {
                    words(NumberWords.cardinal(c - '0'), result);
                }
            } else if (num.length() == 3 && !num.endsWith("00")) {
                words(NumberWords.cardinal(num.charAt(0) - '0'), result);
                if (num.charAt(1) == '0') {
                    result.add(lookup("O", null, -2.0, null));
                    words(NumberWords.cardinal(num.charAt(2) - '0'), result);
                } else {
                    words(NumberWords.cardinal(Integer.parseInt(num.substring(1))), result);
                }
            } else {
                words(NumberWords.cardinal(Long.parseLong(num)), result);
            }
        } else if (word.indexOf('.') != word.lastIndexOf('.') || !head) {
            for (String num : word.replace(",", "").split("\\.")) {
                if (num.isEmpty()) {
                    continue;
                }
                boolean nonZeroTail = false;
                for (int i = 1; i < num.length(); i++) {
                    nonZeroTail |= num.charAt(i) != '0';
                }
                if (num.charAt(0) == '0' || (num.length() != 2 && nonZeroTail)) {
                    for (char c : num.toCharArray()) {
                        words(NumberWords.cardinal(c - '0'), result);
                    }
                } else {
                    words(NumberWords.cardinal(new BigInteger(num)), result);
                }
            }
        } else if (currency != null && CURRENCIES.containsKey(currency) && isCurrency(word)) {
            String[] units = CURRENCIES.get(currency);
            String[] parts = word.replace(",", "").split("\\.", -1);
            List<long[]> pairs = new ArrayList<>();
            for (int i = 0; i < Math.min(parts.length, 2); i++) {
                pairs.add(new long[] {parts[i].isEmpty() ? 0 : Long.parseLong(parts[i]), i});
            }
            if (pairs.size() > 1) {
                if (pairs.get(1)[0] == 0) {
                    pairs = pairs.subList(0, 1);
                } else if (pairs.get(0)[0] == 0) {
                    pairs = pairs.subList(1, 2);
                }
            }
            for (int i = 0; i < pairs.size(); i++) {
                long n = pairs.get(i)[0];
                String unit = units[(int) pairs.get(i)[1]];
                if (i > 0) {
                    result.add(lookup("and", null, null, null));
                }
                words(NumberWords.cardinal(n), result);
                result.add(Math.abs(n) != 1 && !unit.equals("pence") ? stemS(unit + "s", null, null, null)
                        : lookup(unit, null, null, null));
            }
        } else {
            String text;
            String digits = word.replace(",", "");
            if (digits.isEmpty()) {
                return null;
            }
            if (word.chars().allMatch(Character::isDigit)) {
                if (suffix != null && ORDINALS.contains(suffix)) {
                    text = NumberWords.ordinal(new BigInteger(word));
                } else if (result.isEmpty() && word.length() == 4) {
                    text = NumberWords.year(Integer.parseInt(word));
                } else {
                    text = NumberWords.cardinal(new BigInteger(word));
                }
            } else if (!word.contains(".")) {
                text = suffix != null && ORDINALS.contains(suffix) ? NumberWords.ordinal(new BigInteger(digits))
                        : NumberWords.cardinal(new BigInteger(digits));
            } else if (digits.charAt(0) == '.') {
                StringBuilder sb = new StringBuilder("point");
                for (char c : digits.substring(1).toCharArray()) {
                    sb.append(' ').append(NumberWords.cardinal(c - '0'));
                }
                text = sb.toString();
            } else {
                text = NumberWords.decimal(digits);
            }
            words(text, result);
        }
        result.removeIf(java.util.Objects::isNull);
        if (result.isEmpty()) {
            return null;
        }
        StringBuilder ps = new StringBuilder();
        int rating = Integer.MAX_VALUE;
        for (Found f : result) {
            if (!ps.isEmpty()) {
                ps.append(' ');
            }
            ps.append(f.ps);
            rating = Math.min(rating, f.rating);
        }
        String out = ps.toString();
        if ("s".equals(suffix) || "'s".equals(suffix)) {
            out = plural(out);
        } else if ("ed".equals(suffix) || "'d".equals(suffix)) {
            out = past(out);
        } else if ("ing".equals(suffix)) {
            out = ing(out);
        }
        return new Found(out, rating);
    }

    private void words(String text, List<Found> result) {
        for (String w : text.split("[^a-z]+")) {
            if (w.isEmpty() || w.equals("and")) {
                continue;
            }
            result.add(lookup(w, null, w.equals("point") ? -2.0 : null, null));
        }
    }

    private String appendCurrency(String ps, String currency) {
        if (currency == null || !CURRENCIES.containsKey(currency)) {
            return ps;
        }
        Found f = stemS(CURRENCIES.get(currency)[0] + "s", null, null, null);
        return f == null ? ps : ps + " " + f.ps;
    }

    // ------------------------------------------------------------------ stress

    /**
     * Applies a stress level: below -1 removes stress, -1 (or 0 with a primary) demotes
     * primaries, 0 to 1 adds a secondary to unstressed words, 1 and up promotes a lone
     * secondary, above 1 adds a primary.
     */
    static String stress(String ps, Double stress) {
        if (stress == null || ps == null) {
            return ps;
        }
        double s = stress;
        if (s < -1) {
            return ps.replace(PRIMARY, "").replace(SECONDARY, "");
        }
        if (s == -1 || ((s == 0 || s == -0.5) && ps.contains(PRIMARY))) {
            return ps.replace(SECONDARY, "").replace(PRIMARY, SECONDARY);
        }
        boolean none = !ps.contains(PRIMARY) && !ps.contains(SECONDARY);
        if ((s == 0 || s == 0.5 || s == 1) && none) {
            return hasVowel(ps) ? restress(SECONDARY + ps) : ps;
        }
        if (s >= 1 && !ps.contains(PRIMARY) && ps.contains(SECONDARY)) {
            return ps.replace(SECONDARY, PRIMARY);
        }
        if (s > 1 && none) {
            return hasVowel(ps) ? restress(PRIMARY + ps) : ps;
        }
        return ps;
    }

    static String stress(String ps, double stress) {
        return stress(ps, Double.valueOf(stress));
    }

    private static boolean hasVowel(String ps) {
        for (int i = 0; i < ps.length(); i++) {
            if (VOWELS.indexOf(ps.charAt(i)) >= 0) {
                return true;
            }
        }
        return false;
    }

    /** Moves each stress mark to just before the next vowel. */
    private static String restress(String ps) {
        int n = ps.length();
        double[] pos = new double[n];
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            pos[i] = i;
            order[i] = i;
            if (STRESSES.indexOf(ps.charAt(i)) >= 0) {
                for (int j = i; j < n; j++) {
                    if (VOWELS.indexOf(ps.charAt(j)) >= 0) {
                        pos[i] = j - 0.5;
                        break;
                    }
                }
            }
        }
        java.util.Arrays.sort(order, (a, b) -> Double.compare(pos[a], pos[b]));
        StringBuilder out = new StringBuilder();
        for (int i : order) {
            out.append(ps.charAt(i));
        }
        return out.toString();
    }

    // ------------------------------------------------------------------ helpers

    private static boolean allIn(String s, String set) {
        for (int i = 0; i < s.length(); i++) {
            if (set.indexOf(s.charAt(i)) < 0) {
                return false;
            }
        }
        return true;
    }

    private static boolean isAlpha(String s) {
        if (s.isEmpty()) {
            return false;
        }
        for (int i = 0; i < s.length(); i++) {
            if (!Character.isLetter(s.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    static String lower(String s) {
        return s.toLowerCase(Locale.ROOT);
    }

    private static String upper(String s) {
        return s.toUpperCase(Locale.ROOT);
    }

    private static String capitalize(String s) {
        return s.isEmpty() ? s : s.substring(0, 1).toUpperCase(Locale.ROOT) + s.substring(1).toLowerCase(Locale.ROOT);
    }

    /** The phonemes of a dictionary word, or {@code null}; for the letter-to-sound fallback. */
    String known(String word) {
        if (!golds.containsKey(word) && !silvers.containsKey(word)) {
            return null;
        }
        Found f = lookup(word, null, null, null);
        return f == null ? null : f.ps;
    }
}
