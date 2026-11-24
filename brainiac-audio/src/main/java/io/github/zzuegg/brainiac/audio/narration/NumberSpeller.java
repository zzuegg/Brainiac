package io.github.zzuegg.brainiac.audio.narration;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Numbers as the words a narrator says: cardinals, ordinals, decimals, durations, clock
 * times and arithmetic operators, in English or German. Speech models read digits
 * inconsistently ("1:23.4" may come out as "one twenty-three point four" or not at all), so a
 * line is spelled out before it is spoken, and the spelled words are also what speech
 * recognition is checked against.
 *
 * <pre>{@code
 * NumberSpeller de = NumberSpeller.of("de");
 * de.cardinal(21);              // "einundzwanzig"
 * de.ordinal(3, "en");          // "dritten", as in "auf dem dritten Platz"
 * de.duration(83.42, 2);        // "eine Minute und dreiundzwanzig Komma vier zwei Sekunden"
 * }</pre>
 */
public sealed interface NumberSpeller permits EnglishNumbers, GermanNumbers {

    /**
     * The speller of a language.
     *
     * @param language a code or name: {@code en}, {@code english}, {@code de}, {@code german}
     * @return its speller
     * @throws IllegalArgumentException for other languages
     */
    static NumberSpeller of(String language) {
        String l = language.toLowerCase(Locale.ROOT);
        if (l.startsWith("en")) {
            return EnglishNumbers.INSTANCE;
        }
        if (l.startsWith("de") || l.equals("german") || l.equals("deutsch")) {
            return GermanNumbers.INSTANCE;
        }
        throw new IllegalArgumentException("no number words for " + language + "; English and German are known");
    }

    /**
     * The language's code.
     *
     * @return {@code en} or {@code de}
     */
    String language();

    /**
     * The language's name, as Qwen3-TTS takes it.
     *
     * @return {@code english} or {@code german}
     */
    String languageName();

    /**
     * A whole number: 21 as "twenty-one" or "einundzwanzig".
     *
     * @param n the number
     * @return its words
     */
    String cardinal(long n);

    /**
     * A whole number before a noun, where 1 agrees with it: German "eine Runde", "einen Stern".
     *
     * @param n the number
     * @param one the word for exactly 1 (or -1), or {@code null} for the plain cardinal
     * @return its words
     */
    default String cardinal(long n, String one) {
        if (one != null && Math.abs(n) == 1) {
            return n < 0 ? minus() + " " + one : one;
        }
        return cardinal(n);
    }

    /**
     * An ordinal: 3 as "third", or German "dritt" plus an ending.
     *
     * @param n the number, 1 or more
     * @param ending the German ending ({@code e}, {@code en}, {@code er}, {@code es}, {@code em}),
     *     or {@code null} for {@code e}; English ignores it
     * @return its words
     */
    String ordinal(long n, String ending);

    /**
     * A decimal, the digits after the point one by one: 3.14 as "three point one four".
     *
     * @param x the number, as written (trailing zeros are said)
     * @return its words
     */
    String decimal(BigDecimal x);

    /**
     * A number rounded to some decimal places.
     *
     * @param x the number
     * @param places the decimal places, 0 for a whole number
     * @return its words
     */
    default String number(double x, int places) {
        BigDecimal d = BigDecimal.valueOf(x).setScale(Math.max(0, places), RoundingMode.HALF_UP);
        return places <= 0 ? cardinal(d.longValueExact()) : decimal(d);
    }

    /**
     * Digits one at a time: "007" as "zero zero seven".
     *
     * @param digits the digits
     * @return their words
     */
    String digits(String digits);

    /**
     * A length of time: 83.42 seconds as "one minute and twenty-three point four two seconds".
     *
     * @param seconds the time, 0 or more
     * @param places decimal places of the seconds
     * @return its words
     */
    String duration(double seconds, int places);

    /**
     * A time of day: 14:05 as "two oh five" or "vierzehn Uhr fünf".
     *
     * @param hour 0 to 23
     * @param minute 0 to 59
     * @return its words
     */
    String clock(int hour, int minute);

    /**
     * An arithmetic or comparison symbol as words: {@code +} as "plus", {@code ×} as "times"
     * or "mal", {@code =} as "equals" or "ist", {@code (} as "open bracket".
     *
     * @param symbol the symbol
     * @return its words, or {@code null} if it is not an operator
     */
    String operator(String symbol);

    /**
     * The word for a negative sign.
     *
     * @return "minus"
     */
    default String minus() {
        return "minus";
    }

    /**
     * Picks a noun's form for a count.
     *
     * @param n the count
     * @param one the form for 1 ("lap", "Runde")
     * @param other the form for other counts ("laps", "Runden")
     * @return {@code one} for 1 and -1, otherwise {@code other}
     */
    default String plural(double n, String one, String other) {
        return Math.abs(n) == 1 ? one : other;
    }

    /**
     * Spells the numbers and operators in free text, such as a transcription: "12 + 7 = 19"
     * becomes "twelve plus seven equals nineteen". Decimals use the language's separator
     * ({@code .} in English, {@code ,} in German).
     *
     * @param text the text
     * @return the text with words for numbers
     */
    default String spell(String text) {
        char sep = language().equals("de") ? ',' : '.';
        Pattern p = Pattern.compile("(?<![\\p{L}\\d])(?:(\\d{1,2}):(\\d\\d)(?!\\d)|(\\d+)(?:\\Q" + sep + "\\E(\\d+))?(?!\\d))"
                + "|(?<=\\s|^)([+×*·÷/=<>≤≥≠^%:])(?=\\s|$)|(?<=\\d)(%)");
        Matcher m = p.matcher(text);
        StringBuilder out = new StringBuilder();
        while (m.find()) {
            String words;
            if (m.group(1) != null) {
                words = clock(Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2)));
            } else if (m.group(5) != null || m.group(6) != null) {
                String sym = m.group(5) != null ? m.group(5) : m.group(6);
                words = operator(sym);
                if (words == null) {
                    words = sym;
                }
                if (m.group(6) != null) {
                    words = " " + words;
                }
            } else if (m.group(4) != null) {
                words = decimal(new BigDecimal(m.group(3) + "." + m.group(4)));
            } else if (m.group(3).length() > 18) {
                words = digits(m.group(3));
            } else {
                words = cardinal(Long.parseLong(m.group(3)));
            }
            m.appendReplacement(out, Matcher.quoteReplacement(words));
        }
        m.appendTail(out);
        return out.toString();
    }

    /**
     * The parts that numbers up to 9,999 are built from, each said as a piece of its own: 0
     * to 19, the tens, the hundreds and the thousands, and in German the units joined to "und"
     * ("einund", "zweiund", …) that come before a ten, as in "einund|zwanzig".
     *
     * @return the parts
     */
    default java.util.List<String> numberParts() {
        java.util.Set<String> out = new java.util.LinkedHashSet<>();
        for (int n = 0; n < 20; n++) {
            out.add(cardinal(n));
        }
        for (int t = 20; t < 100; t += 10) {
            out.add(cardinal(t));
        }
        for (int h = 100; h < 1000; h += 100) {
            out.add(cardinal(h));
        }
        for (int k = 1000; k < 10000; k += 1000) {
            out.add(cardinal(k));
        }
        if (language().equals("de")) {
            for (int u = 1; u < 10; u++) {
                out.add((u == 1 ? "ein" : cardinal(u)) + "und");
            }
        }
        return java.util.List.copyOf(out);
    }

    /**
     * Splits a spelled number into the fewest {@link #numberParts()}, such as
     * "zweihunderteinundfünfzig" into "zweihundert", "einund", "fünfzig", or "two hundred
     * fifty-one" into "two hundred", "fifty", "one".
     *
     * @param words the spelled number
     * @param parts the parts to build it from
     * @return the parts in order, or {@code null} if it cannot be built from them
     */
    static java.util.List<String> split(String words, java.util.Collection<String> parts) {
        String t = words.toLowerCase(Locale.ROOT);
        int n = t.length();
        int[] count = new int[n + 1];
        String[] via = new String[n + 1];
        int[] from = new int[n + 1];
        java.util.Arrays.fill(count, Integer.MAX_VALUE);
        count[0] = 0;
        for (int i = 0; i < n; i++) {
            if (count[i] == Integer.MAX_VALUE) {
                continue;
            }
            int at = i;
            while (at < n && (t.charAt(at) == ' ' || t.charAt(at) == '-')) {
                at++;
            }
            if (at == n && count[i] < count[n]) {
                count[n] = count[i];
                via[n] = null;
                from[n] = i;
            }
            for (String p : parts) {
                String q = p.toLowerCase(Locale.ROOT);
                if (!q.isEmpty() && t.startsWith(q, at)) {
                    int to = at + q.length();
                    if (count[i] + 1 < count[to]) {
                        count[to] = count[i] + 1;
                        via[to] = p;
                        from[to] = i;
                    }
                }
            }
        }
        if (count[n] == Integer.MAX_VALUE) {
            return null;
        }
        java.util.LinkedList<String> out = new java.util.LinkedList<>();
        for (int at = n; at > 0; at = from[at]) {
            if (via[at] != null) {
                out.addFirst(via[at]);
            }
        }
        return out;
    }
}
