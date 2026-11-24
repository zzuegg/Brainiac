package io.github.zzuegg.brainiac.audio.narration;

import java.math.BigDecimal;
import java.time.Duration;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * A narrator's line with holes for values, such as {@code "Du bist auf dem {n:ordinal:en} Platz!"}
 * or {@code "Noch {n} {n|Runde|Runden}!"}. A hole is
 * <ul>
 * <li>{@code {name}}: a number (whole numbers as cardinals, others as decimals) or text;
 * <li>{@code {name:format}} or {@code {name:format:arg}} with the formats
 *     {@code cardinal} (arg: the word for 1, such as {@code eine}), {@code ordinal} (arg: the
 *     German ending, such as {@code en}), {@code decimal} (arg: decimal places), {@code digits},
 *     {@code time} (a duration in seconds or a {@link Duration}; arg: decimal places),
 *     {@code clock} (a {@link LocalTime} or {@code "14:05"}), {@code math} (an expression such
 *     as {@code "12 + 7 = 19"}, read by {@link MathReader}) and {@code text};
 * <li>{@code {name|one|other}}: the noun form that agrees with the number {@code name}.
 * </ul>
 *
 * <p>{@link #render} spells a line out for given values. The spelled line knows which words
 * came from which value, so a {@link Narrator} can splice it from recorded pieces.
 *
 * @param source the template as written
 * @param parts its literal text and holes, in order
 */
public record LineTemplate(String source, List<Part> parts) {

    /**
     * Copies the parts.
     *
     * @param source the template
     * @param parts its parts
     */
    public LineTemplate {
        Objects.requireNonNull(source, "source");
        parts = List.copyOf(parts);
    }

    /** Literal text or a hole. */
    public sealed interface Part permits Literal, Slot {
    }

    /**
     * Literal text, spaces and punctuation included.
     *
     * @param text the text
     */
    public record Literal(String text) implements Part {
    }

    /**
     * A hole for a value.
     *
     * @param name the value's name
     * @param format how to say it ({@code cardinal}, {@code ordinal}, ...), or {@code null} to decide by the value
     * @param arg the format's argument, or {@code null}
     * @param one for a plural hole, the form for 1, otherwise {@code null}
     * @param other for a plural hole, the form for other counts
     */
    public record Slot(String name, String format, String arg, String one, String other) implements Part {
    }

    /**
     * Parses a template.
     *
     * @param source the template, such as {@code "Noch {n} {n|Runde|Runden}!"}
     * @return the template
     * @throws IllegalArgumentException if a brace is not closed or a hole is empty
     */
    public static LineTemplate parse(String source) {
        List<Part> parts = new ArrayList<>();
        StringBuilder text = new StringBuilder();
        int i = 0;
        while (i < source.length()) {
            char c = source.charAt(i);
            if (c != '{') {
                text.append(c);
                i++;
                continue;
            }
            int close = source.indexOf('}', i);
            if (close < 0) {
                throw new IllegalArgumentException("unclosed { at " + i + " in: " + source);
            }
            if (!text.isEmpty()) {
                parts.add(new Literal(text.toString()));
                text.setLength(0);
            }
            String inner = source.substring(i + 1, close).trim();
            if (inner.isEmpty()) {
                throw new IllegalArgumentException("empty {} at " + i + " in: " + source);
            }
            if (inner.contains("|")) {
                String[] p = inner.split("\\|", -1);
                if (p.length != 3) {
                    throw new IllegalArgumentException("a plural hole is {name|one|other}, got {" + inner + "}");
                }
                parts.add(new Slot(p[0].trim(), "plural", null, p[1], p[2]));
            } else {
                String[] p = inner.split(":", 3);
                parts.add(new Slot(p[0].trim(), p.length > 1 ? p[1].trim() : null, p.length > 2 ? p[2] : null, null,
                        null));
            }
            i = close + 1;
        }
        if (!text.isEmpty()) {
            parts.add(new Literal(text.toString()));
        }
        return new LineTemplate(source, parts);
    }

    /**
     * The names of the values the template takes, in order of first use.
     *
     * @return the names
     */
    public Set<String> names() {
        Set<String> out = new LinkedHashSet<>();
        for (Part p : parts) {
            if (p instanceof Slot s) {
                out.add(s.name());
            }
        }
        return out;
    }

    /**
     * Spells the line for some values.
     *
     * @param values a value for each {@linkplain #names() name}
     * @param speller the language's number words
     * @return the spelled line
     * @throws IllegalArgumentException if a value is missing or does not fit its format
     */
    public Line render(Map<String, ?> values, NumberSpeller speller) {
        List<Piece> pieces = new ArrayList<>();
        for (Part p : parts) {
            switch (p) {
                case Literal l -> pieces.add(new Piece(l.text(), null));
                case Slot s -> {
                    if (!values.containsKey(s.name())) {
                        throw new IllegalArgumentException("no value for {" + s.name() + "} in: " + source);
                    }
                    pieces.add(new Piece(spell(s, values.get(s.name()), speller), s.name()));
                }
            }
        }
        return new Line(source, pieces);
    }

    static String spell(Slot s, Object v, NumberSpeller sp) {
        String f = s.format() == null ? "" : s.format();
        String arg = s.arg();
        return switch (f) {
            case "plural" -> sp.plural(number(v, s).doubleValue(), s.one(), s.other());
            case "ordinal" -> sp.ordinal(whole(v, s), arg);
            case "decimal" -> arg == null ? sp.decimal(number(v, s)) : sp.number(number(v, s).doubleValue(),
                    Integer.parseInt(arg.trim()));
            case "digits" -> sp.digits(v.toString());
            case "time" -> sp.duration(v instanceof Duration d ? d.toNanos() / 1e9 : number(v, s).doubleValue(),
                    arg == null ? 0 : Integer.parseInt(arg.trim()));
            case "clock" -> {
                LocalTime t = v instanceof LocalTime lt ? lt : LocalTime.parse(pad(v.toString().trim()));
                yield sp.clock(t.getHour(), t.getMinute());
            }
            case "math" -> MathReader.read(v.toString(), sp).text();
            case "text" -> v.toString();
            case "cardinal" -> sp.cardinal(whole(v, s), arg);
            case "" -> {
                if (v instanceof CharSequence cs && !isNumber(cs.toString())) {
                    yield cs.toString();
                }
                BigDecimal d = number(v, s);
                yield d.stripTrailingZeros().scale() <= 0 && !(v instanceof BigDecimal bd && bd.scale() > 0)
                        ? sp.cardinal(d.longValueExact()) : sp.decimal(d);
            }
            default -> throw new IllegalArgumentException("unknown format " + f + " in {" + s.name() + ":" + f + "}");
        };
    }

    private static String pad(String t) {
        return t.indexOf(':') == 1 ? "0" + t : t;
    }

    private static boolean isNumber(String s) {
        return s.trim().matches("-?\\d+(?:[.,]\\d+)?");
    }

    private static BigDecimal number(Object v, Slot s) {
        if (v instanceof BigDecimal d) {
            return d;
        }
        if (v instanceof Integer || v instanceof Long || v instanceof Short || v instanceof Byte) {
            return BigDecimal.valueOf(((Number) v).longValue());
        }
        if (v instanceof Number n) {
            return BigDecimal.valueOf(n.doubleValue());
        }
        String t = v.toString().trim().replace(',', '.');
        try {
            return new BigDecimal(t);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("{" + s.name() + "} needs a number, got " + v, e);
        }
    }

    private static long whole(Object v, Slot s) {
        BigDecimal d = number(v, s);
        try {
            return d.longValueExact();
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("{" + s.name() + "} needs a whole number, got " + v, e);
        }
    }

    /**
     * A piece of a spelled line: literal text, or the words of one value.
     *
     * @param text the text, with its spaces
     * @param name the value it says, or {@code null} for literal text
     */
    public record Piece(String text, String name) {
    }

    /**
     * A stretch of a line recorded and spliced as one clip: literal words, or the words of a
     * value with any agreeing noun next to it ("drei Runden").
     *
     * @param text the words, trimmed
     * @param name the value it says, or {@code null} for literal words
     * @param position {@code mid} inside the line, or {@code end} plus the line's last
     *     punctuation ({@code end.}, {@code end?}, {@code end!}) for the last stretch
     */
    public record Unit(String text, String name, String position) {
    }

    /**
     * A spelled line.
     *
     * @param template the template's source
     * @param pieces its pieces, in order
     */
    public record Line(String template, List<Piece> pieces) {

        /**
         * Copies the pieces.
         *
         * @param template the template's source
         * @param pieces the pieces
         */
        public Line {
            pieces = List.copyOf(pieces);
        }

        /**
         * The whole line as text, for speaking it in one go.
         *
         * @return the text
         */
        public String text() {
            StringBuilder b = new StringBuilder();
            for (Piece p : pieces) {
                b.append(p.text());
            }
            return b.toString().replaceAll("\\s+", " ").trim();
        }

        /**
         * The line in stretches to record and splice: literal words, and each value's words.
         * Pieces with no letters or digits (spaces, punctuation) join the stretch before them.
         *
         * @return the stretches, in order
         */
        public List<Unit> units() {
            List<String> texts = new ArrayList<>();
            List<String> names = new ArrayList<>();
            for (Piece p : pieces) {
                boolean audible = p.text().codePoints().anyMatch(Character::isLetterOrDigit);
                int last = texts.size() - 1;
                if (last >= 0 && (!audible || Objects.equals(names.get(last), p.name()))) {
                    texts.set(last, texts.get(last) + p.text());
                } else if (!audible) {
                    texts.add(p.text());
                    names.add(null);
                } else {
                    texts.add(p.text());
                    names.add(p.name());
                }
            }
            // a leading stretch of punctuation joins the next
            if (texts.size() > 1 && texts.get(0).codePoints().noneMatch(Character::isLetterOrDigit)) {
                texts.set(1, texts.get(0) + texts.get(1));
                texts.remove(0);
                names.remove(0);
            }
            String whole = text();
            char endMark = whole.isEmpty() ? '.' : whole.charAt(whole.length() - 1);
            String end = "end" + (endMark == '?' || endMark == '!' ? endMark : '.');
            List<Unit> out = new ArrayList<>();
            for (int i = 0; i < texts.size(); i++) {
                out.add(new Unit(texts.get(i).replaceAll("\\s+", " ").trim(), names.get(i),
                        i == texts.size() - 1 ? end : "mid"));
            }
            return out;
        }
    }
}
