package io.github.zzuegg.brainiac.audio.narration;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Reads arithmetic aloud: {@code "12 + 7 = 19"} as "twelve plus seven equals nineteen" or
 * "zwölf plus sieben ist neunzehn", {@code "(2 + 3) × 4"} with "open bracket ... close
 * bracket", {@code "-5"} as "minus five", {@code "3 + ? = 10"} with "what" for the gap.
 *
 * <p>{@link #template} turns an expression into a {@link LineTemplate} whose holes are its
 * numbers, such as {@code "{n0} plus {n1} ist {n2}"}: every sum of that shape shares the
 * template, so its recorded pieces serve them all.
 */
public final class MathReader {

    private MathReader() {
    }

    /**
     * An expression as a template and the values for it.
     *
     * @param template the template, with holes {@code n0}, {@code n1}, ... for the numbers
     * @param values the numbers, by hole
     */
    public record Templated(LineTemplate template, Map<String, Object> values) {
    }

    /**
     * Spells an expression.
     *
     * @param expression such as {@code "12 + 7 = 19"}
     * @param speller the language's words
     * @return the spelled line, each number a piece of its own
     * @throws IllegalArgumentException if a symbol is not understood
     */
    public static LineTemplate.Line read(String expression, NumberSpeller speller) {
        Templated t = template(expression, speller);
        return t.template().render(t.values(), speller);
    }

    /**
     * Turns an expression into a template with a hole for each number.
     *
     * @param expression such as {@code "12 + 7 = 19"}
     * @param speller the language's words, for the operators
     * @return the template and the numbers
     * @throws IllegalArgumentException if a symbol is not understood
     */
    public static Templated template(String expression, NumberSpeller speller) {
        StringBuilder src = new StringBuilder();
        Map<String, Object> values = new LinkedHashMap<>();
        List<String> tokens = tokens(expression);
        boolean operand = true;   // whether a number may come next (so '-' is a sign)
        for (int i = 0; i < tokens.size(); i++) {
            String t = tokens.get(i);
            char c = t.charAt(0);
            boolean number = Character.isDigit(c) || c == '.' && t.length() > 1;
            if (operand && (t.equals("-") || t.equals("−")) && i + 1 < tokens.size()
                    && Character.isDigit(tokens.get(i + 1).charAt(0))) {
                String name = "n" + values.size();
                values.put(name, new BigDecimal("-" + tokens.get(i + 1).replace(',', '.')));
                append(src, "{" + name + "}");
                i++;
                operand = false;
            } else if (number) {
                String name = "n" + values.size();
                values.put(name, new BigDecimal(t.replace(',', '.')));
                append(src, "{" + name + "}");
                operand = false;
            } else {
                String w = speller.operator(t);
                if (w == null) {
                    throw new IllegalArgumentException("cannot read " + t + " in: " + expression);
                }
                append(src, w);
                operand = !t.equals(")") && !t.equals("%") && !t.equals("²") && !t.equals("?");
            }
        }
        return new Templated(LineTemplate.parse(src.toString()), values);
    }

    private static void append(StringBuilder b, String s) {
        if (!b.isEmpty()) {
            b.append(' ');
        }
        b.append(s);
    }

    /** Numbers (with {@code .} or {@code ,} decimals), and each symbol on its own. */
    static List<String> tokens(String s) {
        List<String> out = new ArrayList<>();
        int i = 0;
        while (i < s.length()) {
            char c = s.charAt(i);
            if (Character.isWhitespace(c)) {
                i++;
            } else if (Character.isDigit(c)) {
                int j = i;
                while (j < s.length() && (Character.isDigit(s.charAt(j))
                        || (s.charAt(j) == '.' || s.charAt(j) == ',') && j + 1 < s.length()
                        && Character.isDigit(s.charAt(j + 1)))) {
                    j++;
                }
                out.add(s.substring(i, j));
                i = j;
            } else if ((c == '<' || c == '>' || c == '!') && i + 1 < s.length() && s.charAt(i + 1) == '=') {
                out.add(s.substring(i, i + 2));
                i += 2;
            } else {
                out.add(String.valueOf(c));
                i++;
            }
        }
        return out;
    }
}
