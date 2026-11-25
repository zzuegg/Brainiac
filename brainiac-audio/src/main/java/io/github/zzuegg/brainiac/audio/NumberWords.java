package io.github.zzuegg.brainiac.audio;

import java.math.BigInteger;

/**
 * English number words, as num2words spells them (without "and" and commas). {@link
 * io.github.zzuegg.brainiac.audio.narration.NumberSpeller} builds times, operators and German
 * on top.
 */
public final class NumberWords {

    private static final String[] ONES = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen"};
    private static final String[] TENS = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
        "ninety"};
    private static final String[] SCALES = {"", "thousand", "million", "billion", "trillion", "quadrillion",
        "quintillion", "sextillion", "septillion", "octillion", "nonillion", "decillion"};

    private NumberWords() {
    }

    /**
     * {@code 1234} as "one thousand two hundred thirty-four".
     *
     * @param n the number
     * @return its words
     */
    public static String cardinal(BigInteger n) {
        if (n.signum() < 0) {
            return "minus " + cardinal(n.negate());
        }
        if (n.signum() == 0) {
            return "zero";
        }
        StringBuilder out = new StringBuilder();
        int scale = 0;
        java.util.List<String> parts = new java.util.ArrayList<>();
        BigInteger thousand = BigInteger.valueOf(1000);
        while (n.signum() > 0) {
            if (scale >= SCALES.length) {
                return digits(n.toString()) + " " + String.join(" ", parts);
            }
            int chunk = n.mod(thousand).intValue();
            if (chunk > 0) {
                parts.add(0, below1000(chunk) + (scale > 0 ? " " + SCALES[scale] : ""));
            }
            n = n.divide(thousand);
            scale++;
        }
        for (String p : parts) {
            if (!out.isEmpty()) {
                out.append(' ');
            }
            out.append(p);
        }
        return out.toString();
    }

    /**
     * {@code 1234} as "one thousand two hundred thirty-four".
     *
     * @param n the number
     * @return its words
     */
    public static String cardinal(long n) {
        return cardinal(BigInteger.valueOf(n));
    }

    private static String below1000(int n) {
        StringBuilder out = new StringBuilder();
        if (n >= 100) {
            out.append(ONES[n / 100]).append(" hundred");
            n %= 100;
            if (n > 0) {
                out.append(' ');
            }
        }
        if (n >= 20) {
            out.append(TENS[n / 10]);
            if (n % 10 > 0) {
                out.append('-').append(ONES[n % 10]);
            }
        } else if (n > 0 || out.isEmpty()) {
            out.append(ONES[n]);
        }
        return out.toString();
    }

    private static String digits(String s) {
        StringBuilder out = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (!out.isEmpty()) {
                out.append(' ');
            }
            out.append(ONES[c - '0']);
        }
        return out.toString();
    }

    /**
     * {@code 21} as "twenty-first".
     *
     * @param n the number
     * @return its ordinal words
     */
    public static String ordinal(BigInteger n) {
        String c = cardinal(n);
        int cut = Math.max(c.lastIndexOf(' '), c.lastIndexOf('-')) + 1;
        String last = c.substring(cut);
        String ord = switch (last) {
            case "one" -> "first";
            case "two" -> "second";
            case "three" -> "third";
            case "five" -> "fifth";
            case "eight" -> "eighth";
            case "nine" -> "ninth";
            case "twelve" -> "twelfth";
            default -> last.endsWith("y") ? last.substring(0, last.length() - 1) + "ieth" : last + "th";
        };
        return c.substring(0, cut) + ord;
    }

    /**
     * A year: 1984 as "nineteen eighty-four", 1905 as "nineteen oh-five", 2005 as "two thousand five".
     *
     * @param n the year
     * @return its words
     */
    public static String year(int n) {
        int high = n / 100;
        int low = n % 100;
        if (high == 0 || (high % 10 == 0 && low < 10) || high >= 100) {
            return cardinal(n);
        }
        String lowText = low == 0 ? "hundred" : low < 10 ? "oh-" + ONES[low] : cardinal(low);
        return cardinal(high) + " " + lowText;
    }

    /**
     * A decimal: "3.14" as "three point one four".
     *
     * @param s the digits, with a point
     * @return its words
     */
    public static String decimal(String s) {
        int dot = s.indexOf('.');
        String whole = dot < 0 ? s : s.substring(0, dot);
        String frac = dot < 0 ? "" : s.substring(dot + 1).replaceAll("0+$", "");
        StringBuilder out = new StringBuilder(cardinal(new BigInteger(whole.isEmpty() ? "0" : whole)));
        if (!frac.isEmpty()) {
            out.append(" point");
            for (char c : frac.toCharArray()) {
                out.append(' ').append(ONES[c - '0']);
            }
        }
        return out.toString();
    }
}
