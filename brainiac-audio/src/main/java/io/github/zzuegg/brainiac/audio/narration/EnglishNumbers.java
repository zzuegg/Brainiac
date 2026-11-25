package io.github.zzuegg.brainiac.audio.narration;

import io.github.zzuegg.brainiac.audio.NumberWords;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

/** English number words, on {@link NumberWords}. */
final class EnglishNumbers implements NumberSpeller {

    static final EnglishNumbers INSTANCE = new EnglishNumbers();

    private static final String[] DIGITS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine"};

    private EnglishNumbers() {
    }

    @Override
    public String language() {
        return "en";
    }

    @Override
    public String languageName() {
        return "english";
    }

    @Override
    public String cardinal(long n) {
        return NumberWords.cardinal(n);
    }

    @Override
    public String ordinal(long n, String ending) {
        return NumberWords.ordinal(BigInteger.valueOf(n));
    }

    @Override
    public String decimal(BigDecimal x) {
        String sign = x.signum() < 0 ? "minus " : "";
        String plain = x.abs().toPlainString();
        int dot = plain.indexOf('.');
        if (dot < 0) {
            return sign + NumberWords.cardinal(new BigInteger(plain));
        }
        return sign + NumberWords.cardinal(new BigInteger(plain.substring(0, dot))) + " point "
                + digits(plain.substring(dot + 1));
    }

    @Override
    public String digits(String digits) {
        StringBuilder out = new StringBuilder();
        for (char c : digits.toCharArray()) {
            if (c >= '0' && c <= '9') {
                if (!out.isEmpty()) {
                    out.append(' ');
                }
                out.append(DIGITS[c - '0']);
            }
        }
        return out.toString();
    }

    @Override
    public String duration(double seconds, int places) {
        BigDecimal t = BigDecimal.valueOf(Math.max(0, seconds)).setScale(Math.max(0, places), RoundingMode.HALF_UP);
        long whole = t.longValue();
        long hours = whole / 3600;
        long minutes = whole / 60 % 60;
        BigDecimal secs = t.subtract(BigDecimal.valueOf(hours * 3600 + minutes * 60));
        List<String> parts = new ArrayList<>();
        if (hours > 0) {
            parts.add(cardinal(hours) + (hours == 1 ? " hour" : " hours"));
        }
        if (minutes > 0) {
            parts.add(cardinal(minutes) + (minutes == 1 ? " minute" : " minutes"));
        }
        if (secs.signum() > 0 || parts.isEmpty()) {
            boolean one = secs.compareTo(BigDecimal.ONE) == 0 && places <= 0;
            parts.add((places > 0 ? decimal(secs) : cardinal(secs.longValue())) + (one ? " second" : " seconds"));
        }
        return join(parts, " and ");
    }

    static String join(List<String> parts, String last) {
        if (parts.size() == 1) {
            return parts.get(0);
        }
        return String.join(" ", parts.subList(0, parts.size() - 1)) + last + parts.get(parts.size() - 1);
    }

    @Override
    public String clock(int hour, int minute) {
        int h = hour % 12 == 0 ? 12 : hour % 12;
        if (minute == 0) {
            return cardinal(h) + " o'clock";
        }
        return cardinal(h) + (minute < 10 ? " oh " : " ") + cardinal(minute);
    }

    @Override
    public String operator(String symbol) {
        return switch (symbol) {
            case "+" -> "plus";
            case "-", "−", "–" -> "minus";
            case "*", "×", "·", "x" -> "times";
            case "/", "÷", ":" -> "divided by";
            case "=" -> "equals";
            case "^" -> "to the power of";
            case "²" -> "squared";
            case "%" -> "percent";
            case "<" -> "is less than";
            case ">" -> "is greater than";
            case "≤", "<=" -> "is at most";
            case "≥", ">=" -> "is at least";
            case "≠", "!=" -> "is not";
            case "(" -> "open bracket";
            case ")" -> "close bracket";
            case "?" -> "what";
            default -> null;
        };
    }
}
