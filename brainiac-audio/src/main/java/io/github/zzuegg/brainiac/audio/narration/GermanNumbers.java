package io.github.zzuegg.brainiac.audio.narration;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

/**
 * German number words: "einundzwanzig", "hunderteins", "zweitausend", "eine Million", ordinals
 * with endings ("dritten"), "drei Komma eins vier", "vierzehn Uhr fünf".
 */
final class GermanNumbers implements NumberSpeller {

    static final GermanNumbers INSTANCE = new GermanNumbers();

    private static final String[] ONES = {"null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht",
        "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn",
        "neunzehn"};
    private static final String[] TENS = {"", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig",
        "achtzig", "neunzig"};
    private static final String[][] BIG = {{"Million", "Millionen"}, {"Milliarde", "Milliarden"},
        {"Billion", "Billionen"}, {"Billiarde", "Billiarden"}, {"Trillion", "Trillionen"}};

    private GermanNumbers() {
    }

    @Override
    public String language() {
        return "de";
    }

    @Override
    public String languageName() {
        return "german";
    }

    @Override
    public String cardinal(long n) {
        return cardinal(n, true);
    }

    /** {@code finalOne}: whether a trailing 1 is "eins" (said alone) or "ein" (before Millionen). */
    private String cardinal(long n, boolean finalOne) {
        if (n == Long.MIN_VALUE) {
            return minus() + " " + digits(Long.toString(n).substring(1));
        }
        if (n < 0) {
            return minus() + " " + cardinal(-n, finalOne);
        }
        if (n == 0) {
            return "null";
        }
        List<String> words = new ArrayList<>();
        long rest = n % 1_000_000;
        long high = n / 1_000_000;
        for (int scale = 0; high > 0; scale++, high /= 1000) {
            long chunk = high % 1000;
            if (chunk > 0) {
                words.add(0, chunk == 1 ? "eine " + BIG[scale][0] : below1000((int) chunk, false) + " " + BIG[scale][1]);
            }
        }
        if (rest > 0) {
            int thousands = (int) (rest / 1000);
            int units = (int) (rest % 1000);
            String w = "";
            if (thousands > 0) {
                w = (thousands == 1 ? "" : below1000(thousands, false)) + "tausend";
            }
            if (units > 0) {
                w += below1000(units, finalOne);
            }
            words.add(w);
        }
        return String.join(" ", words);
    }

    /** 1 to 999 as one word; a trailing 1 is "eins" if {@code finalOne}. */
    private static String below1000(int n, boolean finalOne) {
        StringBuilder out = new StringBuilder();
        int h = n / 100;
        int r = n % 100;
        if (h > 0) {
            out.append(h == 1 ? "" : ONES[h]).append("hundert");
        }
        if (r == 1) {
            out.append(finalOne ? "eins" : "ein");
        } else if (r > 0 && r < 20) {
            out.append(ONES[r]);
        } else if (r >= 20) {
            int u = r % 10;
            if (u > 0) {
                out.append(u == 1 ? "ein" : ONES[u]).append("und");
            }
            out.append(TENS[r / 10]);
        }
        return out.toString();
    }

    @Override
    public String ordinal(long n, String ending) {
        String end = ending == null ? "e" : ending;
        if (n == 0) {
            return "nullt" + end;
        }
        long r = n % 100;
        if (r >= 1 && r < 20) {
            String stem = switch ((int) r) {
                case 1 -> "erst";
                case 3 -> "dritt";
                case 7 -> "siebt";
                case 8 -> "acht";
                default -> ONES[(int) r] + "t";
            };
            String head = n - r == 0 ? "" : cardinal(n - r, false);
            if (head.contains(" ")) {
                head += " ";
            }
            return head + stem + end;
        }
        String c = cardinal(n, false);
        if (c.endsWith("Million") || c.endsWith("Millionen") || c.endsWith("Milliarde") || c.endsWith("Milliarden")) {
            c = c.replace("eine ", "").replace(" ", "").toLowerCase(java.util.Locale.ROOT)
                    .replaceAll("(million|milliarde)n?(en)?$", "$1");
        }
        return c + "st" + end;
    }

    @Override
    public String decimal(BigDecimal x) {
        String sign = x.signum() < 0 ? minus() + " " : "";
        String plain = x.abs().toPlainString();
        int dot = plain.indexOf('.');
        if (dot < 0) {
            return sign + cardinal(Long.parseLong(plain));
        }
        return sign + cardinal(Long.parseLong(plain.substring(0, dot))) + " Komma " + digits(plain.substring(dot + 1));
    }

    @Override
    public String digits(String digits) {
        StringBuilder out = new StringBuilder();
        for (char c : digits.toCharArray()) {
            if (c >= '0' && c <= '9') {
                if (!out.isEmpty()) {
                    out.append(' ');
                }
                out.append(ONES[c - '0']);
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
            parts.add(hours == 1 ? "eine Stunde" : cardinal(hours) + " Stunden");
        }
        if (minutes > 0) {
            parts.add(minutes == 1 ? "eine Minute" : cardinal(minutes) + " Minuten");
        }
        if (secs.signum() > 0 || parts.isEmpty()) {
            boolean one = secs.compareTo(BigDecimal.ONE) == 0 && places <= 0;
            parts.add(one ? "eine Sekunde"
                    : (places > 0 ? decimal(secs) : cardinal(secs.longValue())) + " Sekunden");
        }
        return EnglishNumbers.join(parts, " und ");
    }

    @Override
    public String clock(int hour, int minute) {
        String h = hour == 1 ? "ein" : cardinal(hour);
        return minute == 0 ? h + " Uhr" : h + " Uhr " + cardinal(minute);
    }

    @Override
    public String operator(String symbol) {
        return switch (symbol) {
            case "+" -> "plus";
            case "-", "−", "–" -> "minus";
            case "*", "×", "·", "x" -> "mal";
            case "/", "÷", ":" -> "geteilt durch";
            case "=" -> "ist";
            case "^" -> "hoch";
            case "²" -> "hoch zwei";
            case "%" -> "Prozent";
            case "<" -> "ist kleiner als";
            case ">" -> "ist größer als";
            case "≤", "<=" -> "ist höchstens";
            case "≥", ">=" -> "ist mindestens";
            case "≠", "!=" -> "ist nicht";
            case "(" -> "Klammer auf";
            case ")" -> "Klammer zu";
            case "?" -> "wie viel";
            default -> null;
        };
    }
}
