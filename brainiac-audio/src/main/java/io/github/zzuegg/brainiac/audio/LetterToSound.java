package io.github.zzuegg.brainiac.audio;

import java.util.Locale;

/**
 * Reads words that are in no dictionary: as compounds of dictionary words when they split
 * into two or three, otherwise by English spelling rules (digraphs, a silent final e that
 * lengthens the vowel before it, soft c and g), stressed on the first vowel.
 */
final class LetterToSound {

    private static final String[][] RULES = {
        {"tion", "ʃən"}, {"sion", "ʒən"}, {"ture", "ʧəɹ"}, {"ough", "O"}, {"augh", "ɔ"}, {"eigh", "A"},
        {"tch", "ʧ"}, {"sch", "sk"}, {"igh", "I"}, {"dge", "ʤ"},
        {"ph", "f"}, {"sh", "ʃ"}, {"ch", "ʧ"}, {"th", "θ"}, {"wh", "w"}, {"ck", "k"}, {"ng", "ŋ"}, {"qu", "kw"},
        {"gh", ""}, {"ee", "i"}, {"ea", "i"}, {"ie", "i"}, {"ei", "A"}, {"oo", "u"}, {"ou", "W"}, {"ow", "O"},
        {"oa", "O"}, {"ai", "A"}, {"ay", "A"}, {"oi", "Y"}, {"oy", "Y"}, {"au", "ɔ"}, {"aw", "ɔ"}, {"ew", "u"},
        {"ue", "u"}, {"ui", "u"}, {"ey", "A"}, {"ar", "ɑɹ"}, {"er", "əɹ"}, {"ir", "ɜɹ"}, {"ur", "ɜɹ"}, {"or", "ɔɹ"},
    };
    private static final String VOWEL_LETTERS = "aeiouy";

    private LetterToSound() {
    }

    static String read(String text, EnglishPhonemizer dict) {
        String word = text.toLowerCase(Locale.ROOT).replaceAll("[^a-z]", "");
        if (word.isEmpty()) {
            return "";
        }
        String compound = compound(word, dict, 3);
        if (compound != null) {
            return compound;
        }
        return EnglishPhonemizer.stress(rules(word, dict.isBritish()), 2.0);
    }

    /** The word as up to {@code parts} dictionary words of three or more letters, later ones less stressed. */
    private static String compound(String word, EnglishPhonemizer dict, int parts) {
        String whole = dict.known(word);
        if (whole != null) {
            return whole;
        }
        if (parts < 2) {
            return null;
        }
        for (int cut = word.length() - 3; cut >= 3; cut--) {
            String head = dict.known(word.substring(0, cut));
            if (head == null) {
                continue;
            }
            String tail = compound(word.substring(cut), dict, parts - 1);
            if (tail != null) {
                return head + EnglishPhonemizer.stress(tail, -0.5);
            }
        }
        return null;
    }

    private static String rules(String w, boolean british) {
        StringBuilder out = new StringBuilder();
        int n = w.length();
        int i = 0;
        if (w.startsWith("kn") || w.startsWith("gn")) {
            out.append('n');
            i = 2;
        } else if (w.startsWith("wr")) {
            out.append('ɹ');
            i = 2;
        }
        outer:
        while (i < n) {
            for (String[] r : RULES) {
                if (w.startsWith(r[0], i)) {
                    out.append(r[1]);
                    i += r[0].length();
                    continue outer;
                }
            }
            char c = w.charAt(i);
            char next = i + 1 < n ? w.charAt(i + 1) : 0;
            if (next == c && VOWEL_LETTERS.indexOf(c) < 0) {
                i++;       // doubled consonant
                continue;
            }
            switch (c) {
                case 'a', 'e', 'i', 'o', 'u' -> {
                    boolean last = i == n - 1;
                    if (c == 'e' && last && n > 2) {
                        break;     // silent final e
                    }
                    boolean magic = i + 2 == n - 1 && w.charAt(n - 1) == 'e' && VOWEL_LETTERS.indexOf(next) < 0;
                    if (magic || (last && c == 'o')) {
                        out.append(switch (c) {
                            case 'a' -> "A";
                            case 'e' -> "i";
                            case 'i' -> "I";
                            case 'o' -> "O";
                            default -> "u";
                        });
                    } else {
                        out.append(switch (c) {
                            case 'a' -> "æ";
                            case 'e' -> "ɛ";
                            case 'i' -> "ɪ";
                            case 'o' -> british ? "ɒ" : "ɑ";
                            default -> "ʌ";
                        });
                    }
                }
                case 'y' -> out.append(i == 0 ? "j" : i == n - 1 ? (n <= 3 ? "I" : "i") : "ɪ");
                case 'c' -> out.append(next == 'e' || next == 'i' || next == 'y' ? "s" : "k");
                case 'g' -> out.append(next == 'e' || next == 'i' || next == 'y' ? "ʤ" : "ɡ");
                case 's' -> {
                    boolean between = i > 0 && i + 1 < n && VOWEL_LETTERS.indexOf(w.charAt(i - 1)) >= 0
                            && VOWEL_LETTERS.indexOf(next) >= 0;
                    out.append(between || (i == n - 1 && i > 0 && "aeiouybdglmnrvw".indexOf(w.charAt(i - 1)) >= 0)
                            ? "z" : "s");
                }
                case 'x' -> out.append(i == 0 ? "z" : "ks");
                case 'j' -> out.append("ʤ");
                case 'q' -> out.append("k");
                case 'r' -> out.append("ɹ");
                default -> out.append(c);
            }
            i++;
        }
        return out.toString();
    }
}
