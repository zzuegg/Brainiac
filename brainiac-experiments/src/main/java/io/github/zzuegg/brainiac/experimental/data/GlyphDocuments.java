package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Synthetic documents for OCR, document parsing and layout detection on a CPU, drawn with
 * the 3x5 pixel font of the digit-glyph experiments extended to the letters A-Z and a few
 * punctuation marks (the digit shapes are exactly those of {@code DigitGlyphs}, which is
 * left unchanged).
 *
 * <ul>
 *   <li>{@link #line} renders a text line of height {@link #LINE_HEIGHT} with variable
 *       spacing, touching characters (no blank column between two glyphs), pixel noise and a
 *       slight skew (the baseline moves by one pixel part-way along the line), and returns the
 *       column span of every character, so a segmenter, a CTC recogniser or a transformer can
 *       be scored on character and word error rates.</li>
 *   <li>{@link #form} renders a 2-D form or receipt page: a title word, then key/value pairs
 *       ({@code NAME: BOB}, {@code TOTAL: 42}) with the value to the right of or below its
 *       key, and returns every word with its box, its role (key, value, other), the field it
 *       belongs to, the reading order and the gold field/value pairs.</li>
 *   <li>{@link #layout} renders a page of layout regions (text blocks of glyph rows, ruled
 *       tables and filled figures) with their boxes, for document layout detection.</li>
 * </ul>
 *
 * <p>Pixels are 0 (paper) or 1 (ink) before noise flips them.
 */
public final class GlyphDocuments {

    /** The characters the font draws; index 0 is the space. */
    public static final String CHARSET = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ:.-";
    /** Glyph width in pixels. */
    public static final int GLYPH_WIDTH = 3;
    /** Glyph height in pixels. */
    public static final int GLYPH_HEIGHT = 5;
    /** Height of a rendered line: the glyph plus one pixel above and below. */
    public static final int LINE_HEIGHT = GLYPH_HEIGHT + 2;
    /** The field keys of forms, in their canonical order. */
    public static final String[] KEYS = {"NAME", "DATE", "TOTAL", "TAX", "CITY", "ID"};
    /** Words used in random lines; also the lexicon of lexicon decoding. */
    public static final String[] LEXICON = {"NAME", "DATE", "TOTAL", "TAX", "CITY", "ID", "BOB", "ANN", "EVE",
        "TOM", "SUE", "MAX", "MILK", "TEA", "EGGS", "BREAD", "CASH", "ROME", "OSLO", "LIMA", "ITEM", "QTY", "FORM",
        "RECEIPT", "PAID", "DUE"};
    private static final String[] NAMES = {"BOB", "ANN", "EVE", "TOM", "SUE", "MAX"};
    private static final String[] CITIES = {"ROME", "OSLO", "LIMA", "BERN"};
    private static final String[] TITLES = {"FORM", "RECEIPT", "BILL"};

    private static final String[] FONT = new String[CHARSET.length()];

    static {
        String[] digits = {"111101101101111", "010110010010111", "111001111100111", "111001111001111",
            "101101111001001", "111100111001111", "111100111101111", "111001010010010", "111101111101111",
            "111101111001111"};
        String[] letters = {"010101111101101", "110101110101110", "011100100100011", "110101101101110",
            "111100110100111", "111100110100100", "011100101101011", "101101111101101", "111010010010111",
            "001001001101010", "101101110101101", "100100100100111", "101111111101101", "110101101101101",
            "010101101101010", "110101110100100", "010101101110011", "110101110101101", "011100010001110",
            "111010010010010", "101101101101111", "101101101101010", "101101111111101", "101101010101101",
            "101101010010010", "111001010100111"};
        FONT[0] = "000000000000000";
        System.arraycopy(digits, 0, FONT, 1, 10);
        System.arraycopy(letters, 0, FONT, 11, 26);
        FONT[37] = "000010000010000";
        FONT[38] = "000000000000010";
        FONT[39] = "000000111000000";
    }

    private GlyphDocuments() {
    }

    /**
     * The index of a character in {@link #CHARSET}.
     *
     * @param c the character
     * @return its index
     */
    public static int index(char c) {
        int i = CHARSET.indexOf(c);
        if (i < 0) {
            throw new IllegalArgumentException("no glyph for '" + c + "'");
        }
        return i;
    }

    /**
     * The 3x5 bitmap of a character, row-major, 1 for ink.
     *
     * @param c the character
     * @return 15 values
     */
    public static float[] glyph(char c) {
        String f = FONT[index(c)];
        float[] g = new float[GLYPH_WIDTH * GLYPH_HEIGHT];
        for (int i = 0; i < g.length; i++) {
            g[i] = f.charAt(i) == '1' ? 1 : 0;
        }
        return g;
    }

    /**
     * How a line is drawn.
     *
     * @param minGap the fewest blank columns between glyphs of a word
     * @param maxGap the most blank columns between glyphs of a word
     * @param touch the probability that two neighbouring glyphs of a word touch (no gap)
     * @param wordGap blank columns between words
     * @param noise the probability of flipping each pixel
     * @param skew the probability that the baseline moves by one pixel part-way along the line
     */
    public record Style(int minGap, int maxGap, double touch, int wordGap, double noise, double skew) {

        /**
         * Checks the style.
         *
         * @param minGap fewest gap columns
         * @param maxGap most gap columns
         * @param touch touching probability
         * @param wordGap word gap
         * @param noise flip probability
         * @param skew skew probability
         */
        public Style {
            if (minGap < 0 || maxGap < minGap || wordGap < 1 || touch < 0 || touch > 1 || noise < 0 || skew < 0) {
                throw new IllegalArgumentException("bad line style");
            }
        }

        /** @return clean, evenly spaced glyphs */
        public static Style clean() {
            return new Style(1, 1, 0, 3, 0, 0);
        }

        /** @return spacing of 1-2 columns, 30% touching pairs, 3% noise and 30% skewed lines */
        public static Style hard() {
            return new Style(1, 2, 0.3, 3, 0.03, 0.3);
        }
    }

    /**
     * A rendered text line.
     *
     * @param width the image width
     * @param pixels {@code LINE_HEIGHT x width}, row-major
     * @param text the text, words separated by single spaces
     * @param spans per character of {@code text} its first and last column (spaces included,
     *     spanning their gap)
     * @param touching whether any two glyphs touch
     */
    public record Line(int width, float[] pixels, String text, int[][] spans, boolean touching) {

        /** @return the CTC labels: character indices plus one (0 is the blank) */
        public int[] labels() {
            int[] l = new int[text.length()];
            for (int i = 0; i < l.length; i++) {
                l[i] = index(text.charAt(i)) + 1;
            }
            return l;
        }
    }

    /**
     * Renders a line.
     *
     * @param text the text (characters of {@link #CHARSET})
     * @param style how to draw it
     * @param random the noise source
     * @return the line
     */
    public static Line line(String text, Style style, SplittableRandom random) {
        int n = text.length();
        int[] gaps = new int[n];
        boolean touching = false;
        for (int i = 1; i < n; i++) {
            char a = text.charAt(i - 1);
            char b = text.charAt(i);
            if (a == ' ' || b == ' ') {
                gaps[i] = 0;
            } else if (random.nextDouble() < style.touch()) {
                gaps[i] = 0;
                touching = true;
            } else {
                gaps[i] = style.minGap() + random.nextInt(style.maxGap() - style.minGap() + 1);
            }
        }
        int width = 2;
        int[][] spans = new int[n][2];
        int x = 1;
        for (int i = 0; i < n; i++) {
            x += gaps[i];
            int w = text.charAt(i) == ' ' ? style.wordGap() : GLYPH_WIDTH;
            spans[i][0] = x;
            spans[i][1] = x + w - 1;
            x += w;
        }
        width += x - 1;
        float[] img = new float[LINE_HEIGHT * width];
        int skewAt = random.nextDouble() < style.skew() ? 1 + random.nextInt(Math.max(1, width - 2)) : Integer.MAX_VALUE;
        int shift = random.nextBoolean() ? 1 : -1;
        for (int i = 0; i < n; i++) {
            char c = text.charAt(i);
            if (c == ' ') {
                continue;
            }
            String f = FONT[index(c)];
            int dy = spans[i][0] >= skewAt ? shift : 0;
            for (int r = 0; r < GLYPH_HEIGHT; r++) {
                for (int col = 0; col < GLYPH_WIDTH; col++) {
                    int y = 1 + r + dy;
                    if (f.charAt(r * GLYPH_WIDTH + col) == '1' && y >= 0 && y < LINE_HEIGHT) {
                        img[y * width + spans[i][0] + col] = 1;
                    }
                }
            }
        }
        flip(img, style.noise(), random);
        return new Line(width, img, text, spans, touching);
    }

    private static void flip(float[] img, double p, SplittableRandom random) {
        if (p <= 0) {
            return;
        }
        for (int i = 0; i < img.length; i++) {
            if (random.nextDouble() < p) {
                img[i] = 1 - img[i];
            }
        }
    }

    /**
     * Random line text of lexicon words and numbers.
     *
     * @param minGlyphs the fewest characters (spaces included)
     * @param maxGlyphs the most characters
     * @param random the source
     * @return the text
     */
    public static String randomText(int minGlyphs, int maxGlyphs, SplittableRandom random) {
        int target = minGlyphs + random.nextInt(maxGlyphs - minGlyphs + 1);
        StringBuilder sb = new StringBuilder();
        while (true) {
            String w = random.nextInt(3) == 0 ? Integer.toString(random.nextInt(1000))
                    : LEXICON[random.nextInt(LEXICON.length)];
            int len = sb.isEmpty() ? w.length() : sb.length() + 1 + w.length();
            if (len > maxGlyphs) {
                if (sb.length() >= minGlyphs) {
                    return sb.toString();
                }
                w = w.substring(0, Math.max(1, maxGlyphs - sb.length() - (sb.isEmpty() ? 0 : 1)));
                if (!sb.isEmpty()) {
                    sb.append(' ');
                }
                return sb.append(w).toString();
            }
            if (!sb.isEmpty()) {
                sb.append(' ');
            }
            sb.append(w);
            if (sb.length() >= target) {
                return sb.toString();
            }
        }
    }

    /**
     * Pads lines to one width (blank columns on the right) and stacks them.
     *
     * @param lines the lines
     * @param width the common width, at least the widest line
     * @return {@code [n, 1, LINE_HEIGHT, width]}
     */
    public static Tensor batch(List<Line> lines, int width) {
        float[] out = new float[lines.size() * LINE_HEIGHT * width];
        for (int i = 0; i < lines.size(); i++) {
            Line l = lines.get(i);
            if (l.width() > width) {
                throw new IllegalArgumentException("line wider than " + width);
            }
            for (int r = 0; r < LINE_HEIGHT; r++) {
                System.arraycopy(l.pixels(), r * l.width(), out, (i * LINE_HEIGHT + r) * width, l.width());
            }
        }
        return Tensor.of(Shape.of(lines.size(), 1, LINE_HEIGHT, width), out);
    }

    // ------------------------------------------------------------------ forms

    /** The role of a word on a form. */
    public enum Role {
        /** A field key such as {@code NAME:}. */
        KEY,
        /** A field value. */
        VALUE,
        /** Anything else (titles). */
        OTHER
    }

    /**
     * A word on a page.
     *
     * @param text the word
     * @param x0 left column
     * @param y0 top row
     * @param x1 right column, exclusive
     * @param y1 bottom row, exclusive
     * @param role its role
     * @param field the index into {@link #KEYS} of its field, or -1
     */
    public record Word(String text, int x0, int y0, int x1, int y1, Role role, int field) {
    }

    /**
     * A layout region.
     *
     * @param kind {@link #TEXT}, {@link #TABLE} or {@link #FIGURE}
     * @param box {@code x0, y0, x1, y1} in pixels
     */
    public record Region(int kind, float[] box) {
    }

    /** Region kind: a block of text lines. */
    public static final int TEXT = 0;
    /** Region kind: a ruled table. */
    public static final int TABLE = 1;
    /** Region kind: a filled figure. */
    public static final int FIGURE = 2;
    /** The number of region kinds. */
    public static final int REGION_KINDS = 3;

    /**
     * A rendered page.
     *
     * @param height rows
     * @param width columns
     * @param pixels row-major
     * @param words the words in reading order (top to bottom, left to right)
     * @param fields per key of {@link #KEYS} its value, or null where the form has no such field
     * @param regions layout regions (empty for forms)
     */
    public record Page(int height, int width, float[] pixels, List<Word> words, String[] fields,
            List<Region> regions) {

        /**
         * The gold parse in canonical key order, such as {@code [NAME, BOB, TOTAL, 42]}.
         *
         * @return alternating keys and values
         */
        public List<String> parse() {
            List<String> out = new ArrayList<>();
            for (int k = 0; k < KEYS.length; k++) {
                if (fields[k] != null) {
                    out.add(KEYS[k]);
                    out.add(fields[k]);
                }
            }
            return out;
        }
    }

    private static String value(int key, SplittableRandom r) {
        return switch (KEYS[key]) {
            case "NAME" -> NAMES[r.nextInt(NAMES.length)];
            case "CITY" -> CITIES[r.nextInt(CITIES.length)];
            case "DATE" -> (1 + r.nextInt(12)) + "-" + (1 + r.nextInt(28));
            default -> Integer.toString(r.nextInt(100));
        };
    }

    /**
     * Renders a form or receipt: an optional title, then {@code fields} key/value pairs, one
     * per row; each value sits to the right of its key, or with probability {@code below}
     * on the row below it.
     *
     * @param fields the number of fields, 1 to {@link #KEYS} length
     * @param below the probability of a value below its key
     * @param title whether to draw a title word
     * @param noise the pixel flip probability
     * @param random the source
     * @return the page
     */
    public static Page form(int fields, double below, boolean title, double noise, SplittableRandom random) {
        int[] keys = random.ints(0, KEYS.length).distinct().limit(fields).toArray();
        List<Word> words = new ArrayList<>();
        String[] values = new String[KEYS.length];
        int y = 1;
        int maxX = 0;
        if (title) {
            String t = TITLES[random.nextInt(TITLES.length)];
            int x = 1 + random.nextInt(3);
            words.add(new Word(t, x, y, x + width(t), y + GLYPH_HEIGHT, Role.OTHER, -1));
            maxX = Math.max(maxX, x + width(t));
            y += LINE_HEIGHT;
        }
        for (int k : keys) {
            String key = KEYS[k] + ":";
            String v = value(k, random);
            values[k] = v;
            int x = 1 + random.nextInt(3);
            words.add(new Word(key, x, y, x + width(key), y + GLYPH_HEIGHT, Role.KEY, k));
            if (random.nextDouble() < below) {
                y += LINE_HEIGHT - 1;
                int vx = x + random.nextInt(3);
                words.add(new Word(v, vx, y, vx + width(v), y + GLYPH_HEIGHT, Role.VALUE, k));
                maxX = Math.max(maxX, Math.max(x + width(key), vx + width(v)));
            } else {
                int vx = x + width(key) + 2 + random.nextInt(4);
                words.add(new Word(v, vx, y, vx + width(v), y + GLYPH_HEIGHT, Role.VALUE, k));
                maxX = Math.max(maxX, vx + width(v));
            }
            y += LINE_HEIGHT;
        }
        int height = y;
        int width = maxX + 1;
        float[] img = new float[height * width];
        for (Word w : words) {
            draw(img, width, w.text(), w.x0(), w.y0());
        }
        flip(img, noise, random);
        return new Page(height, width, img, words, values, List.of());
    }

    /**
     * The pixel width of a word drawn with one blank column between glyphs.
     *
     * @param word the word
     * @return its width
     */
    public static int width(String word) {
        return word.length() * (GLYPH_WIDTH + 1) - 1;
    }

    private static void draw(float[] img, int width, String text, int x, int y) {
        for (int i = 0; i < text.length(); i++) {
            String f = FONT[index(text.charAt(i))];
            for (int r = 0; r < GLYPH_HEIGHT; r++) {
                for (int c = 0; c < GLYPH_WIDTH; c++) {
                    if (f.charAt(r * GLYPH_WIDTH + c) == '1') {
                        img[(y + r) * width + x + i * (GLYPH_WIDTH + 1) + c] = 1;
                    }
                }
            }
        }
    }

    /**
     * Crops a page to a fixed size (padding with paper), for batching.
     *
     * @param pages the pages
     * @param height the common height
     * @param width the common width
     * @return {@code [n, 1, height, width]}
     */
    public static Tensor images(List<Page> pages, int height, int width) {
        float[] out = new float[pages.size() * height * width];
        for (int i = 0; i < pages.size(); i++) {
            Page p = pages.get(i);
            for (int r = 0; r < Math.min(height, p.height()); r++) {
                System.arraycopy(p.pixels(), r * p.width(), out, (i * height + r) * width, Math.min(width, p.width()));
            }
        }
        return Tensor.of(Shape.of(pages.size(), 1, height, width), out);
    }

    // ------------------------------------------------------------------ layout pages

    /**
     * Renders a square page of 1 to {@code maxRegions} non-overlapping layout regions: text
     * blocks (rows of short ink runs separated by blank rows), tables (a ruled grid) and
     * figures (a filled ellipse inside a frame).
     *
     * @param side the page side
     * @param maxRegions the most regions
     * @param noise the pixel flip probability
     * @param random the source
     * @return the page, with its regions and no words
     */
    public static Page layout(int side, int maxRegions, double noise, SplittableRandom random) {
        float[] img = new float[side * side];
        boolean[] used = new boolean[side * side];
        List<Region> regions = new ArrayList<>();
        int want = 1 + random.nextInt(maxRegions);
        for (int attempt = 0; attempt < 40 && regions.size() < want; attempt++) {
            int kind = random.nextInt(REGION_KINDS);
            int w = 6 + random.nextInt(Math.max(1, side / 2 - 5));
            int h = 5 + random.nextInt(Math.max(1, side / 2 - 4));
            int x0 = random.nextInt(side - w + 1);
            int y0 = random.nextInt(side - h + 1);
            boolean free = true;
            for (int y = Math.max(0, y0 - 1); y < Math.min(side, y0 + h + 1) && free; y++) {
                for (int x = Math.max(0, x0 - 1); x < Math.min(side, x0 + w + 1); x++) {
                    if (used[y * side + x]) {
                        free = false;
                        break;
                    }
                }
            }
            if (!free) {
                continue;
            }
            for (int y = y0; y < y0 + h; y++) {
                for (int x = x0; x < x0 + w; x++) {
                    used[y * side + x] = true;
                    boolean ink = switch (kind) {
                        case TEXT -> (y - y0) % 2 == 0 && ((x - x0) % 4 != 3) && x < x0 + w - (y % 3);
                        case TABLE -> y == y0 || y == y0 + h - 1 || x == x0 || x == x0 + w - 1 || (y - y0) % 3 == 0
                                || (x - x0) % 4 == 0;
                        default -> {
                            double cx = (x - x0 + 0.5) / w - 0.5;
                            double cy = (y - y0 + 0.5) / h - 0.5;
                            yield cx * cx + cy * cy < 0.16;
                        }
                    };
                    if (ink) {
                        img[y * side + x] = 1;
                    }
                }
            }
            regions.add(new Region(kind, new float[] {x0, y0, x0 + w, y0 + h}));
        }
        flip(img, noise, random);
        return new Page(side, side, img, List.of(), new String[KEYS.length], regions);
    }

    /**
     * Checks that every glyph of the font is distinct.
     *
     * @return true if no two characters other than the space share a bitmap
     */
    public static boolean distinctGlyphs() {
        return Arrays.stream(FONT, 1, FONT.length).distinct().count() == FONT.length - 1;
    }
}
