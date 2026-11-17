package io.github.zzuegg.brainiac.experiments.multimodal;

import java.util.random.RandomGenerator;

/**
 * Synthetic 8x8 digit images with a text side: each digit 0-9 is drawn from a 3x5 pixel font
 * at a random position (6 x 4 = 24 placements), pixels in {-1, +1}, optionally with some
 * pixels flipped. Captions name the digit in a few phrasings ("seven", "digit seven",
 * "number 7", ...), so an image tower and a text tower can be trained on pairs.
 */
public final class DigitGlyphs {
    /** Image side length. */
    public static final int SIDE = 8;
    /** Pixels per image. */
    public static final int PIXELS = SIDE * SIDE;
    /** The number of classes. */
    public static final int CLASSES = 10;
    /** The digit names. */
    public static final String[] NAMES = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine"};

    private static final String[] FONT = {
        "111101101101111", "010110010010111", "111001111100111", "111001111001111", "101101111001001",
        "111100111001111", "111100111101111", "111001010010010", "111101111101111", "111101111001111"};

    private DigitGlyphs() {
    }

    /**
     * Draws a digit at a position.
     *
     * @param digit the digit
     * @param x the left column, 0..5
     * @param y the top row, 0..3
     * @return the image, row-major
     */
    public static float[] render(int digit, int x, int y) {
        float[] img = new float[PIXELS];
        java.util.Arrays.fill(img, -1);
        for (int r = 0; r < 5; r++) {
            for (int c = 0; c < 3; c++) {
                if (FONT[digit].charAt(r * 3 + c) == '1') {
                    img[(y + r) * SIDE + x + c] = 1;
                }
            }
        }
        return img;
    }

    /**
     * Draws a digit at a random position with noise.
     *
     * @param digit the digit
     * @param flip the probability of flipping each pixel
     * @param random the random source
     * @return the image
     */
    public static float[] sample(int digit, double flip, RandomGenerator random) {
        float[] img = render(digit, random.nextInt(SIDE - 2), random.nextInt(SIDE - 4));
        for (int i = 0; i < img.length; i++) {
            if (random.nextDouble() < flip) {
                img[i] = -img[i];
            }
        }
        return img;
    }

    /**
     * The phrasings of a caption.
     *
     * @param digit the digit
     * @return all captions of it
     */
    public static String[] captions(int digit) {
        return new String[] {NAMES[digit], "digit " + NAMES[digit], "number " + digit, "a " + NAMES[digit],
            "the digit " + digit};
    }
}
