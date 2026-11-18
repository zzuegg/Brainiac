package io.github.zzuegg.brainiac.diffusion;

/** The image shapes offered for generation, each mapped to a size the model was trained near. */
public enum AspectRatio {
    /** 1:1. */
    SQUARE(1, 1),
    /** 4:3. */
    LANDSCAPE_4_3(4, 3),
    /** 3:4. */
    PORTRAIT_3_4(3, 4),
    /** 16:9. */
    WIDE_16_9(16, 9),
    /** 9:16. */
    TALL_9_16(9, 16);

    private final int width;
    private final int height;

    AspectRatio(int width, int height) {
        this.width = width;
        this.height = height;
    }

    /**
     * Returns a size of this shape with about the area of a {@code base x base} square, both
     * sides multiples of 64: for SDXL (base 1024) 1024x1024, 1152x896, 896x1152, 1344x768 and
     * 768x1344.
     *
     * @param base the side of the model's native square resolution (512, 768 or 1024)
     * @return {@code {width, height}}
     */
    public int[] size(int base) {
        if (width == height) {
            return new int[] {base, base};
        }
        double ratio = (double) width / height;
        double area = (double) base * base;
        boolean wide = ratio > 1;
        double r = wide ? ratio : 1 / ratio;
        double longSide = Math.sqrt(area * r);
        int[] best = null;
        double bestError = Double.MAX_VALUE;
        for (int l = (int) (longSide / 64) * 64 - 64; l <= longSide + 128; l += 64) {
            if (l < 64) {
                continue;
            }
            int s = (int) Math.max(64, Math.round(area / l / 64) * 64);
            double error = Math.abs((double) l / s - r);
            if (error < bestError - 1e-9) {
                bestError = error;
                best = new int[] {l, s};
            }
        }
        return wide ? best : new int[] {best[1], best[0]};
    }

    /**
     * Returns the label people know the shape by.
     *
     * @return such as {@code "16:9"}
     */
    public String label() {
        return width + ":" + height;
    }
}
