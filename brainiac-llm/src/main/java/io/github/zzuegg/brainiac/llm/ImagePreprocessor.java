package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.io.Json;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Map;

/**
 * A Hugging Face image processor ({@code preprocessor_config.json}): resize (bicubic or
 * bilinear, antialiased like Pillow and torchvision), center crop, rescale and normalize,
 * giving channels-first float pixels.
 */
final class ImagePreprocessor {

    private final boolean resize;
    /** The shorter side's length, or -1 to resize to {@link #height} x {@link #width}. */
    private final int shortest;
    private final int height;
    private final int width;
    private final int resample;
    private final boolean crop;
    private final int cropHeight;
    private final int cropWidth;
    private final boolean rescale;
    private final double factor;
    private final boolean normalize;
    private final double[] mean;
    private final double[] std;

    ImagePreprocessor(Map<String, Object> c, int imageSize) {
        resize = !Boolean.FALSE.equals(c.get("do_resize"));
        Map<String, Object> size = c.get("size") instanceof Map<?, ?> m ? Json.object(m) : Map.of();
        if (size.containsKey("shortest_edge")) {
            shortest = (int) Json.integer(size, "shortest_edge");
            height = width = -1;
        } else {
            shortest = -1;
            height = (int) Json.integer(size, "height", imageSize);
            width = (int) Json.integer(size, "width", imageSize);
        }
        resample = (int) Json.integer(c, "resample", 3);
        crop = Boolean.TRUE.equals(c.get("do_center_crop"));
        Map<String, Object> cs = c.get("crop_size") instanceof Map<?, ?> m ? Json.object(m) : Map.of();
        cropHeight = (int) Json.integer(cs, "height", imageSize);
        cropWidth = (int) Json.integer(cs, "width", imageSize);
        rescale = !Boolean.FALSE.equals(c.get("do_rescale"));
        factor = Json.number(c, "rescale_factor", 1 / 255.0);
        normalize = !Boolean.FALSE.equals(c.get("do_normalize"));
        mean = triple(c.get("image_mean"), 0.5);
        std = triple(c.get("image_std"), 0.5);
    }

    private static double[] triple(Object v, double otherwise) {
        if (v instanceof List<?> l && l.size() == 3) {
            return new double[] {((Number) l.get(0)).doubleValue(), ((Number) l.get(1)).doubleValue(),
                    ((Number) l.get(2)).doubleValue()};
        }
        return new double[] {otherwise, otherwise, otherwise};
    }

    /** @return the image's channels, 0 to 255, row-major */
    static double[][] channels(BufferedImage image) {
        int h = image.getHeight();
        int w = image.getWidth();
        double[][] rgb = new double[3][h * w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int argb = image.getRGB(x, y);
                rgb[0][y * w + x] = argb >> 16 & 0xFF;
                rgb[1][y * w + x] = argb >> 8 & 0xFF;
                rgb[2][y * w + x] = argb & 0xFF;
            }
        }
        return rgb;
    }

    /**
     * Resizes channels as Pillow does (antialiased, rounded to 8 bits after each pass).
     *
     * @param rgb the channels, row-major, replaced by the resized ones
     * @param resample Pillow's filter: 0 nearest, 2 bilinear, 3 bicubic
     */
    static void resize(double[][] rgb, int h, int w, int th, int tw, int resample) {
        ImagePreprocessor p = new ImagePreprocessor(Map.of("resample", resample), 1);
        for (int c = 0; c < rgb.length; c++) {
            double[] across = p.pass(rgb[c], h, w, tw, true);
            rgb[c] = p.pass(across, h, tw, th, false);
        }
    }

    /** @return the mean and standard deviation per channel, and the rescale factor */
    double[][] normalization() {
        return new double[][] {mean, std, {rescale ? factor : 1}, {normalize ? 1 : 0}};
    }

    /** @return {@code [3, height, width]} values, channels first */
    float[] apply(BufferedImage image, int[] outSize) {
        int h = image.getHeight();
        int w = image.getWidth();
        double[][] rgb = channels(image);
        if (resize) {
            int th;
            int tw;
            if (shortest > 0) {
                if (h <= w) {
                    th = shortest;
                    tw = (int) ((long) shortest * w / h);
                } else {
                    tw = shortest;
                    th = (int) ((long) shortest * h / w);
                }
            } else {
                th = height;
                tw = width;
            }
            for (int c = 0; c < 3; c++) {
                double[] across = pass(rgb[c], h, w, tw, true);
                rgb[c] = pass(across, h, tw, th, false);
            }
            h = th;
            w = tw;
        }
        int top = 0;
        int left = 0;
        int oh = h;
        int ow = w;
        if (crop) {
            top = Math.floorDiv(h - cropHeight, 2);
            left = Math.floorDiv(w - cropWidth, 2);
            oh = cropHeight;
            ow = cropWidth;
        }
        float[] out = new float[3 * oh * ow];
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < oh; y++) {
                for (int x = 0; x < ow; x++) {
                    int sy = y + top;
                    int sx = x + left;
                    double v = sy >= 0 && sy < h && sx >= 0 && sx < w ? rgb[c][sy * w + sx] : 0;
                    if (rescale) {
                        v *= factor;
                    }
                    if (normalize) {
                        v = (v - mean[c]) / std[c];
                    }
                    out[(c * oh + y) * ow + x] = (float) v;
                }
            }
        }
        outSize[0] = oh;
        outSize[1] = ow;
        return out;
    }

    /**
     * One antialiased resampling pass, as Pillow does it: along rows ({@code across}) or
     * columns, with the filter widened by the downscaling factor, rounded to 8 bits.
     */
    private double[] pass(double[] in, int h, int w, int to, boolean across) {
        int from = across ? w : h;
        int other = across ? h : w;
        double scale = (double) from / to;
        double filterScale = Math.max(scale, 1);
        double support = (resample == 2 ? 1 : resample == 0 ? 0.5 : 2) * filterScale;
        double[] out = new double[across ? h * to : to * w];
        double[] weights = new double[(int) Math.ceil(support) * 2 + 3];
        for (int i = 0; i < to; i++) {
            double center = (i + 0.5) * scale;
            int lo = Math.max((int) (center - support + 0.5), 0);
            int hi = Math.min((int) (center + support + 0.5), from);
            double sum = 0;
            for (int j = lo; j < hi; j++) {
                double wgt = filter((j - center + 0.5) / filterScale);
                weights[j - lo] = wgt;
                sum += wgt;
            }
            for (int o = 0; o < other; o++) {
                double v = 0;
                for (int j = lo; j < hi; j++) {
                    v += in[across ? o * w + j : j * w + o] * weights[j - lo];
                }
                v = sum == 0 ? 0 : v / sum;
                v = Math.max(0, Math.min(255, Math.rint(v)));
                out[across ? o * to + i : i * w + o] = v;
            }
        }
        return out;
    }

    private double filter(double x) {
        x = Math.abs(x);
        return switch (resample) {
            case 0 -> x < 0.5 ? 1 : 0;
            case 2 -> x < 1 ? 1 - x : 0;
            default -> {   // bicubic, a = -0.5
                double a = -0.5;
                if (x < 1) {
                    yield ((a + 2) * x - (a + 3)) * x * x + 1;
                }
                yield x < 2 ? (((x - 5) * x + 8) * x - 4) * a : 0;
            }
        };
    }
}
