package io.github.zzuegg.brainiac.experimental.world;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MaskGitDecoder;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * MAGVIT's multi-task masked token modelling (Yu et al., CVPR 2023,
 * <a href="https://arxiv.org/abs/2212.05199">arXiv:2212.05199</a>) on the token canvas of a
 * {@link SpaceTimeTokenizer}, around the bidirectional model of a {@link MaskGitDecoder}.
 * Every video task becomes a <em>condition mask</em> over the same canvas: frame prediction
 * keeps the first frame's tokens, interpolation keeps the first and the last tubelet group,
 * inpainting keeps everything outside a random spatial box (in every group), and unconditional
 * generation keeps nothing. Training ({@link #loss}) draws a task per clip, always shows its
 * condition tokens, masks a MaskGIT cosine-schedule share of the rest and scores the masked
 * ones; decoding ({@link #generate}) starts from the condition tokens with every other token
 * masked and reveals the most confident predictions over a few parallel steps, so one model
 * serves every task.
 */
public final class VideoMaskTasks {

    /** The conditional generation tasks. */
    public enum Task {
        /** Nothing given. */
        UNCONDITIONAL,
        /** The first frame given; the rest predicted. */
        PREDICTION,
        /** The first frame and the last tubelet group given; the middle generated. */
        INTERPOLATION,
        /** Everything outside a spatial box given; the box filled in every frame. */
        INPAINTING
    }

    private final Model model;
    private final int vocabulary;
    private final int maskId;
    private final SpaceTimeTokenizer tokenizer;
    private final int length;

    private VideoMaskTasks(Model model, int vocabulary, int maskId, SpaceTimeTokenizer tokenizer, int frames) {
        this.model = model;
        this.vocabulary = vocabulary;
        this.maskId = maskId;
        this.tokenizer = tokenizer;
        this.length = tokenizer.tokens(frames);
    }

    /**
     * Wraps a MaskGIT decoder's model.
     *
     * @param decoder the decoder (its model maps {@code [b, L]} ids to {@code [b, L, V]} logits)
     * @param vocabulary the vocabulary including the mask id
     * @param maskId the mask id
     * @param tokenizer the tokenizer whose canvas the tasks address
     * @param frames the clip length
     * @return the tasks
     */
    public static VideoMaskTasks of(MaskGitDecoder decoder, int vocabulary, int maskId, SpaceTimeTokenizer tokenizer,
            int frames) {
        return new VideoMaskTasks(decoder.model(), vocabulary, maskId, tokenizer, frames);
    }

    /** @return the canvas length */
    public int length() {
        return length;
    }

    /**
     * The condition mask of a task.
     *
     * @param task the task
     * @param random draws the inpainting box
     * @return per token, whether it is given
     */
    public boolean[] condition(Task task, RandomGenerator random) {
        boolean[] given = new boolean[length];
        int p = tokenizer.patchesPerFrame();
        int g = (int) Math.round(Math.sqrt(p));
        switch (task) {
            case UNCONDITIONAL -> {
            }
            case PREDICTION -> Arrays.fill(given, 0, p, true);
            case INTERPOLATION -> {
                Arrays.fill(given, 0, p, true);
                Arrays.fill(given, length - p, length, true);
            }
            case INPAINTING -> {
                int w = 1 + random.nextInt(Math.max(1, g / 2));
                int x0 = random.nextInt(g - w + 1);
                int y0 = random.nextInt(g - w + 1);
                for (int j = 0; j < length; j++) {
                    int q = j % p;
                    int x = q % g;
                    int y = q / g;
                    given[j] = !(x >= x0 && x < x0 + w && y >= y0 && y < y0 + w);
                }
            }
            default -> throw new IllegalArgumentException(task.toString());
        }
        return given;
    }

    /**
     * The multi-task masked-token loss.
     *
     * @param tokens clean token sequences
     * @param random draws tasks, mask ratios and positions
     * @return the mean cross-entropy over masked tokens
     */
    public Var loss(int[][] tokens, RandomGenerator random) {
        int b = tokens.length;
        Task[] tasks = Task.values();
        float[] input = new float[b * length];
        float[] weight = new float[b * length * vocabulary];
        List<int[]> scored = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            boolean[] given = condition(tasks[random.nextInt(tasks.length)], random);
            List<Integer> free = new ArrayList<>();
            for (int j = 0; j < length; j++) {
                if (!given[j]) {
                    free.add(j);
                }
            }
            java.util.Collections.shuffle(free, new java.util.Random(random.nextLong()));
            int k = Math.max(1, (int) Math.ceil(MaskGitDecoder.gamma(random.nextDouble()) * free.size()));
            boolean[] masked = new boolean[length];
            for (int m = 0; m < Math.min(k, free.size()); m++) {
                masked[free.get(m)] = true;
                scored.add(new int[] {i, free.get(m)});
            }
            for (int j = 0; j < length; j++) {
                input[i * length + j] = masked[j] ? maskId : tokens[i][j];
            }
        }
        for (int[] s : scored) {
            weight[(s[0] * length + s[1]) * vocabulary + tokens[s[0]][s[1]]] = 1f / Math.max(1, scored.size());
        }
        Var logits = model.forward(Var.of(Tensor.of(Shape.of(b, length), input)), true);
        Var logp = Activation.LOG_SOFTMAX.forward(logits.reshape(b * length, vocabulary));
        return logp.mul(Tensor.of(Shape.of(b * length, vocabulary), weight)).sum().neg();
    }

    /**
     * Fills the non-given tokens by MaskGIT's confidence-ordered parallel decoding.
     *
     * @param tokens per sequence the given tokens (others are ignored)
     * @param given per sequence which tokens are given
     * @param steps decoding steps (network calls)
     * @param temperature the sampling temperature
     * @param choiceTemperature the Gumbel noise on the confidence ranking
     * @param random the random source
     * @return the completed sequences
     */
    public int[][] generate(int[][] tokens, boolean[][] given, int steps, float temperature, float choiceTemperature,
            RandomGenerator random) {
        int b = tokens.length;
        int[][] x = new int[b][length];
        int[] unknown = new int[b];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < length; j++) {
                x[i][j] = given[i][j] ? tokens[i][j] : maskId;
                unknown[i] += given[i][j] ? 0 : 1;
            }
        }
        for (int t = 0; t < steps; t++) {
            float[] flat = new float[b * length];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < length; j++) {
                    flat[i * length + j] = x[i][j];
                }
            }
            float[] logits = Var.noGrad(() -> model.forward(Var.of(Tensor.of(Shape.of(b, length), flat)), false))
                    .value().toArray();
            double progress = (t + 1.0) / steps;
            for (int i = 0; i < b; i++) {
                int keepMasked = t == steps - 1 ? 0 : (int) Math.floor(MaskGitDecoder.gamma(progress) * unknown[i]);
                List<int[]> picks = new ArrayList<>();
                List<Double> conf = new ArrayList<>();
                for (int j = 0; j < length; j++) {
                    if (x[i][j] != maskId) {
                        continue;
                    }
                    double[] p = softmax(logits, (i * length + j) * vocabulary, temperature);
                    int token = draw(p, random);
                    double u = Math.max(1e-12, random.nextDouble());
                    picks.add(new int[] {j, token});
                    conf.add(Math.log(Math.max(1e-30, p[token])) - choiceTemperature * (1 - progress)
                            * Math.log(-Math.log(u)));
                }
                int reveal = Math.max(Math.min(1, picks.size()), picks.size() - keepMasked);
                Integer[] order = new Integer[picks.size()];
                for (int k = 0; k < order.length; k++) {
                    order[k] = k;
                }
                Arrays.sort(order, (a, c) -> Double.compare(conf.get(c), conf.get(a)));
                for (int k = 0; k < reveal; k++) {
                    int[] pick = picks.get(order[k]);
                    x[i][pick[0]] = pick[1];
                }
            }
        }
        return x;
    }

    private double[] softmax(float[] logits, int offset, float temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (int v = 0; v < vocabulary; v++) {
            if (v != maskId) {
                max = Math.max(max, logits[offset + v] / temperature);
            }
        }
        double[] p = new double[vocabulary];
        double sum = 0;
        for (int v = 0; v < vocabulary; v++) {
            p[v] = v == maskId ? 0 : Math.exp(logits[offset + v] / temperature - max);
            sum += p[v];
        }
        for (int v = 0; v < vocabulary; v++) {
            p[v] /= sum;
        }
        return p;
    }

    private static int draw(double[] p, RandomGenerator random) {
        double u = random.nextDouble();
        int last = 0;
        for (int v = 0; v < p.length; v++) {
            if (p[v] > 0) {
                last = v;
            }
            u -= p[v];
            if (u <= 0 && p[v] > 0) {
                return v;
            }
        }
        return last;
    }
}
