package io.github.zzuegg.brainiac.experiments.world;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeScenes;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.DiffusionForcing;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MaskGitDecoder;
import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;
import io.github.zzuegg.brainiac.experimental.vision.SimVp;
import io.github.zzuegg.brainiac.experimental.world.FrameDenoiser;
import io.github.zzuegg.brainiac.experimental.world.HistoryGuidance;
import io.github.zzuegg.brainiac.experimental.world.SelfForcingTrainer;
import io.github.zzuegg.brainiac.experimental.world.SpaceTimeDenoiser;
import io.github.zzuegg.brainiac.experimental.world.SpaceTimeTokenizer;
import io.github.zzuegg.brainiac.experimental.world.VideoMaskTasks;
import io.github.zzuegg.brainiac.experimental.world.VideoMetrics;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 9 items 9.5.1-9.5.4 and the video row of 9.5.13, on {@link ShapeScenes#video} clips: two
 * textured 4-pixel tiles moving at constant speed (0.5 to 1.5 px per frame) and bouncing off
 * the walls of a 16x16 frame over a smooth background.
 *
 * <ul>
 *   <li>{@link #spaceTime} (9.5.1): factorised space-time attention against a frame-wise image
 *       diffusion model and full 3-D attention, unconditional and first-frame-conditioned, and
 *       SimVP from the first frame as the deterministic predictor.
 *   <li>{@link #tokenizers} (9.5.2, 9.5.13): the causal 3-D tokenizer against per-frame tokens,
 *       with FSQ and VQ; {@link #maskedGeneration} trains multi-task MaskGIT on the 3-D tokens and
 *       compares it per network call with raster-order decoding and with 9.5.1's diffusion.
 *   <li>{@link #longRollouts} (9.5.3, 9.5.4): frame-autoregressive models trained on 8-frame
 *       clips and rolled out to 32 frames: teacher forcing, the core {@code DiffusionForcing}
 *       on flattened frames, diffusion forcing with and without history guidance, Self Forcing
 *       and the bidirectional 9.5.1 model in sliding windows.
 * </ul>
 *
 * <p>Scores come from {@link VideoMetrics}: a frozen blob detector counts objects (2 is right)
 * and a nearest-neighbour tracker measures deviation from constant velocity and track breaks;
 * a kernel MMD between frame features (per-frame quality) and between clip features (appearance
 * plus motion, an FVD proxy) compares generated and real sets.
 */
public final class VideoGeneration {

    /** The frame side. */
    public static final int SIDE = 16;
    /** The training clip length. */
    public static final int FRAMES = 8;
    /** Objects per clip. */
    public static final int OBJECTS = 2;
    private static final float THRESHOLD = 0.4f;
    private static final int MIN_PIXELS = 3;
    private static final double RADIUS = 4;

    private VideoGeneration() {
    }

    /**
     * Random clips.
     *
     * @param n how many
     * @param length frames per clip
     * @param seed the seed
     * @return {@code [n, length, SIDE, SIDE]}
     */
    public static Tensor clips(int n, int length, long seed) {
        ShapeScenes gen = new ShapeScenes(SIDE, 0f);
        SplittableRandom r = new SplittableRandom(seed);
        int p = SIDE * SIDE;
        float[] out = new float[n * length * p];
        for (int i = 0; i < n; i++) {
            ShapeScenes.Video v = gen.video(length, OBJECTS, 4, 0.5f, 1.5f, false, r);
            for (int t = 0; t < length; t++) {
                System.arraycopy(v.frames()[t], 0, out, (i * length + t) * p, p);
            }
        }
        return Tensor.of(Shape.of(n, length, SIDE, SIDE), out);
    }

    private static float[][][] unpack(Tensor clips) {
        int n = clips.shape().dim(0);
        int t = clips.shape().dim(1);
        int p = SIDE * SIDE;
        float[] v = clips.toArray();
        float[][][] out = new float[n][t][p];
        for (int i = 0; i < n; i++) {
            for (int f = 0; f < t; f++) {
                System.arraycopy(v, (i * t + f) * p, out[i][f], 0, p);
            }
        }
        return out;
    }

    /**
     * The scores of a set of clips.
     *
     * @param frameMmd MMD of frame features against real frames
     * @param clipMmd MMD of clip features against real clips (FVD proxy)
     * @param countError mean absolute object-count error per frame
     * @param velocityError mean deviation from constant velocity (pixels)
     * @param trackBreaks the share of detections without a predecessor
     */
    public record Quality(double frameMmd, double clipMmd, double countError, double velocityError,
            double trackBreaks) {
    }

    /**
     * Scores generated clips against real ones.
     *
     * @param generated {@code [n, T, SIDE, SIDE]}
     * @param real {@code [m, T, SIDE, SIDE]}
     * @return the scores
     */
    public static Quality quality(Tensor generated, Tensor real) {
        float[][][] g = unpack(generated);
        float[][][] r = unpack(real);
        List<float[]> gf = new ArrayList<>();
        List<float[]> rf = new ArrayList<>();
        float[][] gc = new float[g.length][];
        float[][] rc = new float[r.length][];
        List<VideoMetrics.ClipScore> scores = new ArrayList<>();
        for (int i = 0; i < g.length; i++) {
            gc[i] = VideoMetrics.clipFeatures(g[i], SIDE);
            for (float[] f : g[i]) {
                gf.add(VideoMetrics.frameFeatures(f, SIDE));
            }
            scores.add(VideoMetrics.score(g[i], SIDE, OBJECTS, THRESHOLD, MIN_PIXELS, RADIUS));
        }
        for (int i = 0; i < r.length; i++) {
            rc[i] = VideoMetrics.clipFeatures(r[i], SIDE);
            for (float[] f : r[i]) {
                rf.add(VideoMetrics.frameFeatures(f, SIDE));
            }
        }
        double[] m = VideoMetrics.mean(scores);
        return new Quality(VideoMetrics.mmd(gf.toArray(new float[0][]), rf.toArray(new float[0][])),
                VideoMetrics.mmd(gc, rc), m[0], m[1], m[2]);
    }

    // ---------------------------------------------------------------- 9.5.1 space-time diffusion

    /**
     * A 9.5.1 row.
     *
     * @param model the model
     * @param unconditional scores of unconditional samples
     * @param conditional scores of first-frame-conditioned samples
     * @param millisPerStep training time per step
     * @param parameters the parameter count
     */
    public record DiffusionRow(String model, Quality unconditional, Quality conditional, double millisPerStep,
            long parameters) {
    }

    /**
     * Trains a space-time denoiser.
     *
     * @param mode the attention layout
     * @param steps Adam steps of 8 clips
     * @param jointImages whether to add an image batch (single frames) to every step
     * @param seed the seed
     * @return the trained model
     */
    public static SpaceTimeDenoiser trainDenoiser(SpaceTimeDenoiser.Attention mode, int steps, boolean jointImages,
            long seed) {
        SpaceTimeDenoiser net = SpaceTimeDenoiser.of(SIDE, 4, 32, 2, 2, FRAMES, mode, false, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        Optimizer opt = Optimizer.adam(2e-3f);
        for (int s = 0; s < steps; s++) {
            Tensor batch = clips(8, FRAMES, r.nextLong());
            Var loss = net.loss(batch, 1, r);
            if (jointImages) {
                Tensor frames = batch.reshape(8 * FRAMES, 1, SIDE, SIDE).slice(0, 0, 16);
                loss = loss.add(net.imageLoss(frames, r).mul(0.5f));
            }
            loss.backward();
            opt.step(net.parameters());
        }
        return net;
    }

    /**
     * 9.5.1: frame-wise, factorised (with joint image training) and full 3-D attention, plus
     * SimVP from the first frame (conditional only) and the real clips as the reference row.
     *
     * @param steps training steps per model
     * @param samples clips generated per score
     * @param sampling Euler steps
     * @param seed the seed
     * @return one row per model, the real clips first
     */
    public static List<DiffusionRow> spaceTime(int steps, int samples, int sampling, long seed) {
        Tensor real = clips(samples, FRAMES, seed + 101);
        Tensor other = clips(samples, FRAMES, seed + 202);
        Tensor first = other.slice(1, 0, 1);
        List<DiffusionRow> rows = new ArrayList<>();
        Quality ref = quality(other, real);
        rows.add(new DiffusionRow("real clips", ref, ref, 0, 0));
        for (SpaceTimeDenoiser.Attention mode : SpaceTimeDenoiser.Attention.values()) {
            long start = System.nanoTime();
            SpaceTimeDenoiser net = trainDenoiser(mode, steps, mode == SpaceTimeDenoiser.Attention.FACTORISED, seed);
            double ms = (System.nanoTime() - start) / 1e6 / steps;
            SplittableRandom r = new SplittableRandom(seed + 3);
            Quality unc = quality(net.sample(samples, FRAMES, null, sampling, r), real);
            Quality cond = quality(net.sample(samples, FRAMES, first, sampling, r), real);
            rows.add(new DiffusionRow(mode.toString().toLowerCase(java.util.Locale.ROOT), unc, cond, ms,
                    CurvatureProducts.size(net.parameters())));
        }
        long start = System.nanoTime();
        SimVp simvp = new SimVp(1, FRAMES - 1, 8, 2, new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 4);
        for (int s = 0; s < steps; s++) {
            Tensor batch = clips(8, FRAMES, r.nextLong());
            simvp.forward(Var.of(batch.slice(1, 0, 1))).sub(batch.slice(1, 1, FRAMES)).square().mean().backward();
            opt.step(simvp.parameters());
        }
        double ms = (System.nanoTime() - start) / 1e6 / steps;
        Tensor pred = Var.noGrad(() -> simvp.forward(Var.of(first))).value();
        Quality cond = quality(Tensor.concat(1, first, pred), real);
        rows.add(new DiffusionRow("SimVP (deterministic)", null, cond, ms, CurvatureProducts.size(simvp.parameters())));
        return rows;
    }

    // ---------------------------------------------------------------- 9.5.2 / 9.5.13 tokenizers

    /** Clip length for the tokenizers: a first frame plus two 4-frame tubelet groups. */
    public static final int TOKEN_FRAMES = 9;

    /**
     * A tokenizer row.
     *
     * @param tokenizer the tokenizer
     * @param tokens tokens per clip
     * @param psnr reconstruction PSNR on held-out clips
     * @param perplexity code perplexity on held-out clips
     * @param utilisation the share of codes used
     */
    public record TokenizerRow(String tokenizer, int tokens, double psnr, double perplexity, double utilisation) {
    }

    /**
     * Trains a tokenizer.
     *
     * @param tok the tokenizer
     * @param steps Adam steps of 8 clips
     * @param seed the seed
     * @return the tokenizer
     */
    public static SpaceTimeTokenizer train(SpaceTimeTokenizer tok, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            Tensor batch = clips(8, TOKEN_FRAMES, r.nextLong());
            Var l = tok.loss(batch);
            l.backward();
            opt.step(tok.parameters());
            if (s > 0 && s % 50 == 0) {
                tok.restartDeadCodes(batch);
            }
        }
        return tok;
    }

    private static TokenizerRow score(String name, SpaceTimeTokenizer tok, Tensor test) {
        int[][] codes = tok.encode(test);
        long[] usage = tok.usage(codes);
        double psnr = SpaceTimeTokenizer.psnr(tok.decode(codes, TOKEN_FRAMES), test);
        return new TokenizerRow(name, codes[0].length, psnr,
                io.github.zzuegg.brainiac.experimental.audio.FiniteScalarQuantizer.perplexity(usage),
                io.github.zzuegg.brainiac.experimental.audio.FiniteScalarQuantizer.utilisation(usage));
    }

    /**
     * 9.5.2 and the 9.5.13 video row: 3-D tubelet tokens (stride 4) against per-frame tokens at
     * the same patch (4x the tokens) and at twice the patch (fewer tokens than 3-D), with FSQ
     * levels {@code (8, 5, 5, 5)} (1,000 codes) and a 1,024-entry VQ.
     *
     * @param steps training steps per tokenizer
     * @param seed the seed
     * @return one row per tokenizer
     */
    public static List<TokenizerRow> tokenizers(int steps, long seed) {
        Tensor test = clips(32, TOKEN_FRAMES, seed + 77);
        int[] levels = {8, 5, 5, 5};
        List<TokenizerRow> rows = new ArrayList<>();
        rows.add(score("3-D FSQ (4x4x4 tubelets)", train(SpaceTimeTokenizer.fsq(SIDE, 4, 4, 64, levels, seed), steps,
                seed), test));
        rows.add(score("3-D VQ (4x4x4 tubelets)", train(SpaceTimeTokenizer.vq(SIDE, 4, 4, 64, 1024, 4, seed), steps,
                seed), test));
        rows.add(score("per-frame FSQ (4x4 patches)", train(SpaceTimeTokenizer.fsq(SIDE, 4, 1, 64, levels, seed), steps,
                seed), test));
        rows.add(score("per-frame FSQ (8x8 patches)", train(SpaceTimeTokenizer.fsq(SIDE, 8, 1, 64, levels, seed), steps,
                seed), test));
        return rows;
    }

    /**
     * A generation row of 9.5.2.
     *
     * @param model the generator
     * @param calls network calls per clip
     * @param quality the scores
     * @param millis generation time for the whole set
     */
    public record GenerationRow(String model, int calls, Quality quality, double millis) {
    }

    /**
     * 9.5.2 generation: multi-task MaskGIT on 3-D FSQ tokens (125 codes), decoding unconditional
     * clips in 8 parallel steps and in raster order (one token per call), plus frame prediction
     * from the first frame; 9.5.1's factorised diffusion at 16 Euler steps for comparison.
     *
     * @param tokenizerSteps tokenizer training steps
     * @param maskGitSteps MaskGIT training steps
     * @param diffusionSteps diffusion training steps
     * @param samples clips per score
     * @param seed the seed
     * @return the rows
     */
    public static List<GenerationRow> maskedGeneration(int tokenizerSteps, int maskGitSteps, int diffusionSteps,
            int samples, long seed) {
        SpaceTimeTokenizer tok = train(SpaceTimeTokenizer.fsq(SIDE, 4, 4, 64, new int[] {5, 5, 5}, seed),
                tokenizerSteps, seed);
        int length = tok.tokens(TOKEN_FRAMES);
        int vocab = tok.vocabulary() + 1;
        int mask = vocab - 1;
        Model model = Model.sequential().input(length).embedding(vocab, 32).positionalEncoding().transformer(2, false)
                .transformer(2, false).dense(vocab).seed(seed).build();
        MaskGitDecoder decoder = MaskGitDecoder.of(model, vocab, mask);
        VideoMaskTasks tasks = VideoMaskTasks.of(decoder, vocab, mask, tok, TOKEN_FRAMES);
        SplittableRandom r = new SplittableRandom(seed + 5);
        Optimizer opt = Optimizer.adam(2e-3f);
        for (int s = 0; s < maskGitSteps; s++) {
            Var l = tasks.loss(tok.encode(clips(16, TOKEN_FRAMES, r.nextLong())), r);
            l.backward();
            opt.step(model.parameters());
        }
        Tensor real = clips(samples, TOKEN_FRAMES, seed + 101);
        Tensor other = clips(samples, TOKEN_FRAMES, seed + 202);
        List<GenerationRow> rows = new ArrayList<>();
        int[][] blank = new int[samples][length];
        boolean[][] none = new boolean[samples][length];
        for (int steps : new int[] {8, length}) {
            long start = System.nanoTime();
            int[][] gen = tasks.generate(blank, none, steps, 1f, steps == length ? 0f : 4.5f, r);
            Tensor clipsOut = tok.decode(gen, TOKEN_FRAMES);
            rows.add(new GenerationRow(steps == length ? "MaskGIT, one token per call" : "MaskGIT, 8 steps", steps,
                    quality(clipsOut, real), (System.nanoTime() - start) / 1e6));
        }
        int[][] given = tok.encode(other);
        boolean[][] firstFrame = new boolean[samples][];
        for (int i = 0; i < samples; i++) {
            firstFrame[i] = tasks.condition(VideoMaskTasks.Task.PREDICTION, r);
        }
        long start = System.nanoTime();
        Tensor predicted = tok.decode(tasks.generate(given, firstFrame, 8, 1f, 4.5f, r), TOKEN_FRAMES);
        rows.add(new GenerationRow("MaskGIT, frame prediction, 8 steps", 8, quality(predicted, real),
                (System.nanoTime() - start) / 1e6));
        SpaceTimeDenoiser net = trainDenoiser(SpaceTimeDenoiser.Attention.FACTORISED, diffusionSteps, true, seed);
        start = System.nanoTime();
        Tensor diff = net.sample(samples, FRAMES, null, 16, r);
        rows.add(new GenerationRow("9.5.1 factorised diffusion, 16 steps", 16, quality(diff, real.slice(1, 0, FRAMES)),
                (System.nanoTime() - start) / 1e6));
        return rows;
    }

    // ---------------------------------------------------------------- 9.5.3 / 9.5.4 long rollouts

    /** The rollout length (4x the training length). */
    public static final int LONG = 32;

    /**
     * A rollout row: scores per frame bucket {@code [1, 8)}, {@code [8, 16)}, {@code [16, 32)}.
     *
     * @param model the model
     * @param countError object-count error per bucket
     * @param frameMmd frame-feature MMD against real frames per bucket
     * @param velocityError deviation from constant velocity over the whole rollout
     * @param millis rollout time for the set
     */
    public record RolloutRow(String model, double[] countError, double[] frameMmd, double velocityError,
            double millis) {
    }

    private static final int[][] BUCKETS = {{1, 8}, {8, 16}, {16, LONG}};

    /**
     * Scores long rollouts per frame bucket.
     *
     * @param name the row name
     * @param videos {@code [n, LONG, SIDE, SIDE]}
     * @param real real long clips
     * @param millis generation time
     * @return the row
     */
    public static RolloutRow rolloutRow(String name, Tensor videos, Tensor real, double millis) {
        float[][][] g = unpack(videos);
        float[][][] r = unpack(real);
        double[] count = new double[BUCKETS.length];
        double[] mmd = new double[BUCKETS.length];
        for (int b = 0; b < BUCKETS.length; b++) {
            List<float[]> gf = new ArrayList<>();
            List<float[]> rf = new ArrayList<>();
            int frames = 0;
            for (float[][] clip : g) {
                for (int f = BUCKETS[b][0]; f < BUCKETS[b][1]; f++) {
                    gf.add(VideoMetrics.frameFeatures(clip[f], SIDE));
                    count[b] += Math.abs(VideoMetrics.blobs(clip[f], SIDE, THRESHOLD, MIN_PIXELS).length - OBJECTS);
                    frames++;
                }
            }
            count[b] /= frames;
            for (float[][] clip : r) {
                for (int f = BUCKETS[b][0]; f < BUCKETS[b][1]; f++) {
                    rf.add(VideoMetrics.frameFeatures(clip[f], SIDE));
                }
            }
            mmd[b] = VideoMetrics.mmd(gf.toArray(new float[0][]), rf.toArray(new float[0][]));
        }
        List<VideoMetrics.ClipScore> scores = new ArrayList<>();
        for (float[][] clip : g) {
            scores.add(VideoMetrics.score(clip, SIDE, OBJECTS, THRESHOLD, MIN_PIXELS, RADIUS));
        }
        return new RolloutRow(name, count, mmd, VideoMetrics.mean(scores)[1], millis);
    }

    /**
     * Trains a frame-autoregressive model.
     *
     * @param mode teacher or diffusion forcing
     * @param window history frames
     * @param steps Adam steps of 8 clips
     * @param seed the seed
     * @return the model
     */
    public static FrameDenoiser trainFrameModel(FrameDenoiser.Training mode, int window, int steps, long seed) {
        FrameDenoiser m = FrameDenoiser.of(SIDE, window, 8, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        Optimizer opt = Optimizer.adam(2e-3f);
        for (int s = 0; s < steps; s++) {
            Var l = m.loss(clips(8, FRAMES, r.nextLong()), mode, 0.3f, r);
            l.backward();
            opt.step(m.parameters());
        }
        return m;
    }

    /**
     * 9.5.3 and 9.5.4: long rollouts from real first frames.
     *
     * @param pretrainSteps training steps of every base model
     * @param selfForcingSteps Self Forcing steps on top of the teacher-forced model
     * @param samples rollouts per row
     * @param weights history-guidance weights to sweep
     * @param seed the seed
     * @return one row per model; the first row is real 32-frame clips
     */
    public static List<RolloutRow> longRollouts(int pretrainSteps, int selfForcingSteps, int samples, float[] weights,
            long seed) {
        int window = 2;
        int sampling = 8;
        Tensor real = clips(samples, LONG, seed + 101);
        Tensor other = clips(samples, LONG, seed + 202);
        Tensor first = other.slice(1, 0, 1);
        List<RolloutRow> rows = new ArrayList<>();
        rows.add(rolloutRow("real clips", other, real, 0));
        SplittableRandom r = new SplittableRandom(seed + 9);
        FrameDenoiser tf = trainFrameModel(FrameDenoiser.Training.TEACHER_FORCING, window, pretrainSteps, seed);
        long start = System.nanoTime();
        Tensor v = tf.rollout(first, LONG, 0, sampling, null, r);
        rows.add(rolloutRow("teacher forcing", v, real, (System.nanoTime() - start) / 1e6));
        FrameDenoiser df = trainFrameModel(FrameDenoiser.Training.DIFFUSION_FORCING, window, pretrainSteps, seed);
        start = System.nanoTime();
        v = df.rollout(first, LONG, 0.1f, sampling, null, r);
        rows.add(rolloutRow("diffusion forcing (history level 0.1)", v, real, (System.nanoTime() - start) / 1e6));
        for (float w : weights) {
            start = System.nanoTime();
            v = df.rollout(first, LONG, 0.1f, sampling, HistoryGuidance.vanilla(w), r);
            rows.add(rolloutRow("history guidance w=" + w, v, real, (System.nanoTime() - start) / 1e6));
        }
        start = System.nanoTime();
        v = df.rollout(first, LONG, 0.1f, sampling, HistoryGuidance.temporal(weights[weights.length / 2]), r);
        rows.add(rolloutRow("temporal history guidance w=" + weights[weights.length / 2], v, real,
                (System.nanoTime() - start) / 1e6));
        rows.add(coreDiffusionForcing(window, pretrainSteps, first, real, seed));
        SpaceTimeDenoiser realScore = trainDenoiser(SpaceTimeDenoiser.Attention.FACTORISED, pretrainSteps, false, seed);
        start = System.nanoTime();
        v = realScore.extend(samples, LONG, FRAMES, 4, first, sampling, r);
        rows.add(rolloutRow("bidirectional 9.5.1, sliding windows", v, real, (System.nanoTime() - start) / 1e6));
        SpaceTimeDenoiser fakeScore = SpaceTimeDenoiser.of(SIDE, 4, 32, 2, 2, FRAMES,
                SpaceTimeDenoiser.Attention.FACTORISED, false, seed);
        CurvatureProducts.assign(fakeScore.parameters(), CurvatureProducts.values(realScore.parameters()));
        SelfForcingTrainer sf = SelfForcingTrainer.of(tf, realScore, fakeScore, FRAMES, 4, 2e-4f, 1e-3f);
        for (int s = 0; s < selfForcingSteps; s++) {
            sf.step(clips(4, 1, r.nextLong()), r);
        }
        start = System.nanoTime();
        v = sf.generate(first, LONG, r);
        rows.add(rolloutRow("Self Forcing (4 steps per frame)", v, real, (System.nanoTime() - start) / 1e6));
        return rows;
    }

    /** The core {@link DiffusionForcing} (an MLP over windows of flattened frames) rolled out. */
    private static RolloutRow coreDiffusionForcing(int window, int steps, Tensor first, Tensor real, long seed) {
        int dim = SIDE * SIDE;
        DiffusionForcing df = DiffusionForcing.of(dim, window, DiffusionForcing.Mode.INDEPENDENT, 50, 256, 2, seed);
        SplittableRandom r = new SplittableRandom(seed + 11);
        Optimizer opt = Optimizer.adam(1e-3f);
        for (int s = 0; s < steps; s++) {
            float[][][] c = unpack(clips(8, FRAMES, r.nextLong()));
            float[][] windows = new float[c.length][(window + 1) * dim];
            for (int i = 0; i < c.length; i++) {
                int startFrame = r.nextInt(FRAMES - window);
                for (int k = 0; k <= window; k++) {
                    System.arraycopy(c[i][startFrame + k], 0, windows[i], k * dim, dim);
                }
            }
            df.loss(windows, r).backward();
            opt.step(df.parameters());
        }
        long start = System.nanoTime();
        int n = first.shape().dim(0);
        float[][] f0 = new float[n][];
        float[] fv = first.toArray();
        for (int i = 0; i < n; i++) {
            f0[i] = java.util.Arrays.copyOfRange(fv, i * dim, (i + 1) * dim);
        }
        List<float[][]> frames = new ArrayList<>();
        for (int k = 0; k < window; k++) {
            frames.add(f0);
        }
        float[] out = new float[n * LONG * dim];
        for (int i = 0; i < n; i++) {
            System.arraycopy(f0[i], 0, out, i * LONG * dim, dim);
        }
        for (int t = 1; t < LONG; t++) {
            float[][] ctx = new float[n][window * dim];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < window; k++) {
                    System.arraycopy(frames.get(frames.size() - window + k)[i], 0, ctx[i], k * dim, dim);
                }
            }
            float[][] next = df.next(ctx, 3, 20, r);
            frames.add(next);
            for (int i = 0; i < n; i++) {
                System.arraycopy(next[i], 0, out, (i * LONG + t) * dim, dim);
            }
        }
        return rolloutRow("core DiffusionForcing (MLP)", Tensor.of(Shape.of(n, LONG, SIDE, SIDE), out), real,
                (System.nanoTime() - start) / 1e6);
    }
}
