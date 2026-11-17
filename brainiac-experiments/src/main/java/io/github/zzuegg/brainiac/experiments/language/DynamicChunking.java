package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.BytePairTokenizer;
import io.github.zzuegg.brainiac.experimental.data.ByteNgramModel;
import io.github.zzuegg.brainiac.experimental.data.EntropyPatcher;
import io.github.zzuegg.brainiac.experimental.nn.BytePatchPooling;
import io.github.zzuegg.brainiac.experimental.nn.DynamicChunker;
import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.experimental.nn.ResidualMixerBlock;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Learned byte chunking (H-Net, {@link DynamicChunker}) against fixed tokenization on a
 * byte-level toy language: words from a 24-word lexicon over the letters {@code a-h},
 * separated by spaces, with a Markov chain over words (each word has three likely
 * successors). Word interiors are predictable, word starts are not, so a good chunker cuts
 * at word boundaries.
 *
 * <p>Every byte model has the same outer parts (byte embedding plus previous-byte embedding,
 * one encoder block, one decoder block, a readout) and the same main network (two blocks of
 * causal RoPE attention); only the middle differs:
 * <ul>
 *   <li><b>per byte</b>: the main network runs on every byte (no compression);</li>
 *   <li><b>BLT</b>: fixed entropy patches from a byte bigram model ({@link EntropyPatcher}),
 *       mean-pooled and handed back one patch late ({@link BytePatchPooling});</li>
 *   <li><b>H-Net</b>: the {@link DynamicChunker}'s learned boundaries, target 3 bytes per
 *       chunk.</li>
 * </ul>
 * A BPE token model (the core {@link BytePairTokenizer}, 32 merges) with the same blocks is the
 * tokenizer baseline. Reported: bits per byte on held-out text and bytes per main-network
 * step (the compression ratio).
 */
public final class DynamicChunking {

    private static final String LETTERS = "abcdefgh";
    private static final String ALPHABET = LETTERS + " ";
    private static final int LENGTH = 48;
    private static final int WIDTH = 16;

    private DynamicChunking() {
    }

    /**
     * One model's result.
     *
     * @param name the model
     * @param bitsPerByte the held-out bits per byte
     * @param bytesPerStep bytes per main-network step
     */
    public record Row(String name, double bitsPerByte, double bytesPerStep) {
    }

    /**
     * Generates the toy text.
     *
     * @param random the random source
     * @param chars the number of characters
     * @return the text
     */
    public static String text(SplittableRandom random, int chars) {
        SplittableRandom lex = new SplittableRandom(17);
        String[] words = new String[24];
        int[][] next = new int[words.length][3];
        for (int w = 0; w < words.length; w++) {
            StringBuilder sb = new StringBuilder();
            int len = 2 + lex.nextInt(5);
            for (int i = 0; i < len; i++) {
                sb.append(LETTERS.charAt(lex.nextInt(LETTERS.length())));
            }
            words[w] = sb.toString();
            for (int k = 0; k < 3; k++) {
                next[w][k] = lex.nextInt(words.length);
            }
        }
        StringBuilder sb = new StringBuilder();
        int w = random.nextInt(words.length);
        while (sb.length() < chars) {
            sb.append(words[w]).append(' ');
            w = random.nextDouble() < 0.8 ? next[w][random.nextInt(3)] : random.nextInt(words.length);
        }
        return sb.substring(0, chars);
    }

    /** The middle of a byte model: maps encoder states to decoder inputs, causally. */
    private interface Middle {
        Var apply(Var x, int[][] bytes, boolean training);

        List<Var> parameters();

        Var auxiliaryLoss();

        double bytesPerStep();
    }

    private static LayerSpec main() {
        return LayerChain.repeat(ResidualMixerBlock.of(RefinedAttention.of(2).causal(true).rotary(10000f)), 2);
    }

    private static LayerSpec block() {
        return ResidualMixerBlock.of(RefinedAttention.of(2).causal(true).rotary(10000f));
    }

    /**
     * Trains and scores the three byte models and the BPE model.
     *
     * @param steps training steps (batches of 16 windows)
     * @param seed the seed
     * @return per byte, BLT, H-Net, BPE
     */
    public static List<Row> run(int steps, long seed) {
        String train = text(new SplittableRandom(seed), 20_000);
        String test = text(new SplittableRandom(seed + 77), 64 * LENGTH);
        List<Row> rows = new ArrayList<>();
        SplittableRandom r1 = new SplittableRandom(seed);
        Layer perByte = main().create(Shape.of(LENGTH, WIDTH), r1);
        rows.add(byteModel("per byte", wrap(perByte, 1), train, test, steps, seed));

        ByteNgramModel ngram = ByteNgramModel.train(train, 2);
        EntropyPatcher patcher = new EntropyPatcher(ngram, EntropyPatcher.Rule.GLOBAL, threshold(ngram, train, 3.0),
                8);
        rows.add(byteModel("BLT entropy patches", blt(patcher, new SplittableRandom(seed)), train, test, steps, seed));

        SplittableRandom r3 = new SplittableRandom(seed);
        DynamicChunker chunker = new DynamicChunker(WIDTH, main().create(Shape.of(LENGTH, WIDTH), r3), 3f, 0.03f, r3);
        rows.add(byteModel("H-Net dynamic chunks", hnet(chunker), train, test, steps, seed));
        rows.add(bpe(train, test, steps, seed));
        return rows;
    }

    /** The global entropy threshold that cuts the training text into patches of about {@code ratio} bytes. */
    private static double threshold(ByteNgramModel ngram, String text, double ratio) {
        double[] h = ngram.entropies(text.substring(0, 4000).getBytes(StandardCharsets.UTF_8));
        double[] sorted = h.clone();
        java.util.Arrays.sort(sorted);
        return sorted[(int) (sorted.length * (1 - 1 / ratio))];
    }

    private static Middle wrap(Layer layer, double ratio) {
        return new Middle() {
            @Override
            public Var apply(Var x, int[][] bytes, boolean training) {
                return layer.forward(x, training);
            }

            @Override
            public List<Var> parameters() {
                return layer.parameters();
            }

            @Override
            public Var auxiliaryLoss() {
                return layer.auxiliaryLoss();
            }

            @Override
            public double bytesPerStep() {
                return ratio;
            }
        };
    }

    private static Middle hnet(DynamicChunker chunker) {
        return new Middle() {
            @Override
            public Var apply(Var x, int[][] bytes, boolean training) {
                return chunker.forward(x, training);
            }

            @Override
            public List<Var> parameters() {
                return chunker.parameters();
            }

            @Override
            public Var auxiliaryLoss() {
                return chunker.auxiliaryLoss();
            }

            @Override
            public double bytesPerStep() {
                return chunker.lastBytesPerChunk();
            }
        };
    }

    private static Middle blt(EntropyPatcher patcher, SplittableRandom random) {
        Layer latent = main().create(Shape.of(LENGTH, WIDTH), random);
        BytePatchPooling pooling = new BytePatchPooling(WIDTH, WIDTH, random);
        double[] ratio = {1};
        return new Middle() {
            @Override
            public Var apply(Var x, int[][] bytes, boolean training) {
                int b = bytes.length;
                int t = bytes[0].length;
                List<Var> outs = new ArrayList<>();
                long patches = 0;
                for (int n = 0; n < b; n++) {
                    byte[] raw = new byte[t];
                    for (int i = 0; i < t; i++) {
                        raw[i] = (byte) ALPHABET.charAt(bytes[n][i]);
                    }
                    int[] ids = patcher.patchIds(raw);
                    int count = ids[t - 1] + 1;
                    patches += count;
                    Var xs = x.slice(0, n, n + 1).reshape(t, WIDTH);
                    Var z = latent.forward(pooling.pool(xs, ids).reshape(1, count, WIDTH), training).reshape(count,
                            WIDTH);
                    outs.add(pooling.unpool(z, ids).add(xs).reshape(1, t, WIDTH));
                }
                ratio[0] = b * t / (double) patches;
                return Var.concat(0, outs.toArray(Var[]::new));
            }

            @Override
            public List<Var> parameters() {
                List<Var> p = new ArrayList<>(latent.parameters());
                p.addAll(pooling.parameters());
                return p;
            }

            @Override
            public Var auxiliaryLoss() {
                return null;
            }

            @Override
            public double bytesPerStep() {
                return ratio[0];
            }
        };
    }

    private static int[][] windows(String text, SplittableRandom random, int count) {
        int[][] out = new int[count][LENGTH];
        for (int n = 0; n < count; n++) {
            int start = random.nextInt(text.length() - LENGTH);
            for (int i = 0; i < LENGTH; i++) {
                out[n][i] = ALPHABET.indexOf(text.charAt(start + i));
            }
        }
        return out;
    }

    private static int[][] chopped(String text, int count) {
        int[][] out = new int[count][LENGTH];
        for (int n = 0; n < count; n++) {
            for (int i = 0; i < LENGTH; i++) {
                out[n][i] = ALPHABET.indexOf(text.charAt(n * LENGTH + i));
            }
        }
        return out;
    }

    private static Row byteModel(String name, Middle middle, String train, String test, int steps, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int vocab = ALPHABET.length();
        Var embed = Var.param(Tensor.normal(Shape.of(vocab, WIDTH), 0, 1, random));
        Var previous = Var.param(Tensor.normal(Shape.of(vocab, WIDTH), 0, 1, random));
        Layer encoder = block().create(Shape.of(LENGTH, WIDTH), random);
        Layer decoder = block().create(Shape.of(LENGTH, WIDTH), random);
        Var readout = Var.param(Tensor.uniform(Shape.of(WIDTH, vocab), -1, 1, random)
                .mul((float) Math.sqrt(3.0 / WIDTH)));
        List<Var> params = new ArrayList<>(List.of(embed, previous, readout));
        params.addAll(encoder.parameters());
        params.addAll(decoder.parameters());
        params.addAll(middle.parameters());
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < steps; s++) {
            int[][] batch = windows(train, random, 16);
            Var loss = byteLoss(batch, embed, previous, encoder, middle, decoder, readout, true);
            Var aux = middle.auxiliaryLoss();
            (aux == null ? loss : loss.add(aux)).backward();
            opt.step(params);
        }
        int[][] held = chopped(test, test.length() / LENGTH);
        double nats = byteLoss(held, embed, previous, encoder, middle, decoder, readout, false).value().item();
        return new Row(name, nats / Math.log(2), middle.bytesPerStep());
    }

    private static Var byteLoss(int[][] bytes, Var embed, Var previous, Layer encoder, Middle middle, Layer decoder,
            Var readout, boolean training) {
        int b = bytes.length;
        int t = bytes[0].length;
        float[] ids = new float[b * t];
        float[] prev = new float[b * t];
        float[] next = new float[b * (t - 1)];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                ids[i * t + j] = bytes[i][j];
                prev[i * t + j] = bytes[i][Math.max(0, j - 1)];
                if (j < t - 1) {
                    next[i * (t - 1) + j] = bytes[i][j + 1];
                }
            }
        }
        Var x = embed.gather(0, Tensor.of(Shape.of(b * t), ids)).add(previous.gather(0, Tensor.of(Shape.of(b * t),
                prev))).reshape(b, t, WIDTH);
        Var h = decoder.forward(middle.apply(encoder.forward(x, training), bytes, training), training);
        Var normed = h.div(h.square().mean(-1, true).add(1e-6f).sqrt());
        int vocab = readout.shape().dim(1);
        Var logits = normed.matmul(readout).slice(1, 0, t - 1).reshape(b * (t - 1), vocab);
        return Functional.crossEntropy(logits, Tensor.of(Shape.of(next.length), next));
    }

    /** The BPE token model: same outer blocks and main network, over tokens. */
    private static Row bpe(String train, String test, int steps, long seed) {
        BytePairTokenizer tok = BytePairTokenizer.train(train, 256 + 32);
        int[] trainIds = tok.encode(train);
        int[] testIds = tok.encode(test);
        int vocab = tok.size();
        int len = 16;
        SplittableRandom random = new SplittableRandom(seed);
        Var embed = Var.param(Tensor.normal(Shape.of(vocab, WIDTH), 0, 1, random));
        Layer body = LayerChain.of(block(), main(), block()).create(Shape.of(len, WIDTH), random);
        Var readout = Var.param(Tensor.uniform(Shape.of(WIDTH, vocab), -1, 1, random)
                .mul((float) Math.sqrt(3.0 / WIDTH)));
        List<Var> params = new ArrayList<>(List.of(embed, readout));
        params.addAll(body.parameters());
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < steps; s++) {
            int[][] batch = new int[16][len];
            for (int[] row : batch) {
                int start = random.nextInt(trainIds.length - len);
                System.arraycopy(trainIds, start, row, 0, len);
            }
            tokenLoss(batch, embed, body, readout, true).backward();
            opt.step(params);
        }
        int windows = (testIds.length - 1) / len;
        int[][] held = new int[windows][len];
        long bytes = 0;
        for (int n = 0; n < windows; n++) {
            System.arraycopy(testIds, n * len, held[n], 0, len);
            for (int j = 1; j < len; j++) {
                bytes += tok.tokenBytes(held[n][j]).length;
            }
        }
        double meanNats = tokenLoss(held, embed, body, readout, false).value().item();
        double totalBits = meanNats * windows * (len - 1) / Math.log(2);
        return new Row("BPE tokens (32 merges)", totalBits / bytes, (double) bytes / (windows * (len - 1)));
    }

    private static Var tokenLoss(int[][] tokens, Var embed, Layer body, Var readout, boolean training) {
        int b = tokens.length;
        int t = tokens[0].length;
        float[] ids = new float[b * t];
        float[] next = new float[b * (t - 1)];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                ids[i * t + j] = tokens[i][j];
                if (j < t - 1) {
                    next[i * (t - 1) + j] = tokens[i][j + 1];
                }
            }
        }
        Var h = body.forward(embed.gather(0, Tensor.of(Shape.of(b * t), ids)).reshape(b, t, WIDTH), training);
        Var normed = h.div(h.square().mean(-1, true).add(1e-6f).sqrt());
        int vocab = readout.shape().dim(1);
        Var logits = normed.matmul(readout).slice(1, 0, t - 1).reshape(b * (t - 1), vocab);
        return Functional.crossEntropy(logits, Tensor.of(Shape.of(next.length), next));
    }

    /**
     * Formats rows.
     *
     * @param rows the rows
     * @return a text table
     */
    public static String table(List<Row> rows) {
        StringBuilder sb = new StringBuilder(String.format("%-24s %13s %11s%n", "model", "bits/byte", "bytes/step"));
        for (Row r : rows) {
            sb.append(String.format("%-24s %13.4f %11.2f%n", r.name(), r.bitsPerByte(), r.bytesPerStep()));
        }
        return sb.toString();
    }

    /**
     * Runs the comparison.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.print(table(run(600, 1)));
    }
}
