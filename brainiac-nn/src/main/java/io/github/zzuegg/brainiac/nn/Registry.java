package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/** The layer types and activations that model files can name. */
final class Registry {

    record LayerType<L extends Layer>(String name, Class<L> type, LayerCodec<L> codec) {

        Config write(Layer layer) {
            return codec.write(type.cast(layer));
        }
    }

    private static final Map<String, LayerType<?>> LAYERS_BY_NAME = new ConcurrentHashMap<>();
    private static final Map<Class<?>, LayerType<?>> LAYERS_BY_CLASS = new ConcurrentHashMap<>();
    private static final Map<String, Function<Config, Activation>> ACTIVATIONS = new ConcurrentHashMap<>();

    static {
        layer("dense", Dense.class, new LayerCodec<>() {
            @Override
            public Config write(Dense d) {
                return Config.builder()
                        .put("inputs", d.weight().shape().dim(0))
                        .put("units", d.weight().shape().dim(1))
                        .put("activation", d.activation())
                        .build();
            }

            @Override
            public Dense read(Config c) {
                int inputs = c.getInt("inputs");
                int units = c.getInt("units");
                return new Dense(inputs, units, c.getActivation("activation"),
                        Tensor.zeros(Shape.of(inputs, units)), Tensor.zeros(Shape.of(units)));
            }
        });
        layer("switching_dense", SwitchingDense.class, new LayerCodec<>() {
            @Override
            public Config write(SwitchingDense d) {
                Config.Builder b = Config.builder()
                        .put("inputs", d.inputSize())
                        .put("units", d.units())
                        .put("hard", d.hard())
                        .put("count", d.candidates().size());
                for (int i = 0; i < d.candidates().size(); i++) {
                    b.put("a" + i, d.candidates().get(i));
                }
                return b.build();
            }

            @Override
            public SwitchingDense read(Config c) {
                int inputs = c.getInt("inputs");
                int units = c.getInt("units");
                Activation[] candidates = new Activation[c.getInt("count")];
                for (int i = 0; i < candidates.length; i++) {
                    candidates[i] = c.getActivation("a" + i);
                }
                Shape w = Shape.of(inputs, units);
                Shape kw = Shape.of(candidates.length, units);
                return new SwitchingDense(inputs, units, candidates, c.getBoolean("hard"), Tensor.zeros(w),
                        Tensor.zeros(Shape.of(units)), Tensor.zeros(w), Tensor.zeros(Shape.of(units)),
                        Tensor.zeros(kw), Tensor.zeros(kw));
            }
        });
        layer("dropout", Dropout.class, new LayerCodec<>() {
            @Override
            public Config write(Dropout d) {
                return Config.builder().put("rate", d.rate()).put("channelwise", d.channelwise()).build();
            }

            @Override
            public Dropout read(Config c) {
                return new Dropout(c.getFloat("rate"), new SplittableRandom(),
                        c.has("channelwise") && c.getBoolean("channelwise"));
            }
        });
        layer("batchnorm", BatchNorm.class, new LayerCodec<>() {
            @Override
            public Config write(BatchNorm b) {
                return Config.builder()
                        .put("features", b.features())
                        .put("momentum", b.momentum())
                        .put("epsilon", b.epsilon())
                        .build();
            }

            @Override
            public BatchNorm read(Config c) {
                return new BatchNorm(c.getInt("features"), c.getFloat("momentum"), c.getFloat("epsilon"));
            }
        });
        simple("conv", Conv.class, Conv::write, Conv::read);
        simple("pool", Pool.class, Pool::write, Pool::read);
        simple("global_pool", GlobalPool.class, GlobalPool::write, GlobalPool::read);
        simple("adaptive_pool", AdaptivePool.class, AdaptivePool::write, AdaptivePool::read);
        simple("pad", Pad.class, Pad::write, Pad::read);
        simple("reshape", Reshape.class, Reshape::write, Reshape::read);
        simple("permute", Permute.class, Permute::write, Permute::read);
        simple("resize", Resize.class, Resize::write, Resize::read);
        simple("pixel_shuffle", PixelShuffle.class, PixelShuffle::write, PixelShuffle::read);
        simple("layer_norm", LayerNorm.class, LayerNorm::write, LayerNorm::read);
        simple("group_norm", GroupNorm.class, GroupNorm::write, GroupNorm::read);
        simple("recurrent", Recurrent.class, Recurrent::write, Recurrent::read);
        simple("attention", MultiHeadAttention.class, MultiHeadAttention::write, MultiHeadAttention::read);
        simple("transformer", TransformerBlock.class, TransformerBlock::write, TransformerBlock::read);
        simple("positions", PositionalEncoding.class, PositionalEncoding::write, PositionalEncoding::read);
        simple("sequence_pool", SequencePool.class, SequencePool::write, SequencePool::read);
        simple("embedding", Embedding.class, Embedding::write, Embedding::read);
        simple("padding_mask", PaddingMask.class, PaddingMask::write, PaddingMask::read);
        simple("masked_pool", MaskedPool.class, MaskedPool::write, MaskedPool::read);
        simple("brain", Brain.class, Brain::write, Brain::read);
        simple("mel_spectrogram", MelSpectrogram.class, MelSpectrogram::write, MelSpectrogram::read);
        simple("relative_attention", RelativeAttention.class, RelativeAttention::write, RelativeAttention::read);
        simple("hifigan_block", HifiGanBlock.class, HifiGanBlock::write, HifiGanBlock::read);
        simple("wavenet", WaveNet.class, WaveNet::write, WaveNet::read);
        simple("style_norm", StyleNorm.class, StyleNorm::write, StyleNorm::read);
        simple("adain_res_block", AdainResBlock.class, AdainResBlock::write, AdainResBlock::read);
        simple("adain_snake_block", AdainSnakeBlock.class, AdainSnakeBlock::write, AdainSnakeBlock::read);
        DiffusionCodecs.register();
        simple("convnext1d", ConvNeXt1d.class, ConvNeXt1d::write, ConvNeXt1d::read);
        simple("dit_block", DitBlock.class, DitBlock::write, DitBlock::read);
        simple("latent_attention", LatentAttention.class, LatentAttention::write, LatentAttention::read);
        simple("snake", Snake.class, Snake::write, Snake::read);
        for (Activation a : new Activation[] {Activation.IDENTITY, Activation.RELU, Activation.SIGMOID,
                Activation.TANH, Activation.SILU, Activation.GELU, Activation.GELU_TANH, Activation.QUICK_GELU,
                Activation.SOFTPLUS, Activation.MISH, Activation.SELU, Activation.RELU6, Activation.HARD_TANH, Activation.HARD_SIGMOID,
                Activation.HARD_SWISH, Activation.SOFTSIGN}) {
            activation(a.name(), c -> a);
        }
        activation("softmax", c -> c.has("axis") ? Activation.softmax(c.getInt("axis")) : Activation.SOFTMAX);
        activation("log_softmax",
                c -> c.has("axis") ? Activation.logSoftmax(c.getInt("axis")) : Activation.LOG_SOFTMAX);
        activation("leaky_relu", c -> Activation.leakyRelu(c.getFloat("slope")));
        activation("elu", c -> Activation.elu(c.getFloat("alpha")));
        activation("split", SplitActivation.factory());
        activation("mix", MixedActivation.factory());
        activation("per_unit", PerUnitActivation.factory());
        activation("prelu", c -> Activation.prelu());
        activation("swish", c -> Activation.swish());
    }

    private Registry() {
    }

    static <L extends Layer> void simple(String name, Class<L> type, java.util.function.Function<L, Config> write,
            Function<Config, L> read) {
        layer(name, type, new LayerCodec<>() {
            @Override
            public Config write(L layer) {
                return write.apply(layer);
            }

            @Override
            public L read(Config config) {
                return read.apply(config);
            }
        });
    }

    static <L extends Layer> void layer(String name, Class<L> type, LayerCodec<L> codec) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(codec, "codec");
        if (Activation.class.isAssignableFrom(type)) {
            throw new IllegalArgumentException("register activations with Activation.register(...)");
        }
        LayerType<L> entry = new LayerType<>(name, type, codec);
        LayerType<?> old = LAYERS_BY_NAME.get(name);
        if (old != null && old.type() != type) {
            throw new IllegalArgumentException("layer type `" + name + "` is already registered for " + old.type());
        }
        LAYERS_BY_NAME.put(name, entry);
        LAYERS_BY_CLASS.put(type, entry);
    }

    static void activation(String name, Function<Config, Activation> factory) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(factory, "factory");
        ACTIVATIONS.put(name, factory);
    }

    static LayerType<?> layerType(Class<?> type) {
        return LAYERS_BY_CLASS.get(type);
    }

    static LayerType<?> layerType(String name) {
        return LAYERS_BY_NAME.get(name);
    }

    static Function<Config, Activation> activation(String name) {
        return ACTIVATIONS.get(name);
    }
}
