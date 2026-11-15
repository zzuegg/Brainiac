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
        layer("dropout", Dropout.class, new LayerCodec<>() {
            @Override
            public Config write(Dropout d) {
                return Config.builder().put("rate", d.rate()).build();
            }

            @Override
            public Dropout read(Config c) {
                return new Dropout(c.getFloat("rate"), new SplittableRandom());
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
        for (Activation a : new Activation[] {Activation.IDENTITY, Activation.RELU, Activation.SIGMOID,
                Activation.TANH, Activation.SOFTMAX}) {
            activation(a.toString(), c -> a);
        }
        activation("prelu", c -> Activation.prelu());
        activation("swish", c -> Activation.swish());
    }

    private Registry() {
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
