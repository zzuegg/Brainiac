package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class ToolsTest {

    public enum Size { SMALL, LARGE }

    public record Item(@Tools.Param("the product") String product, int quantity, Optional<Size> size) {
    }

    public record Order(int number, List<Item> items, double total) {
    }

    public static final class Shop {
        int orders;

        @Tools.Tool("Looks up the price of a product, in euros")
        public double price(@Tools.Param("the product's name") String product, Optional<Integer> quantity) {
            return ("tea".equals(product) ? 2.5 : 4) * quantity.orElse(1);
        }

        @Tools.Tool(value = "Places an order", name = "place_order")
        public Order order(List<Item> items, @Tools.Param(value = "a note", optional = true) String note) {
            double total = 0;
            for (Item i : items) {
                total += price(i.product(), Optional.of(i.quantity()));
            }
            return new Order(++orders, items, total);
        }

        @Tools.Tool("Forgets every order")
        public void reset() {
            orders = 0;
        }

        public String notATool() {
            return "";
        }
    }

    private static Map<String, Object> args(String json) {
        return Json.object(Json.parse(json));
    }

    @Test
    void derivesSchemasFromMethods() {
        List<Agent.Tool> tools = Tools.from(new Shop());
        assertEquals(List.of("place_order", "price", "reset"), tools.stream().map(Agent.Tool::name).toList());
        Agent.Tool price = tools.get(1);
        assertEquals("Looks up the price of a product, in euros", price.description());
        assertEquals("{\"type\":\"object\",\"properties\":{\"product\":{\"type\":\"string\",\"description\":"
                + "\"the product's name\"},\"quantity\":{\"type\":\"integer\"}},\"required\":[\"product\"]}",
                Json.write(price.parameters()));
        String order = Json.write(tools.getFirst().parameters());
        assertTrue(order.contains("\"items\":{\"type\":\"array\",\"items\":{\"type\":\"object\",\"properties\":"
                + "{\"product\":{\"type\":\"string\",\"description\":\"the product\"},\"quantity\":{\"type\":\"integer\"},"
                + "\"size\":{\"type\":\"string\",\"enum\":[\"SMALL\",\"LARGE\"]}},\"required\":[\"product\",\"quantity\"]}}"),
                order);
        assertTrue(order.endsWith("\"required\":[\"items\"]}"), order);
    }

    @Test
    void convertsArgumentsAndResults() {
        List<Agent.Tool> tools = Tools.from(new Shop());
        assertEquals("7.5", tools.get(1).run().apply(args("{\"product\":\"tea\",\"quantity\":3}")));
        assertEquals("4", tools.get(1).run().apply(args("{\"product\":\"cake\"}")));
        String placed = tools.getFirst().run().apply(args(
                "{\"items\":[{\"product\":\"tea\",\"quantity\":2,\"size\":\"large\"},{\"product\":\"cake\",\"quantity\":1}]}"));
        assertEquals("{\"number\":1,\"items\":[{\"product\":\"tea\",\"quantity\":2,\"size\":\"LARGE\"},"
                + "{\"product\":\"cake\",\"quantity\":1,\"size\":null}],\"total\":9}", placed);
        assertEquals("ok", tools.get(2).run().apply(Map.of()));
    }

    @Test
    void badArgumentsSayWhatIsWrong() {
        Agent.Tool order = Tools.from(new Shop()).getFirst();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> order.run().apply(args("{\"items\":[{\"product\":\"tea\",\"quantity\":1.5}]}")));
        assertEquals("items[0].quantity must be a whole number, got 1.5", e.getMessage());
        e = assertThrows(IllegalArgumentException.class, () -> order.run().apply(args("{\"items\":[{\"quantity\":1}]}")));
        assertEquals("missing argument items[0].product", e.getMessage());
        e = assertThrows(IllegalArgumentException.class,
                () -> order.run().apply(args("{\"items\":[{\"product\":\"x\",\"quantity\":1,\"size\":\"huge\"}]}")));
        assertEquals("items[0].size must be one of [SMALL, LARGE], got \"huge\"", e.getMessage());
    }

    public record Add(@Tools.Param("first addend") long a, long b) {
    }

    @Test
    void recordTools() {
        Agent.Tool add = Tools.of("add", "Adds two numbers", Add.class, x -> x.a() + x.b());
        assertEquals("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\",\"description\":\"first addend\"},"
                + "\"b\":{\"type\":\"integer\"}},\"required\":[\"a\",\"b\"]}", Json.write(add.parameters()));
        assertEquals("126181", add.run().apply(args("{\"a\":48213,\"b\":\"77968\"}")));
    }

    @Test
    void staticToolsAndUnsupportedTypes() {
        assertThrows(IllegalArgumentException.class, () -> Tools.from(String.class));
        assertThrows(IllegalArgumentException.class, () -> Tools.schema(Thread.class));
        assertEquals("{\"type\":\"object\",\"additionalProperties\":{\"type\":\"array\",\"items\":{\"type\":\"number\"}}}",
                Json.write(Tools.schema(new java.lang.reflect.ParameterizedType() {
                    public java.lang.reflect.Type[] getActualTypeArguments() {
                        return new java.lang.reflect.Type[] {String.class, double[].class};
                    }

                    public java.lang.reflect.Type getRawType() {
                        return Map.class;
                    }

                    public java.lang.reflect.Type getOwnerType() {
                        return null;
                    }
                })));
    }
}
