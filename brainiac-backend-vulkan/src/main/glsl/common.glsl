// Shared by every Brainiac compute shader. Buffers are reached through device addresses
// (GL_EXT_buffer_reference) passed in push constants, so no descriptor sets are needed.
//
// Each shader is compiled once per number format, chosen with one of TYPE_F32 (default),
// TYPE_F64, TYPE_F16 or TYPE_BF16:
//   T              the storage type of tensor values
//   ACC            the type computed in: double for float64, float otherwise
//   Buf, In        buffers of T values (In: read-only inputs, never aliased by an output)
//   LOAD(b, i)     reads element i as ACC
//   STORE(b, i, v) writes v (an ACC), rounding to the storage type
//   K(c)           a short constant (exact in float) as an ACC; long ones are defined per format
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#if defined(TYPE_F64)
#define T double
#define ACC double
#define T_ALIGN 8
#define K(c) double(c)
#define LOAD(b, i) (b.v[i])
#define STORE(b, i, x) b.v[i] = (x)
#elif defined(TYPE_F16)
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define T float16_t
#define ACC float
#define T_ALIGN 2
#define K(c) c
#define LOAD(b, i) float(b.v[i])
#define STORE(b, i, x) b.v[i] = float16_t(x)
#elif defined(TYPE_BF16)
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#define T uint16_t
#define ACC float
#define T_ALIGN 2
#define K(c) c
#define LOAD(b, i) uintBitsToFloat(uint(b.v[i]) << 16)
#define STORE(b, i, x) b.v[i] = to_bf16(x)
// rounds to nearest even, as DType.floatToBfloat16 does
uint16_t to_bf16(float f) {
    uint bits = floatBitsToUint(f);
    if (isnan(f)) {
        return uint16_t((bits >> 16) | 0x40u);
    }
    return uint16_t((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}
#else
#define TYPE_F32
#define T float
#define ACC float
#define T_ALIGN 4
#define K(c) c
#define LOAD(b, i) (b.v[i])
#define STORE(b, i, x) b.v[i] = (x)
#endif

layout(buffer_reference, std430, buffer_reference_align = T_ALIGN) buffer Buf { T v[]; };
layout(buffer_reference, std430, buffer_reference_align = T_ALIGN) restrict readonly buffer In { T v[]; };
// float32 whatever the format: random numbers, indices
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Floats { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer InFloats { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer InInts { int v[]; };

#define GROUP 256

// The index of this invocation; dispatches wider than 65535 groups use a second dimension.
uint gid() {
    return (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * GROUP + gl_LocalInvocationID.x;
}

#define NEG_INF ACC(uintBitsToFloat(0xff800000u))
#define POS_INF ACC(uintBitsToFloat(0x7f800000u))
#define NAN_ ACC(uintBitsToFloat(0x7fc00000u))
