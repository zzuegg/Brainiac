// Shared by every Brainiac compute shader. Buffers are reached through device addresses
// (GL_EXT_buffer_reference) passed in push constants, so no descriptor sets are needed.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer Floats { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Ints { int v[]; };

#define GROUP 256

// The index of this invocation; dispatches wider than 65535 groups use a second dimension.
uint gid() {
    return (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * GROUP + gl_LocalInvocationID.x;
}

#define NEG_INF uintBitsToFloat(0xff800000u)
#define POS_INF uintBitsToFloat(0x7f800000u)
