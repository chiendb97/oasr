// Copyright 2024 OASR Authors
// SPDX-License-Identifier: Apache-2.0
//
// The element functors of the scaled GEMM's epilogue, shared by both kernel
// generations: the CUTLASS 2.x ``mma.sync`` lane (``scaled_mm_epilogue.h``) and
// the native CUTLASS 3.x lanes (``scaled_mm_epilogue_sm90.h``).  One definition
// of each rounding step is what keeps the two generations' epilogues computing
// the same values.

#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/float8.h>

#include <oasr/common/types.h>

namespace oasr {
namespace scaled_mm {

// OASR ActivationType -> the CUTLASS functor the epilogue applies.  The two GELU
// spellings stay distinct: ``GELU`` is the tanh approximation, ``GELU_ERF``
// exact erf (AGENTS.md: a fused epilogue must not change checkpoint semantics).
template <ActivationType act>
struct ScaledMmActivation;

template <>
struct ScaledMmActivation<ActivationType::IDENTITY> {
    template <class T>
    using Fn = cutlass::epilogue::thread::Identity<T>;
};

template <>
struct ScaledMmActivation<ActivationType::RELU> {
    template <class T>
    using Fn = cutlass::epilogue::thread::ReLu<T>;
};

template <>
struct ScaledMmActivation<ActivationType::SWISH> {
    template <class T>
    using Fn = cutlass::epilogue::thread::SiLu<T>;
};

template <>
struct ScaledMmActivation<ActivationType::GELU> {
    template <class T>
    using Fn = cutlass::epilogue::thread::GELU_taylor<T>;
};

template <>
struct ScaledMmActivation<ActivationType::GELU_ERF> {
    template <class T>
    using Fn = cutlass::epilogue::thread::GELU<T>;
};

// Whether an ActivationType id is one the scaled GEMM fuses.
CUTLASS_HOST_DEVICE
constexpr bool isScaledMmActivation(int activation) {
    return activation == static_cast<int>(ActivationType::IDENTITY) ||
           activation == static_cast<int>(ActivationType::RELU) ||
           activation == static_cast<int>(ActivationType::SWISH) ||
           activation == static_cast<int>(ActivationType::GELU) ||
           activation == static_cast<int>(ActivationType::GELU_ERF);
}

// The activation as a runtime argument (the native lanes).  A CUTLASS 3.x
// kernel compiles 10-20x slower than its 2.x counterpart, so one kernel per
// activation would make the native modules' cold build impractical; instead the
// activation id rides in the epilogue's parameters and a uniform switch picks
// the same CUTLASS functor ``ScaledMmActivation`` names -- the same arithmetic
// per element, one kernel for all five.
struct ScaledMmActivationArgs {
    int activation = static_cast<int>(ActivationType::IDENTITY);
};

template <class T>
struct ScaledMmRuntimeActivation {
    // The scalar form: CUTLASS's Sm90Compute looks up ``Arguments`` on it.
    using Arguments = ScaledMmActivationArgs;

    CUTLASS_HOST_DEVICE
    T operator()(T const& x, Arguments const& args) const {
        cutlass::Array<T, 1> frag;
        frag[0] = x;
        return ScaledMmRuntimeActivation<cutlass::Array<T, 1>>{}(frag, args)[0];
    }
};

template <class T, int N>
struct ScaledMmRuntimeActivation<cutlass::Array<T, N>> {
    using Arguments = ScaledMmActivationArgs;
    using Frag = cutlass::Array<T, N>;

    CUTLASS_HOST_DEVICE
    Frag operator()(Frag const& x, Arguments const& args) const {
        switch (static_cast<ActivationType>(args.activation)) {
            case ActivationType::RELU:
                return ScaledMmActivation<ActivationType::RELU>::Fn<Frag>{}(x);
            case ActivationType::SWISH:
                return ScaledMmActivation<ActivationType::SWISH>::Fn<Frag>{}(x);
            case ActivationType::GELU:
                return ScaledMmActivation<ActivationType::GELU>::Fn<Frag>{}(x);
            case ActivationType::GELU_ERF:
                return ScaledMmActivation<ActivationType::GELU_ERF>::Fn<Frag>{}(x);
            default:  // IDENTITY; the launcher refuses any other id
                return x;
        }
    }
};

// ``clamp(rint(y * m), -127, 127)`` per element, in fp32 with an IEEE multiply
// (``__fmul_rn``: never contracted into an FMA under --use_fast_math).  The result
// is integral and in range, so the store's float->int8 conversion is exact.
template <class T>
struct MulRintClampInt8 {
    CUTLASS_DEVICE
    T operator()(T const& y, T const& mul) const {
        const float r = rintf(__fmul_rn(y, mul));
        return fminf(fmaxf(r, -127.0f), 127.0f);
    }
};

template <int N>
struct MulRintClampInt8<cutlass::Array<float, N>> {
    CUTLASS_DEVICE
    cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& y,
                                        cutlass::Array<float, N> const& mul) const {
        cutlass::Array<float, N> q;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            q[i] = MulRintClampInt8<float>{}(y[i], mul[i]);
        }
        return q;
    }
};

// ``clamp(y * m, -448, 448)`` per element, an IEEE multiply; the e4m3 store's
// round-to-nearest-even conversion (``cvt.rn.satfinite.e4m3x2.f32``, as
// ``quantize.cuh`` uses) then rounds it, so the clamp only makes the saturation
// explicit.
template <class T>
struct MulSatFp8 {
    CUTLASS_DEVICE
    T operator()(T const& y, T const& mul) const {
        return fminf(fmaxf(__fmul_rn(y, mul), -448.0f), 448.0f);
    }
};

template <int N>
struct MulSatFp8<cutlass::Array<float, N>> {
    CUTLASS_DEVICE
    cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& y,
                                        cutlass::Array<float, N> const& mul) const {
        cutlass::Array<float, N> q;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            q[i] = MulSatFp8<float>{}(y[i], mul[i]);
        }
        return q;
    }
};

// The per-element quantize of each 8-bit output format.
template <typename ElementQ>
struct ScaledMmQuantize;

template <>
struct ScaledMmQuantize<int8_t> {
    template <class T>
    using Fn = MulRintClampInt8<T>;
};

template <>
struct ScaledMmQuantize<cutlass::float_e4m3_t> {
    template <class T>
    using Fn = MulSatFp8<T>;
};

}  // namespace scaled_mm
}  // namespace oasr
