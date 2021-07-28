#pragma once
#include <cstring>
#include <stdint.h>

enum BSDFLobe
{
    NullLobe                 = 0,
    GlossyReflectionLobe     = (1 << 0),
    GlossyTransmissionLobe   = (1 << 1),
    DiffuseReflectionLobe    = (1 << 2),
    DiffuseTransmissionLobe  = (1 << 3),
    SpecularReflectionLobe   = (1 << 4),
    SpecularTransmissionLobe = (1 << 5),
    AnisotropicLobe          = (1 << 6),
    ForwardLobe              = (1 << 7),

    GlossyLobe   =   GlossyReflectionLobe |   GlossyTransmissionLobe,
    DiffuseLobe  =  DiffuseReflectionLobe |  DiffuseTransmissionLobe,
    SmoothLobe = GlossyLobe | DiffuseLobe,
    SpecularLobe = SpecularReflectionLobe | SpecularTransmissionLobe,

    TransmissiveLobe = GlossyTransmissionLobe | DiffuseTransmissionLobe | SpecularTransmissionLobe,
    ReflectiveLobe   = GlossyReflectionLobe   | DiffuseReflectionLobe   | SpecularReflectionLobe,

    AllLobes = TransmissiveLobe | ReflectiveLobe | AnisotropicLobe,
    AllButSpecular = ~(SpecularLobe | ForwardLobe)
};

/**
 * \brief This list of flags is used to classify the different types of lobes
 * that are implemented in a BSDF instance.
 *
 * They are also useful for picking out individual components, e.g., by setting
 * combinations in \ref BSDFContext::type_mask.
 */
enum class BSDFFlags : uint32_t {
    // =============================================================
    //                      BSDF lobe types
    // =============================================================

    /// No flags set (default value)
    None                 = 0x00000,

    /// 'null' scattering event, i.e. particles do not undergo deflection
    Null                 = 0x00001,

    /// Ideally diffuse reflection
    DiffuseReflection    = 0x00002,

    /// Ideally diffuse transmission
    DiffuseTransmission  = 0x00004,

    /// Glossy reflection
    GlossyReflection     = 0x00008,

    /// Glossy transmission
    GlossyTransmission   = 0x00010,

    /// Reflection into a discrete set of directions
    DeltaReflection      = 0x00020,

    /// Transmission into a discrete set of directions
    DeltaTransmission    = 0x00040,

    /// Reflection into a 1D space of directions
    Delta1DReflection    = 0x00080,

    /// Transmission into a 1D space of directions
    Delta1DTransmission  = 0x00100,

    // =============================================================
    //!                   Other lobe attributes
    // =============================================================

    /// The lobe is not invariant to rotation around the normal
    Anisotropic          = 0x01000,

    /// The BSDF depends on the UV coordinates
    SpatiallyVarying     = 0x02000,

    /// Flags non-symmetry (e.g. transmission in dielectric materials)
    NonSymmetric         = 0x04000,

    /// Supports interactions on the front-facing side
    FrontSide            = 0x08000,

    /// Supports interactions on the back-facing side
    BackSide             = 0x10000,

    /// Does the implementation require access to texture-space differentials
    NeedsDifferentials   = 0x20000,

    // =============================================================
    //!                 Compound lobe attributes
    // =============================================================

    /// Any reflection component (scattering into discrete, 1D, or 2D set of directions)
    Reflection   = DiffuseReflection | DeltaReflection |
                   Delta1DReflection | GlossyReflection,

    /// Any transmission component (scattering into discrete, 1D, or 2D set of directions)
    Transmission = DiffuseTransmission | DeltaTransmission |
                   Delta1DTransmission | GlossyTransmission | Null,

    /// Diffuse scattering into a 2D set of directions
    Diffuse      = DiffuseReflection | DiffuseTransmission,

    /// Non-diffuse scattering into a 2D set of directions
    Glossy       = GlossyReflection | GlossyTransmission,

    /// Scattering into a 2D set of directions
    Smooth       = Diffuse | Glossy,

    /// Scattering into a discrete set of directions
    Delta        = Null | DeltaReflection | DeltaTransmission,

    /// Scattering into a 1D space of directions
    Delta1D      = Delta1DReflection | Delta1DTransmission,

    /// Any kind of scattering
    All          = Diffuse | Glossy | Delta | Delta1D
};

RT_FUNCTION constexpr uint32_t operator |(BSDFFlags f1, BSDFFlags f2)     { return (uint32_t) f1 | (uint32_t) f2; }
RT_FUNCTION constexpr uint32_t operator |(uint32_t f1, BSDFFlags f2)      { return f1 | (uint32_t) f2; }
RT_FUNCTION constexpr uint32_t operator &(BSDFFlags f1, BSDFFlags f2)     { return (uint32_t) f1 & (uint32_t) f2; }
RT_FUNCTION constexpr uint32_t operator &(uint32_t f1, BSDFFlags f2)      { return f1 & (uint32_t) f2; }
RT_FUNCTION constexpr uint32_t operator ~(BSDFFlags f1)                   { return ~(uint32_t) f1; }
RT_FUNCTION constexpr uint32_t operator +(BSDFFlags e)                    { return (uint32_t) e; }
