#pragma once
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