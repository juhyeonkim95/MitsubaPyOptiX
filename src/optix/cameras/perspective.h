#include "optix/common/rt_function.h"
#include "optix/common/random.h"

using namespace optix;
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float,        focalDistance, , );
rtDeclareVariable(float,        apertureRadius, , );

RT_FUNCTION void generate_ray_perspective(const float2 &pixel, float3 &origin, float3 &direction, unsigned int &seed)
{
    float3 rayOrigin = eye;
    float3 rayDirection = normalize(pixel.x*U + pixel.y*V + W);

    if(focalDistance > 0.0f && apertureRadius > 0.0f){

        const float3 focalPoint = eye + focalDistance * rayDirection;

        const float r1 = rnd(seed);
	    const float r2 = rnd(seed);
	    const float rho = sqrtf(r1);
        const float phi = r2 * 2.0f * M_PIf;
        float sinPhi, cosPhi;
		sincos(phi, &sinPhi, &cosPhi);

		const float dx = apertureRadius * rho * cosPhi;
		const float dy = apertureRadius * rho * sinPhi;

		rayOrigin = eye + dx * U + dy * V;
		rayDirection = normalize(focalPoint - rayOrigin);
    }

    origin = rayOrigin;
    direction = rayDirection;
}