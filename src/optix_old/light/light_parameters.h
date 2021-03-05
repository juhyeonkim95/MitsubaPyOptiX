
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

using namespace optix;

#define LIGHT_QUAD 0
#define LIGHT_SPHERE 1
#define LIGHT_POINT 2
#define LIGHT_DIRECTIONAL 3
#define LIGHT_SPOT 4
#define LIGHT_DISK 5
#define LIGHT_TRIANGLE_MESH 6

struct LightParameter
{
	optix::float3 position;
	optix::float3 direction;
	optix::float3 normal;
	// for area light -> radiance (W/sr*m^2)
	// for point light -> radiant intensity (W/sr)
	optix::float3 emission;
    optix::float3 intensity;

	optix::float3 u;
	optix::float3 v;
	float radius;
	float area;
	float cosTotalWidth;
	float cosFalloffStart;
	unsigned int lightType;
	rtBufferId<float3, 1> pos_buffer_id;
	rtBufferId<int3, 1> indice_buffer_id;
	rtBufferId<float3, 1> normal_buffer_id;
	int n_triangles;
	Matrix4x4 transformation;
	int envmapID;
	int isTwosided;
};

struct LightSample
{
	optix::float3 wi;
	float pdf;
	optix::float3 Li;
	float lightDist;
};

struct AreaSample
{
    optix::float3 position;
    optix::float3 normal;
    float pdf;
};