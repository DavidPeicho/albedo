#ifndef INTERSECTION_UTILS_H
#define INTERSECTION_UTILS_H

#define STACK_SIZE 32
// #define DEBUG_CWBVH_TRAVERSAL

struct Primitive
{
  Vertex v0;
  Vertex v1;
  Vertex v2;
};

Vertex
getVertex(uint vertexOffset, uint index)
{
  return vertices[vertexOffset + index];
}

/**
 * Retrieve the triangle from vertex offset and index
 *
 * @param instance Instance to extract the primitive from
 * @param intersection Intersection data
 */
Primitive extractPrimitive(Instance instance, Intersection intersection)
{
  Primitive p;
  p.v0 = getVertex(instance.vertexRootIndex, intersection.index);
  p.v1 = getVertex(instance.vertexRootIndex, intersection.index + 1);
  p.v2 = getVertex(instance.vertexRootIndex, intersection.index + 2);
  return p;
}

vec2 interpolateBarycentric(vec2 v0, vec2 v1, vec2 v2, vec3 barycentric)
{
  return (
    barycentric.x * v0 +
    barycentric.y * v1 +
    barycentric.z * v2
  );
}
vec3 interpolateBarycentric(vec3 v0, vec3 v1, vec3 v2, vec3 barycentric)
{
  return (
    barycentric.x * v0 +
    barycentric.y * v1 +
    barycentric.z * v2
  );
}

vec3 barycentricCoordinates(vec2 uv)
{
  return vec3(1.0 - uv.x - uv.y, uv);
}

Ray
transformRay(inout Ray ray, mat4 transform)
{
  Ray result;
  // @todo: radiance and throughput should go somewhere else.
  result.origin = transformPosition(ray.origin, transform);
  result.dir = transformDirection(ray.dir, transform);
  return result;
}

float
intersectPlane(Ray ray, vec3 normal, vec3 origin, vec3 edge01, vec3 edge02)
{
  float NdotD = dot(normal, ray.dir);
  if (NdotD < EPSILON) { return MAX_FLOAT; }

  float t = dot(normal, origin - ray.origin) / NdotD;
  if (t < EPSILON) { return MAX_FLOAT; }

  vec3 intersection = (ray.origin + ray.dir * t) - origin;

  // Check not before first edge.
  float interProj = dot(edge01, intersection);
  if (interProj < EPSILON || interProj > dot(edge01, edge01)) { return MAX_FLOAT; }

  interProj = dot(edge02, intersection);
  if (interProj < EPSILON || interProj > dot(edge02, edge02)) { return MAX_FLOAT; }

  return t;
}

uint
sign_extend_s8x4(uint i)
{
	return ((i >> 7) & 0x01010101) * 0xff;
}

uvec4
as_uchar4(float val)
{
	uint bits = floatBitsToUint(val);
	return uvec4(
		bitfieldExtract(bits, 0, 8),
		bitfieldExtract(bits, 8, 8),
		bitfieldExtract(bits, 16, 8),
		bitfieldExtract(bits, 24, 8)
	);
}

/**
 * Traverse CWBVH BLAS
 *
 * @note Taken from: https://github.com/jbikker/tinybvh/blob/main/traverse_cwbvh.cl
 *
 * Modifications:
 * - __bind -> findMSB
 * - popc -> bitCount
 * - (u)char -> bitfieldExtract
 */
vec4
#ifndef DEBUG_CWBVH_TRAVERSAL
traverse_cwbvh(Ray ray, uint bvhNodeStart, uint primitiveStart, float t)
#else
traverse_cwbvh(Ray ray, uint bvhNodeStart, uint primitiveStart, float t, inout uint stepCount)
#endif
{
	const vec4 O4 = vec4(ray.origin, 1.0);
	const vec4 D4 = vec4(ray.dir, 0.0);
	const vec4 rD4 = vec4(1.0) / D4;

	vec4 hit;
	hit.x = t; // not fetching this from ray data to avoid one memory operation.
	// prepare traversal
	uvec2 stack[STACK_SIZE];
	uint hitAddr;
	uint stackPtr = 0;
	vec2 uv;
	float tmax = t;
	const uint octinv4 = (7 - ((D4.x < 0 ? 4 : 0) | (D4.y < 0 ? 2 : 0) | (D4.z < 0 ? 1 : 0))) * 0x1010101;

	uvec2 ngroup = uvec2(0u, 1u << 31u);
	uvec2 tgroup = uvec2(0);

	do
	{
		#ifdef DEBUG_CWBVH_TRAVERSAL
		++stepCount;
		#endif

		if (ngroup.y > 0x00FFFFFF)
		{
			uint hits = ngroup.y;
			uint imask = ngroup.y;
			int child_bit_index = findMSB( hits );
			uint child_node_base_index = ngroup.x;
			ngroup.y &= ~(1 << child_bit_index);
			if (ngroup.y > 0x00FFFFFF)
			{
				stack[stackPtr++] = ngroup;
			}

			const uint slot_index = (child_bit_index - 24) ^ (octinv4 & 255);
			const uint relative_index = bitCount( imask & ~(0xFFFFFFFF << slot_index) );
			const uint child_node_index = bvhNodeStart + child_node_base_index + relative_index;

			BVHNode node = nodes[child_node_index];

			ngroup.x = floatBitsToUint(node.n1.x);
			tgroup = uvec2(floatBitsToUint(node.n1.y), 0u);
			uint hitmask = 0;

			uint n0_w = floatBitsToUint(node.n0.w);
			uvec3 e = uvec3(bitfieldExtract(n0_w, 0, 8), bitfieldExtract(n0_w, 8, 8), bitfieldExtract(n0_w, 16, 8));
			e = uvec3((e.x + 127) & 0xFF, (e.y + 127) & 0xFF, (e.z + 127) & 0xFF);

			vec3 e_f = vec3(uintBitsToFloat(e.x << 23u), uintBitsToFloat(e.y << 23u), uintBitsToFloat(e.z << 23u));
			vec4 idir4 = vec4(e_f * rD4.xyz, 1.0);
			const vec4 orig4 = (node.n0 - O4) * rD4;

			{	// first 4
				uint meta4 = floatBitsToUint( node.n1.z );
				uint is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
				uint inner_mask4 = sign_extend_s8x4( is_inner4 << 3 );
				uint bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
				uint child_bits4 = (meta4 >> 5) & 0x07070707;
				vec4 lox4 = vec4( as_uchar4( rD4.x < 0 ? node.n3.z : node.n2.x ) );
				vec4 hix4 = vec4( as_uchar4( rD4.x < 0 ? node.n2.x : node.n3.z ) );
				vec4 loy4 = vec4( as_uchar4( rD4.y < 0 ? node.n4.x : node.n2.z ) );
				vec4 hiy4 = vec4( as_uchar4( rD4.y < 0 ? node.n2.z : node.n4.x ) );
				vec4 loz4 = vec4( as_uchar4( rD4.z < 0 ? node.n4.z : node.n3.x ) );
				vec4 hiz4 = vec4( as_uchar4( rD4.z < 0 ? node.n3.x : node.n4.z ) );
				{
					vec4 tminx4 = lox4 * idir4.xxxx + orig4.xxxx, tmaxx4 = hix4 * idir4.xxxx + orig4.xxxx;
					vec4 tminy4 = loy4 * idir4.yyyy + orig4.yyyy, tmaxy4 = hiy4 * idir4.yyyy + orig4.yyyy;
					vec4 tminz4 = loz4 * idir4.zzzz + orig4.zzzz, tmaxz4 = hiz4 * idir4.zzzz + orig4.zzzz;
					float cmina = max( max( max( tminx4.x, tminy4.x ), tminz4.x ), 0 );
					float cmaxa = min( min( min( tmaxx4.x, tmaxy4.x ), tmaxz4.x ), tmax );
					float cminb = max( max( max( tminx4.y, tminy4.y ), tminz4.y ), 0 );
					float cmaxb = min( min( min( tmaxx4.y, tmaxy4.y ), tmaxz4.y ), tmax );
					if (cmina <= cmaxa) {
						hitmask = (child_bits4 & 255) << (bit_index4 & 31);
					}
					if (cminb <= cmaxb) {
						hitmask |= ((child_bits4 >> 8) & 255) << ((bit_index4 >> 8) & 31);
					}
					float cminc = max( max( max( tminx4.z, tminy4.z ), tminz4.z ), 0 );
					float cmaxc = min( min( min( tmaxx4.z, tmaxy4.z ), tmaxz4.z ), tmax );
					float cmind = max( max( max( tminx4.w, tminy4.w ), tminz4.w ), 0 );
					float cmaxd = min( min( min( tmaxx4.w, tmaxy4.w ), tmaxz4.w ), tmax );
					if (cminc <= cmaxc) {
						hitmask |= ((child_bits4 >> 16) & 255) << ((bit_index4 >> 16) & 31);
					}
					if (cmind <= cmaxd) {
						hitmask |= (child_bits4 >> 24) << (bit_index4 >> 24);
					}
				}
			}
			{	// second 4
				uint meta4 = floatBitsToUint( node.n1.w );
				uint is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
				uint inner_mask4 = sign_extend_s8x4( is_inner4 << 3 );
				uint bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
				uint child_bits4 = (meta4 >> 5) & 0x07070707;
				vec4 lox4 = vec4( as_uchar4( rD4.x < 0 ? node.n3.w : node.n2.y ) );
				vec4 hix4 = vec4( as_uchar4( rD4.x < 0 ? node.n2.y : node.n3.w ) );
				vec4 loy4 = vec4( as_uchar4( rD4.y < 0 ? node.n4.y : node.n2.w ) );
				vec4 hiy4 = vec4( as_uchar4( rD4.y < 0 ? node.n2.w : node.n4.y ) );
				vec4 loz4 = vec4( as_uchar4( rD4.z < 0 ? node.n4.w : node.n3.y ) );
				vec4 hiz4 = vec4( as_uchar4( rD4.z < 0 ? node.n3.y : node.n4.w ) );
				{
					const vec4 tminx4 = lox4 * idir4.xxxx + orig4.xxxx, tmaxx4 = hix4 * idir4.xxxx + orig4.xxxx;
					const vec4 tminy4 = loy4 * idir4.yyyy + orig4.yyyy, tmaxy4 = hiy4 * idir4.yyyy + orig4.yyyy;
					const vec4 tminz4 = loz4 * idir4.zzzz + orig4.zzzz, tmaxz4 = hiz4 * idir4.zzzz + orig4.zzzz;
					const float cmina = max( max( max( tminx4.x, tminy4.x ), tminz4.x ), 0 );
					const float cmaxa = min( min( min( tmaxx4.x, tmaxy4.x ), tmaxz4.x ), tmax );
					const float cminb = max( max( max( tminx4.y, tminy4.y ), tminz4.y ), 0 );
					const float cmaxb = min( min( min( tmaxx4.y, tmaxy4.y ), tmaxz4.y ), tmax );
					if (cmina <= cmaxa) {
						hitmask |= (child_bits4 & 255) << (bit_index4 & 31);
					}
					if (cminb <= cmaxb) {
						hitmask |= ((child_bits4 >> 8) & 255) << ((bit_index4 >> 8) & 31);
					}
					float cminc = max( max( max( tminx4.z, tminy4.z ), tminz4.z ), 0 );
					float cmaxc = min( min( min( tmaxx4.z, tmaxy4.z ), tmaxz4.z ), tmax );
					float cmind = max( max( max( tminx4.w, tminy4.w ), tminz4.w ), 0 );
					float cmaxd = min( min( min( tmaxx4.w, tmaxy4.w ), tmaxz4.w ), tmax );
					if (cminc <= cmaxc) {
						hitmask |= ((child_bits4 >> 16) & 255) << ((bit_index4 >> 16) & 31);
					}
					if (cmind <= cmaxd) {
						hitmask |= (child_bits4 >> 24) << (bit_index4 >> 24);
					}
				}
			}

			uint mask = bitfieldExtract(floatBitsToUint(node.n0.w), 24, 8);
			ngroup.y = (hitmask & 0xFF000000) | mask;
			tgroup.y = hitmask & 0x00FFFFFF;
		}
		else
		{
			tgroup = ngroup;
			ngroup = uvec2(0u);
		}

		while (tgroup.y != 0u)
		{
			// M�ller-Trumbore intersection; triangles are stored as 3x16 bytes,
			// with the original primitive index in the (otherwise unused) w
			// component of vertex 0.
			int triangleIndex = findMSB( tgroup.y );
			tgroup.y -= 1 << triangleIndex;

			uint triAddr = tgroup.x + (primitiveStart + triangleIndex) * 3;

			vec3 e1 = trianglesCWBVH[triAddr].xyz;
			vec3 e2 = trianglesCWBVH[triAddr + 1].xyz;
			vec4 v0 = trianglesCWBVH[triAddr + 2];
			vec3 r = cross( D4.xyz, e1 );
			float a = dot( e2, r );
			if (abs( a ) < EPSILON) continue;
			float f = 1.0 / a;
			vec3 s = O4.xyz - v0.xyz;
			float u = f * dot( s, r );
			if (u < EPSILON || u > EPSILON1) continue;
			vec3 q = cross( s, e2 );
			float v = f * dot( D4.xyz, q );
			if (v < EPSILON || u + v > EPSILON1) continue;
			float d = f * dot( e1, q );
			if (d <= EPSILON || d >= tmax) continue;
			uv = vec2(u, v);
			tmax = d;
			hitAddr = floatBitsToUint( v0.w );
		}

		if (ngroup.y <= 0x00FFFFFFu)
		{
			if (stackPtr > 0)
			{
				ngroup = stack[--stackPtr];
			}
			else
			{
				hit = vec4(tmax, uv.x, uv.y, uintBitsToFloat( hitAddr ));
				break;
			}
		}
	} while (true);

	return hit;
}

Intersection
sceneHit(Ray ray)
{
  #ifdef DEBUG_CWBVH_TRAVERSAL
  uint stepCount = 0;
  #endif

  Intersection intersection;
  intersection.dist = MAX_FLOAT;
  intersection.index = INVALID_UINT;
  intersection.instance = INVALID_UINT;
  intersection.emitter = INVALID_UINT;

  for (uint i = 0; i < instances.length(); ++i)
  {
    Instance instance = instances[i];

    // Performs intersection in model space.
    Ray rayModel = transformRay(ray, instance.worldToModel);
	#ifndef DEBUG_CWBVH_TRAVERSAL
	vec4 hit = traverse_cwbvh(rayModel, instance.bvhRootIndex, instance.primitiveRootIndex, intersection.dist);
	#else
	vec4 hit = traverse_cwbvh(rayModel, instance.bvhRootIndex, instance.primitiveRootIndex, intersection.dist, stepCount);
	#endif
	if (hit.x < intersection.dist)
    {
		intersection.dist = hit.x;
		intersection.uv = hit.yz;
		intersection.index = floatBitsToUint(hit.w) * 3;
		intersection.instance = i;
		intersection.emitter = INVALID_UINT;
		intersection.materialIndex = instance.materialIndex;
    }
  }
  return intersection;
}

#ifdef DEBUG_CWBVH_TRAVERSAL
uint
sceneTraversal(Ray ray)
{
	uint stepCount = 0;
	for (uint i = 0; i < instances.length(); ++i)
  	{
		Instance instance = instances[i];
		Ray rayModel = transformRay(ray, instance.worldToModel);
		traverse_cwbvh(rayModel, instance.bvhRootIndex, instance.primitiveRootIndex, MAX_FLOAT, stepCount);
	}
	return stepCount;
}
#endif

#endif // INTERSECTION_UTILS_H
