#version 460
#extension GL_EXT_ray_tracing : enable

layout(binding = 0, set = 0) uniform accelerationStructureEXT TLAS;
layout(binding = 1, set = 0, rgba8) uniform image2D image;

layout(binding = 2, set = 0) uniform Camera
{
    mat4 viewInverse;
    mat4 projInverse;
} cam;

// XXX debug
struct ray {
    highp vec3 P;
    mediump vec3 D;
};

ray ray_transfer(in ray inray, highp float t, vec3 normal)
{
    ray outray;

    outray.P = inray.P + inray.D * t;
    outray.D = inray.D;

    return outray;
}

ray ray_reflect(in ray inray, in vec3 normal)
{
    ray outray;
    outray.D = reflect(inray.D, normal);
    outray.P = inray.P + normal * .0001; // XXX surface fudge

    return outray;
}

void ray_transform(in ray r, in mat4 matrix, in mat4 normal_matrix, out ray t)
{
    t.P = (matrix * vec4(r.P, 1.0)).xyz;
    t.D = (normal_matrix * vec4(r.D, 0.0)).xyz;
}

struct surface_hit
{
    highp float t;
    highp float which;
    mediump vec3 uvw;
};

const highp float infinitely_far = 10000000.0;


surface_hit surface_hit_init()
{
    return surface_hit(infinitely_far, -1.0, vec3(1, 0, 0));
}

struct range
{
    highp float t0, t1;
};

range make_range(highp float t0, highp float t1)
{
    range r;
    r.t0 = t0;
    r.t1 = t1;
    return r;
}

range range_full()
{
    return make_range(-100000000.0, 100000000.0);
}

range range_intersect(in range r1, in range r2)
{
    highp float t0 = max(r1.t0, r2.t0);
    highp float t1 = min(r1.t1, r2.t1);
    return make_range(t0, t1);
}

bool range_is_empty(in range r)
{
    return r.t0 >= r.t1;
}

void triangle_intersect(in highp vec3 v0, in highp vec3 v1, in highp vec3 v2, highp float which, in ray theray, in range r, inout surface_hit hit)
{

    highp vec3 e0 = v1 - v0;
    highp vec3 e1 = v0 - v2;
    highp vec3 e2 = v2 - v1;

    highp vec3 M = cross(e1, theray.D);

    highp float det = dot(e0, M);

    const highp float epsilon = 0.0000001; // .000001 from M-T paper too large for bunny
    if(det > -epsilon && det < epsilon) {
        return;
    }

    float inv_det = 1.0 / det;

    // Do this in a somewhat different order than M-T in order to early out
    // if previous intersection is closer than this intersection
    highp vec3 T = theray.P - v0;
    highp vec3 Q = cross(T, e0);
    highp float d = -dot(e1, Q) * inv_det;
    if(d > hit.t) {
        return;
    }
    if(d < r.t0 || d > r.t1) {
        return;
    }

    mediump float u = dot(T, M) * inv_det;
    if(u < 0.0 || u > 1.0) {
        return;
    }

    mediump float v = dot(theray.D, Q) * inv_det;
    if(v < 0.0 || u + v > 1.0) {
        return;
    }

    hit.which = which;
    hit.t = d;
    hit.uvw[0] = 1.0 - u - v;
    hit.uvw[1] = u;
    hit.uvw[2] = v;
}


layout(location = 0) rayPayloadEXT vec3 hitValue;

// Adapted from Sascha's rgen
void main()
{
    const vec2 pixel_center = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 uv = pixel_center / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = uv * 2.0 - 1.0;

#if 1
    vec4 origin = cam.viewInverse * vec4(0,0,0,1);
    vec4 target = cam.projInverse * vec4(d.x, d.y, 1, 1) ;
    vec4 direction = cam.viewInverse * vec4(normalize(target.xyz), 0) ;
#elif 1
    vec4 origin = vec4(0, 0, 5, 1);
    vec4 direction = vec4(normalize(vec3(d.x, d.y, -1)), 0);
#elif 0
    float a = (uv.y * 2.0 - 1.0) * 3.14159 / 2;
    float b = uv.x * 3.14159 * 2;
    vec4 origin = vec4(0, 0, 100, 1);
    vec4 direction = vec4(normalize(vec3(cos(a) * cos(b), cos(a) * sin(b), sin(a))), 0);
    // origin = direction * -50;
#else
    vec4 origin = vec4(d * 10, 100, 1);
    vec4 direction = vec4(0, 0, -1, 0);
#endif

    float tmin = 0.001;
    float tmax = 10000.0;

    hitValue = vec3(1, 0, 0); // vec3(0.0);

    if(true) {
        traceRayEXT(TLAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(origin), tmin, vec3(direction), tmax, 0);
    } else {
        surface_hit hit = surface_hit_init();
        range r = range_full();
        ray theray;

        theray.P = vec3(origin);
        theray.D = vec3(direction);

        highp vec3 v0, v1, v2;
        v0 = vec3(-1, -1, 0);
        v1 = vec3(1, -1, 0);
        v2 = vec3(0, 1, 0);
        triangle_intersect(v0, v1, v2, 0, theray, r, hit);

        if(hit.t < infinitely_far) {
            hitValue = hit.uvw;
        } else {
            hitValue = vec3(0, 0, 1);
        }
    }

    // hitValue = vec3(direction.x, direction.y, direction.z);

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(hitValue, 0.0));
}
