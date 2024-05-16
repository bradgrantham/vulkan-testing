#version 460
#extension GL_EXT_ray_tracing : enable

struct surface_hit
{
    highp float t;
    vec3 position;
    vec3 color;
    vec3 normal;
};

layout(binding = 0, set = 0) uniform accelerationStructureEXT TLAS;
layout(binding = 1, set = 0, rgba8) uniform image2D image;

layout(binding = 2, set = 0) uniform Camera
{
    mat4 viewInverse;
    mat4 projInverse;
} cam;

layout(location = 0) rayPayloadEXT surface_hit hit_value;

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
#elif 0
    vec4 origin = vec4(0, 0, 5, 1);
    vec4 direction = vec4(normalize(vec3(d.x, d.y, -1)), 0);
#elif 0
    float a = (uv.y * 2.0 - 1.0) * 3.14159 / 2;
    float b = uv.x * 3.14159 * 2;
    vec4 origin = cam.viewInverse * vec4(0, 0, 5, 1);
    vec4 direction = cam.viewInverse * vec4(normalize(vec3(cos(a) * sin(b), sin(a), cos(a) * cos(b))), 0);
#else
    vec4 origin = vec4(d * 10, 100, 1);
    vec4 direction = vec4(0, 0, -1, 0);
#endif

    float tmin = 0.001;
    float tmax = 10000.0;

    vec3 accumulated = vec3(1, 1, 1);

    hit_value.t = tmax;
    hit_value.color = vec3(1, 0, 0);
    hit_value.normal = vec3(0, 0, 1);

    traceRayEXT(TLAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(origin), tmin, vec3(direction), tmax, 0);
    if(hit_value.t == tmax) {
        accumulated = hit_value.color;
    } else {
        vec3 reflected = reflect(direction.xyz, hit_value.normal);
        vec3 color = hit_value.color;
        traceRayEXT(TLAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, vec3(hit_value.position), tmin, vec3(reflected), tmax, 0);
        accumulated = color + hit_value.color;
    }

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(accumulated, 0.0));
}
