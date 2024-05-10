#version 460
#extension GL_EXT_ray_tracing : enable

layout(binding = 0 /*, set = 0 */ ) uniform accelerationStructureEXT TLAS;
layout(binding = 1 /*, set = 0 */ , rgba8) uniform image2D image;

// XXX ??? 
// CameraProperties From Sascha's rgen
layout(binding = 2, set = 0) uniform CameraProperties
{
    mat4 viewInverse;
    mat4 projInverse;
} cam;


layout(location = 0) rayPayloadEXT vec3 hitValue;

    // From Sascha's rgen
void main()
{
    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = inUV * 2.0 - 1.0;

    vec4 origin = cam.viewInverse * vec4(0,0,0,1);
    vec4 target = cam.projInverse * vec4(d.x, d.y, 1, 1) ;
    vec4 direction = cam.viewInverse * vec4(normalize(target.xyz), 0) ;

    float tmin = 0.001;
    float tmax = 10000.0;

    hitValue = vec3(0.0);

    traceRayEXT(TLAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin.xyz, tmin, direction.xyz, tmax, 0);

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(hitValue, 0.0));
}
