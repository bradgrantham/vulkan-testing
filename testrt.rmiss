#    6           
  SPV_KHR_ray_tracing      GLSL.std.450               �     main       $   0        testrt.rmiss.glsl    �    �     #version 460
#extension GL_EXT_ray_tracing : enable

struct surface_hit
{
    highp float t;
    vec3 position;
    vec3 color;
    vec3 normal;
};

layout(location = 0) rayPayloadInEXT surface_hit hitValue;

void main()
{
    float shininess = 50;
    vec3 eye = gl_WorldRayOriginEXT;
    vec3 edir = normalize(eye);

    vec3 light_position = vec3(1000, 1000, 1000); // get global light position
    vec3 light_color = vec3(1, 1, 1); // get global light color

    vec3 light_direction = normalize(light_position - eye);

    vec3 result_color = vec3(.2, .2, .8) + pow(max(0, dot(gl_WorldRayDirectionEXT, light_direction)), shininess) * light_color;

    hitValue.color = result_color;
}


      GL_EXT_ray_tracing       main      	   shininess        eye      gl_WorldRayOriginEXT         edir         light_position       light_color      light_direction      result_color      $   gl_WorldRayDirectionEXT   .   surface_hit   .       t     .      position      .      color     .      normal    0   hitValue    J client vulkan100    J target-env spirv1.6 J target-env vulkan1.3    J entry-point main    G        �  G  $      �       !                             +     
     HB                                   ;           +          zD,                 +          �?,                 +         ��L>+     !   ��L?,     "           !   +     #       ;     $        .                  /   �  .   ;  /   0   �    1          +  1   2         4   �                6               �     ;     	      ;           ;           ;           ;           ;           ;                       >  	   
               =           >                    =                      E      >                    >                    >                    =           =           �                         E      >                    =     %   $   =     &      �     '   %   &        (      (   #   '   =     )   	        *         (   )   =     +      �     ,   +   *   �     -   "   ,   >     -               =     3      A  4   5   0   2   >  5   3   �  8  