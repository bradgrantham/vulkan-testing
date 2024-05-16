#    k           
  SPV_KHR_ray_tracing      GLSL.std.450               �     main       i        testrt.rchit.glsl    �   �     #version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

// layout (binding = 2) uniform sampler2D color_texture;

layout(location = 0) rayPayloadInEXT vec3 hitValue;
hitAttributeEXT vec3 attribs;

void main()
{
    vec3 specular_color = vec3(1, 1, 1); // from material
    float shininess = 10; // from material
    vec3 light_position = vec3(1000, 1000, 1000); // get global light position
    vec3 light_color = vec3(1, 1, 1); // get global light color
    vec3 vertex_normal = vec3(0, 0, 1); // interpolate normal from per-vertex normals;
    vec3 vertex_color = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y); // interpolate color from per-color normals; 
    vec3 normal = normalize(vertex_normal);
    vec3 vertex_position = vec3(0, 0, 0); // interpolate position from vertex data; 
    vec2 vertex_texcoord = vec2(0, 0); // interpolate texcoord from per-vertex texcoords
    vec3 eye_position = -vertex_position.xyz; 
    vec3 edir = normalize(eye_position);

    if(dot(normal, edir) < 0) {
        normal *= -1;
    }

    vec3 ldir = normalize(light_position - vertex_position.xyz);
    vec3 refl = reflect(-ldir, normal);
    // vec3 diffuse = max(0, dot(normal, ldir)) * light_color * texture(color_texture, vertex_texcoord).xyz;
    vec3 diffuse = max(0, dot(normal, ldir)) * light_color;
    vec3 ambient = vec3(1, 1, 1); // ubo2.light_color;
    vec3 specular = pow(max(0, dot(refl, edir)), shininess) * specular_color * light_color;
    vec3 material_diffuse = vec3(.8, .8, .8);
    vec3 material_ambient = vec3(.1, .1, .1);
    vec3 material_specular = vec3(.8, .8, .8);

    // vec3 result_color = vertex_color.xyz * (diffuse * material_diffuse + ambient * material_ambient + specular * material_specular );

    vec3 result_color = vertex_color;

    hitValue = result_color;
}

     GL_EXT_nonuniform_qualifier   GL_EXT_ray_tracing       main      
   specular_color       shininess        light_position       light_color      vertex_normal        vertex_color         attribs   )   normal    ,   vertex_position   0   vertex_texcoord   2   eye_position      5   edir      B   ldir      G   refl      L   diffuse   S   ambient   T   specular      _   material_diffuse      b   material_ambient      e   material_specular     f   result_color      i   hitValue    J client vulkan100    J target-env spirv1.6 J target-env vulkan1.3    J entry-point main         !                               	         +          �?,                             +           A+          zD,                 +            ,                       �     ;        �               +                  �     +            ,     -              .            /      .   ,  .   1           ;   +     ?     ��+     `   ��L?,     a   `   `   `   +     c   ���=,     d   c   c   c      h   �     ;  h   i   �       
      6               �     ;  	   
      ;           ;  	         ;  	         ;  	         ;  	         ;  	   )      ;  	   ,      ;  /   0      ;  	   2      ;  	   5      ;  	   B      ;  	   G      ;  	   L      ;  	   S      ;  	   T      ;  	   _      ;  	   b      ;  	   e      ;  	   f                  >  
                  >                    >                    >                    >                    A              =           �              A     !          =     "   !   �     #      "   A     $         =     %   $   A     &          =     '   &   P     (   #   %   '   >     (               =     *           +      E   *   >  )   +               >  ,   -               >  0   1               =     3   ,        4   3   >  2   4               =     6   2        7      E   6   >  5   7               =     8   )   =     9   5   �     :   8   9   �  ;   <   :      �  >       �  <   =   >   �  =               =     @   )   �     A   @   ?   >  )   A   �  >   �  >               =     C      =     D   ,   �     E   C   D        F      E   E   >  B   F               =     H   B        I   H   =     J   )        K      G   I   J   >  G   K               =     M   )   =     N   B   �     O   M   N        P      (      O   =     Q      �     R   Q   P   >  L   R                >  S           !       =     U   G   =     V   5   �     W   U   V        X      (      W   =     Y           Z         X   Y   =     [   
   �     \   [   Z   =     ]      �     ^   \   ]   >  T   ^        "       >  _   a        #       >  b   d        $       >  e   a        (       =     g      >  f   g        *       =     j   f   >  i   j   �  8  