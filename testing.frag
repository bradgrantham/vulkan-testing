#    }                 GLSL.std.450                     main          '   B   F   U   h   i                testing.frag.glsl    ’   Β     #version 450

layout (binding = 1) uniform UBO2
{
    vec3 light_position;
    vec3 light_color;
} ubo2;

layout (binding = 2) uniform Shading
{
    vec3 specular_color;
    float shininess;
} shading;

layout (binding = 3) uniform sampler2D color_texture;

layout (location = 0) in vec4 vertex_position;
layout (location = 1) in vec3 vertex_normal;
layout (location = 2) in vec4 vertex_color;
layout (location = 3) in vec2 vertex_texcoord;

layout (location = 0) out vec4 outFragColor;

void main()
{
    vec3 normal = normalize(vertex_normal);
    vec3 eye_position = -vertex_position.xyz; 
    vec3 edir = normalize(eye_position);
    if(dot(normal, edir) < 0) {
        normal *= -1;
    }
    vec3 ldir = normalize(ubo2.light_position - vertex_position.xyz);
    vec3 refl = reflect(-ldir, normal);
    vec3 diffuse = max(0, dot(normal, ldir)) * ubo2.light_color * texture(color_texture, vertex_texcoord).xyz;
    vec3 ambient = ubo2.light_color;
    // vec4 specular = pow(max(0, dot(refl, edir)), material_shininess) * light_color * .8; // from spin
    vec3 specular = pow(max(0, dot(refl, edir)), shading.shininess) * shading.specular_color * ubo2.light_color;
    vec3 material_diffuse = vec3(.8, .8, .8);
    vec3 material_ambient = vec3(.1, .1, .1);
    vec3 material_specular = vec3(.8, .8, .8);
    // specular = vec4(0.0, 0.0, 0.0, 0.0); // from spin
    // color = diffuse * material_diffuse * vertex_color + ambient * material_ambient * vertex_color + specular * material_specular; // from spin
    outFragColor = vec4(vertex_color.xyz * (diffuse * material_diffuse + ambient * material_ambient + specular * material_specular ), 1.0);
}
       main      
   normal       vertex_normal        eye_position         vertex_position      edir      $   ldir      %   UBO2      %       light_position    %      light_color   '   ubo2      1   refl      6   diffuse   B   color_texture     F   vertex_texcoord   K   ambient   N   specular      S   Shading   S       specular_color    S      shininess     U   shading   `   material_diffuse      c   material_ambient      f   material_specular     h   outFragColor      i   vertex_color    J client vulkan100    J target-env spirv1.6 J target-env vulkan1.3    J entry-point main    G           G            H  %       #       H  %      #      G  %      G  '   "       G  '   !      G  B   "       G  B   !      G  F         H  S       #       H  S      #      G  S      G  U   "       G  U   !      G  h          G  i              !                               	                     ;                                  ;           +                 +     !     Ώ  %            &      %   ;  &   '        (          +  (   )          *         +  (   ;       	 ?                              @   ?      A       @   ;  A   B         D            E      D   ;  E   F        S            T      S   ;  T   U         V         +     a   ΝΜL?,     b   a   a   a   +     d   ΝΜΜ=,     e   d   d   d      g         ;  g   h      ;     i      +     x     ?           6               ψ     ;  	   
      ;  	         ;  	         ;  	   $      ;  	   1      ;  	   6      ;  	   K      ;  	   N      ;  	   `      ;  	   c      ;  	   f                  =                      E      >  
                  =           O                                   >                    =                      E      >                    =        
   =                         Έ              χ          ϊ            ψ                 =     "   
        #   "   !   >  
   #   ω      ψ                   A  *   +   '   )   =     ,   +   =     -      O     .   -   -                  /   ,   .        0      E   /   >  $   0        !       =     2   $        3   2   =     4   
        5      G   3   4   >  1   5        "       =     7   
   =     8   $        9   7   8        :      (      9   A  *   <   '   ;   =     =   <        >   =   :   =  @   C   B   =  D   G   F   W     H   C   G   O     I   H   H                  J   >   I   >  6   J        #       A  *   L   '   ;   =     M   L   >  K   M        %       =     O   1   =     P           Q   O   P        R      (      Q   A  V   W   U   ;   =     X   W        Y         R   X   A  *   Z   U   )   =     [   Z        \   [   Y   A  *   ]   '   ;   =     ^   ]        _   \   ^   >  N   _        &       >  `   b        '       >  c   e        (       >  f   b        +       =     j   i   O     k   j   j             =     l   6   =     m   `        n   l   m   =     o   K   =     p   c        q   o   p        r   n   q   =     s   N   =     t   f        u   s   t        v   r   u        w   k   v   Q     y   w       Q     z   w      Q     {   w      P     |   y   z   {   x   >  h   |   ύ  8  