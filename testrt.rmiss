#               
  SPV_KHR_ray_tracing      GLSL.std.450               �     main            testrt.rmiss.glsl    6    �     #version 460
#extension GL_EXT_ray_tracing : enable

layout(location = 0) rayPayloadInEXT vec3 hitValue;

void main()
{
    vec3 result_color = vec3(.1, .1, .2);

    hitValue = result_color;
}


      GL_EXT_ray_tracing       main      
   result_color         hitValue    J client vulkan100    J target-env spirv1.6 J target-env vulkan1.3    J entry-point main         !                               	         +        ���=+        ��L>,                       �     ;        �             6               �     ;  	   
                  >  
           
       =        
   >        �  8  