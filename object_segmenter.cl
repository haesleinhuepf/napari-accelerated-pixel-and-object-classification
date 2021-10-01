/*
OpenCL RandomForestClassifier
feature_specification = original gaussian_blur=1 sobel_of_gaussian_blur=1
num_classes = 2
num_features = 3
max_depth = 2
num_trees = 10
positive_class_identifier = 2
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_out_TYPE out) {
 sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 const int x = get_global_id(0);
 const int y = get_global_id(1);
 const int z = get_global_id(2);
 float i0 = READ_IMAGE(in0, sampler, POS_in0_INSTANCE(x,y,z,0)).x;
 float i1 = READ_IMAGE(in1, sampler, POS_in1_INSTANCE(x,y,z,0)).x;
 float i2 = READ_IMAGE(in2, sampler, POS_in2_INSTANCE(x,y,z,0)).x;
 float s0=0;
 float s1=0;
if(i2<36.880149841308594){
 if(i0<96.0){
  s0+=138.0;
 } else {
  s1+=44.0;
 }
} else {
 if(i0<116.0){
  s0+=55.0;
 } else {
  s1+=108.0;
 }
}
if(i0<108.0){
 s0+=187.0;
} else {
 s1+=158.0;
}
if(i0<108.0){
 s0+=221.0;
} else {
 s1+=124.0;
}
if(i1<113.27566528320312){
 s0+=215.0;
} else {
 s1+=130.0;
}
if(i0<100.0){
 s0+=209.0;
} else {
 s1+=136.0;
}
if(i1<112.94844055175781){
 s0+=223.0;
} else {
 s1+=122.0;
}
if(i2<34.11639404296875){
 if(i1<99.91258239746094){
  s0+=127.0;
 } else {
  s1+=27.0;
 }
} else {
 if(i1<114.74969482421875){
  s0+=85.0;
 } else {
  s1+=106.0;
 }
}
if(i1<112.94844055175781){
 s0+=200.0;
} else {
 s1+=145.0;
}
if(i1<112.94844055175781){
 s0+=196.0;
} else {
 s1+=149.0;
}
if(i0<108.0){
 s0+=194.0;
} else {
 s1+=151.0;
}
 float max_s=s0;
 int cls=1;
 if (max_s < s1) {
  max_s = s1;
  cls=2;
 }
 WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);
}
