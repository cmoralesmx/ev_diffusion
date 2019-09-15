#version 330 core
// modified from https://stackoverflow.com/a/10506172/
layout (location=0) in vec2 position;
layout (location=1) in float radius;

out vData {
	float radius;
	vec3 sphereColor;
	vec4 transformedPosition;
	vec3 normalizedViewCoordinate;
} vertex;

uniform mat4 mvp;

void main(){
	vertex.radius = radius * 2.0f;
	/*
	if(radius < 0.06)
		vertex.sphereColor = vec3(1., 0., 0.);
	else if (radius < 0.08)
		vertex.sphereColor = vec3(0., 1., 0.);
	else if (radius < 0.1)
		vertex.sphereColor = vec3(0., 0., 1.);
	else if (radius < 0.12)
		vertex.sphereColor = vec3(0., 1., 1.);
	else if (radius < 0.14)
		vertex.sphereColor = vec3(1., 1., 0.);
	else
		vertex.sphereColor = vec3(1., 0., 1.);
	*/
	vec4 transformedPosition = mvp * vec4(position, 0., 1.);
	vertex.sphereColor = vec3(.3, .3, .3);
	vertex.normalizedViewCoordinate = (transformedPosition.xyz + 1.0) / 2.0;
	vertex.transformedPosition = transformedPosition;

	gl_Position = transformedPosition;
}

