/*
* FLAME GPU v 1.5.X for CUDA 9
* Copyright University of Sheffield.
* Original Author: Dr Paul Richmond (user contributions tracked on https://github.com/FLAMEGPU/FLAMEGPU)
* Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
*
* University of Sheffield retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* University of Sheffield is strictly prohibited.
*
* For terms of licence agreement please attached licence or view licence
* on www.flamegpu.com website.
*
*/

// includes, project
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>

#include <GL/glew.h>
//#include <GL/glut.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include "header.h"
#include "visualisation.h"
#define __MYVISUAL
#include "myVisualisation.h"

extern float *h_xs;
extern float *h_ys;
#define FOVY 45.0

GLuint vboIdCiliary;
GLuint vboIdSecretory;
GLuint vaoIdSecretory;
GLuint vaoIdCiliary;
cudaGraphicsResource *CGRsecretory, *CGRciliary;
bool display_secretory = true;

//Simulation output buffers/textures
cudaGraphicsResource_t EV_default_cgr;
// vertex Shader
GLuint vertexShader;
GLuint fragmentShader;
GLuint shaderProgram;
GLuint vs_mapIndex;

GLuint secretory_vertexShader;
GLuint secretory_shaderProgram;
GLuint boundaries_fragmentShader;
GLuint ciliary_vertexShader;
GLuint ciliary_shaderProgram;

// bo variables
GLuint sphereVerts;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -VIEW_DISTANCE;  // This parameter comes from the visualisation.h file

// camera movement
const float DEFAULT_OFFSET_X = 0.0f, DEFAULT_OFFSET_Y = 0.0f, DEFAULT_ZOOM_LEVEL = -VIEW_DISTANCE;
float offset_x = DEFAULT_OFFSET_X, offset_y = DEFAULT_OFFSET_Y, zoom_level = DEFAULT_ZOOM_LEVEL;
glm::mat4 model, view, projection;

// keyboard controls
#if defined(PAUSE_ON_START)
bool paused = true;
#else
bool paused = false;
#endif

//timer
cudaEvent_t start, stop;
const int display_rate = 50;
int frame_count;
float frame_time = 0.0;
unsigned long iterations = 0;

#ifdef SIMULATION_DELAY
//delay
int delay_count = 0;
#endif

// prototypes
int initGL();
void initShader();
void createVBO(GLuint* vbo, GLuint size);
void deleteVBO(GLuint* vbo);
void reshape(int width, int height);
void display();
void close();
void keyboard(unsigned char key, int x, int y);
void special(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void runCuda();
void checkGLError(char * caller);

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort = true)
{
	gpuAssert(cudaPeekAtLastError(), file, line);
#ifdef _DEBUG
	gpuAssert(cudaDeviceSynchronize(), file, line);
#endif

}

/**
This is an OpenGL 3

uses an uniform of type samplerBuffer, which is defined by the GL_EXT_gpu_shader4 extension
Inputs:
float mapIndex - specifies the target element to read from the buffer
Ouputs:
vec3 normal
vec3 lightDir
vec4 colour

This shader creates a sampler type (samplerBuffer) for buffer textures
and adds a lookup function (texelFetchBuffer) to explicitly access the
buffer texture ussing texture hardware

store in 'position' the value of gl_Vertex
store in 'lookup' the value at 'mapIndex' stored in 'displacementMap'

Then, the value in lookup.w determines the colour to use for the diffuse colour,
which is the colour reflected from the object

The first 3 elements (lookup.xyz) affect the positional data obtained from gl_Vertex
*/
const char vertexShaderSource[] =
{
	"uniform mat4 mvp;															\n"
	"varying vec4 colour;														\n"
	"attribute in vec2 position;												\n"
	"attribute in float radius;													\n"
	"void main()																\n"
	"{																			\n"
	"	if (radius > 0.0664)	                								\n"
	"		colour = vec4(0., 0.59, 0.53, 1.0);						    		\n" // teal (0,150,133)
	"	else if (radius > 0.0579)	               								\n"
	"		colour = vec4(0.9, 0.12, 0.39, 1.0);							    \n" // pink (229,30,99)
	"	else if (radius > 0.0493)	                							\n"
	"		colour = vec4(0.61, 0.15, 0.69, 1.0);							    \n" // purple (155,38,175)
	"	else if (radius > 0.0407)	                							\n"
	"		colour = vec4(0.1, 0.66, 0.96, 1.0);							    \n"	// light blue (25,168,244)
	"	else if (radius > 0.0321)	                							\n"
	"		colour = vec4(1.0, 0.92, 0.23, 1.0);								\n" // yellow (255,234,58)
	"	else if (radius > 0.0236)	                							\n"
	"		colour = vec4(0.5, 0.2, 0.0, 1.0);									\n" // brown (128, 51, 0)
	"	else                      	                							\n"
	"		colour = vec4(0.37, 0.49, 0.55, 1.0);								\n" // blue grey (91, 124, 140)
	"																    		\n"
	"   gl_Position = mvp * vec4(position, 0, 1);		    					\n"
	"	gl_PointSize = radius * 20;												\n"
	"}																			\n"
};

/*
A nice explanation of how shadding works is available at
https://learnopengl.com/Lighting/Basic-Lighting
http://www.opengl-tutorial.org/beginners-tutorials/tutorial-8-basic-shading/
*/
const char fragmentShaderSource[] =
{
	"varying vec4 colour;														\n"
	"void main (void)															\n"
	"{																			\n"
	"	// Defining The Material Colors											\n"
	"	vec4 AmbientColor = colour;												\n"
	"	vec4 DiffuseColor = vec4(0.0, 0.0, 0.25, 1.0);	           		    	\n"
	"	gl_FragColor = AmbientColor + DiffuseColor;								\n"
	"}																			\n"
};

const char secretory_vertexShaderSource[] =
{
	"#version 330 core															\n"
	"layout (location=0) in vec2 position;									\n"
	"out vec4 colour;														\n"
	"uniform mat4 mvp;															\n"
	"void main()																\n"
	"{																			\n"
	"   gl_Position = mvp * vec4(position, 0.0, 1.0);							\n"
	"	colour = vec4(1., 0., 0., 1.);											\n"
	"}																			\n"
};

const char ciliary_vertexShaderSource[] =
{
	"#version 330 core														\n"
	"layout (location=0) in vec2 position;									\n"
	"out vec4 colour;														\n"
	"uniform mat4 mvp;														\n"
	"void main()																\n"
	"{																			\n"
	"   gl_Position = mvp * vec4(position, 0.0, 1.0);							\n"
	"	colour = vec4(0., 0., 1., 1.);											\n"
	"}																			\n"
};

const char boundaries_fragmentShaderSource[] =
{
	"#version 330 core																\n"
	"in vec4 colour;														\n"
	"void main (void)															\n"
	"{																			\n"
	"	gl_FragColor = colour;													\n"
	"}																			\n"
};

//GPU Kernels
__global__ void output_EV_agent_to_VBO(xmachine_memory_EV_list* agents, glm::vec3* vbo){

	//global thread index
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	vbo[index].x = agents->x[index];
	vbo[index].y = agents->y[index];
	vbo[index].z = agents->radius_um[index];
}

__device__ int copyVertexToVBO(glm::vec2* vbo, int index, float p1_x, float p1_y, float p2_x, float p2_y, float type) {
	int index2 = index * 2;

	vbo[index2].x = p1_x;
	vbo[index2].y = p1_y;
	vbo[index2 + 1].x = p2_x;
	vbo[index2 + 1].y = p2_y;
	return 0;
}

__global__ void output_SecretoryCell_agent_to_VBO(xmachine_memory_SecretoryCell_list* agents, glm::vec2* vbo) {

	//global thread index
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	copyVertexToVBO(vbo, index, agents->p1_x[index], agents->p1_y[index], agents->p2_x[index], agents->p2_y[index], 1.0);
}

__global__ void output_CiliaryCell_agent_to_VBO(xmachine_memory_CiliaryCell_list* agents, glm::vec2* vbo) {

	//global thread index
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	copyVertexToVBO(vbo, index, agents->p1_x[index], agents->p1_y[index],  agents->p2_x[index], agents->p2_y[index], 0.0);
}

void initVisualisation()
{
	printf("Initialising visualisation\n");
	// Create GL context
	int argc = 1;
	char glutString[] = "GLUT application";
	char *argv[] = { glutString, NULL };
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("FLAME GPU Visualiser");

	// initialize GL
	if (!initGL()) {
		return;
	}
	initShader();

	// register callbacks
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutCloseFunc(close);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	// create VBO's
	createVBO(&sphereVerts, get_agent_EV_MAX_count() * sizeof(glm::vec3));
	
	// registers a GraphicsGLResource with CUDA for interop access
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&EV_default_cgr, sphereVerts, cudaGraphicsMapFlagsNone));

	//create a events for timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	printf("Visualisation initialised\n");
}

void runVisualisation(){
	printf(">>>>> Allocating OpenGL resources for: [Secretory cells: %d, Ciliary cells: %d]\n", get_agent_SecretoryCell_s_default_count(), get_agent_CiliaryCell_c_default_count());

	// prepare the containers
	unsigned int size;
	int threads_per_tile = 256;
	int tile_size;
	dim3 grid;
	dim3 threads;

	if (get_agent_SecretoryCell_s_default_count() > 0) {
		size = get_agent_SecretoryCell_MAX_count() * 4 * sizeof(float);
		createVBO(&vboIdSecretory, size);
		gpuErrchk(cudaGraphicsGLRegisterBuffer(&CGRsecretory, vboIdSecretory, cudaGraphicsMapFlagsReadOnly));
		checkGLError("createVBO - secretory");

		glm::vec2 *secretory_dptr;
		size_t size_secretory;
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, &CGRsecretory, 0));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&secretory_dptr, &size_secretory, CGRsecretory));

		//cuda block size
		tile_size = (int)ceil((float)get_agent_SecretoryCell_s_default_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

		output_SecretoryCell_agent_to_VBO << < grid, threads >> > (get_device_SecretoryCell_s_default_agents(), secretory_dptr);
		cudaDeviceSynchronize();
		gpuErrchkLaunch();

		gpuErrchk(cudaGraphicsUnmapResources(1, &CGRsecretory, 0));
	}

	if (get_agent_CiliaryCell_c_default_count() > 0) {
		size = get_agent_CiliaryCell_MAX_count() * 4 * sizeof(float);
		createVBO(&vboIdCiliary, size);
		gpuErrchk(cudaGraphicsGLRegisterBuffer(&CGRciliary, vboIdCiliary, cudaGraphicsMapFlagsReadOnly));
		checkGLError("createVBO - ciliary");

		glm::vec2 *ciliary_dptr;
		size_t size_ciliary;
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, &CGRciliary, 0));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&ciliary_dptr, &size_ciliary, CGRciliary));

		//cuda block size
		tile_size = (int)ceil((float)get_agent_CiliaryCell_c_default_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

		output_CiliaryCell_agent_to_VBO << < grid, threads >> > (get_device_CiliaryCell_c_default_agents(), ciliary_dptr);
		cudaDeviceSynchronize();
		gpuErrchkLaunch();

		gpuErrchk(cudaGraphicsUnmapResources(1, &CGRciliary, 0));
	}

	// start rendering mainloop
	glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
	if (!paused){
#ifdef SIMULATION_DELAY
		delay_count++;
		if (delay_count == SIMULATION_DELAY){
			delay_count = 0;
			singleIteration();
		}
#else
		singleIteration();
		++iterations;
#endif
	}

	//kernals sizes
	int threads_per_tile = 256;
	int tile_size;
	dim3 grid;
	dim3 threads;

	glm::vec3 *dptr;

	if (get_agent_EV_default_count() > 0)
	{
		size_t accessibleBufferSize = 0;
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, &EV_default_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &accessibleBufferSize, EV_default_cgr));
		//cuda block size
		tile_size = (int)ceil((float)get_agent_EV_default_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);
		
		output_EV_agent_to_VBO << < grid, threads >> >(get_device_EV_default_agents(), dptr);
		gpuErrchkLaunch();
		// unmap buffer object
		gpuErrchk(cudaGraphicsUnmapResources(1, &EV_default_cgr));
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL - method called by initVisualisation
////////////////////////////////////////////////////////////////////////////////
int initGL()
{
	// initialize necessary OpenGL extensions
	glewInit();
	if (!glewIsSupported("GL_VERSION_3_3")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
		fflush(stderr);
		return 1;
	}

	// default initialization
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);

	reshape(WINDOW_WIDTH, WINDOW_HEIGHT);
	checkGLError("initGL");

	//lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	return 1;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GLSL Vertex Shader
////////////////////////////////////////////////////////////////////////////////
void initShader()
{
	const char* v = vertexShaderSource;
	const char* f = fragmentShaderSource;

	//vertex shader
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &v, 0);
	glCompileShader(vertexShader);

	//fragment shader
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &f, 0);
	glCompileShader(fragmentShader);

	//program
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// check for errors
	GLint status;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error: vertexShader\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(vertexShader, 262144, &len, data);
		printf("%s", data);
	}
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Compilation Error: fragmentShader \n");
		char data[262144];
		int len;
		glGetShaderInfoLog(fragmentShader, 262144, &len, data);
		printf("%s", data);
	}
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Program Link Error: shaderProgram\n");
	}

	// get shader variables
	//vs_displacementMap = glGetUniformLocation(shaderProgram, "displacementMap");
	vs_mapIndex = glGetAttribLocation(shaderProgram, "mapIndex");

	const char* v2 = secretory_vertexShaderSource;
	const char* f2 = boundaries_fragmentShaderSource;

	//vertex shader
	secretory_vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(secretory_vertexShader, 1, &v2, 0);
	glCompileShader(secretory_vertexShader);

	//fragment shader
	boundaries_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(boundaries_fragmentShader, 1, &f2, 0);
	glCompileShader(boundaries_fragmentShader);

	//program
	secretory_shaderProgram = glCreateProgram();
	glAttachShader(secretory_shaderProgram, secretory_vertexShader);
	glAttachShader(secretory_shaderProgram, boundaries_fragmentShader);
	glLinkProgram(secretory_shaderProgram);

	// check for errors
	//GLint status;
	glGetShaderiv(secretory_vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Compilation Error: secretory_vertexShader\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(secretory_vertexShader, 262144, &len, data);
		printf("%s", data);
	}
	glGetShaderiv(boundaries_fragmentShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Compilation Error: secretory_fragmentShader\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(boundaries_fragmentShader, 262144, &len, data);
		printf("%s", data);
	}
	glGetProgramiv(secretory_shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Program Link Error: secretory_shaderProgram\n");
	}

	const char* v3 = ciliary_vertexShaderSource;
	//const char* f2 = secretory_fragmentShaderSource;

	//vertex shader
	ciliary_vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(ciliary_vertexShader, 1, &v3, 0);
	glCompileShader(ciliary_vertexShader);

	//program
	ciliary_shaderProgram = glCreateProgram();
	glAttachShader(ciliary_shaderProgram, ciliary_vertexShader);
	glAttachShader(ciliary_shaderProgram, boundaries_fragmentShader);
	glLinkProgram(ciliary_shaderProgram);

	// check for errors
	glGetShaderiv(ciliary_vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Compilation Error: ciliary_vertexShader\n");
		char data[262144];
		int len;
		glGetShaderInfoLog(ciliary_vertexShader, 262144, &len, data);
		printf("%s", data);
	}
	glGetProgramiv(ciliary_shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Shader Program Link Error: ciliary_shaderProgram\n");
	}

}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, GLuint size)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkGLError("createVBO");
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint* vbo)
{
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Reshape callback
////////////////////////////////////////////////////////////////////////////////
void reshape(int width, int height){
	// viewport
	glViewport(0, 0, width, height);
	projection = glm::perspective(glm::radians((float)FOVY), (float)width / (float)height, (float)NEAR_CLIP, (float)FAR_CLIP);
	checkGLError("reshape");
}


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	float millis;

	//CUDA start Timing
	cudaEventRecord(start);

	// run CUDA kernel to generate vertex positions
	runCuda();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Set light position
	glLightfv(GL_LIGHT0, GL_POSITION, LIGHT_POSITION);
	checkGLError("display 0");

	model = glm::mat4(1);
	view = glm::translate(glm::mat4(1), glm::vec3(offset_x, offset_y, zoom_level));
	glm::mat4 mv = view * model;
	glm::mat4 mvp = projection * view * model;

	//Draw EV Agents in default state
	if (get_agent_EV_default_count() > 0) {
		glUseProgram(shaderProgram);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "mvp"), 1, GL_FALSE, &mvp[0][0]);
		checkGLError("display 1a");

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glDrawArrays(GL_POINTS, 0, get_agent_SecretoryCell_s_default_count());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	checkGLError("display 1b");
		
	if (get_agent_SecretoryCell_s_default_count() > 0) {
		glUseProgram(secretory_shaderProgram);
		glUniformMatrix4fv(glGetUniformLocation(secretory_shaderProgram, "mvp"), 1, GL_FALSE, &mvp[0][0]);
		glBindBuffer(GL_ARRAY_BUFFER, vboIdSecretory);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
		glEnableVertexAttribArray(0);

		glDrawArrays(GL_LINES, 0, 4 * get_agent_SecretoryCell_s_default_count());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	checkGLError("display 2");
		
	if (get_agent_CiliaryCell_c_default_count() > 0) {
		glUseProgram(ciliary_shaderProgram);
		glUniformMatrix4fv(glGetUniformLocation(ciliary_shaderProgram, "mvp"), 1, GL_FALSE, &mvp[0][0]);
		glBindBuffer(GL_ARRAY_BUFFER, vboIdCiliary);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
		glEnableVertexAttribArray(0);

		glDrawArrays(GL_LINES, 0, 4 * get_agent_SecretoryCell_s_default_count());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	checkGLError("display 3");
		
	//CUDA stop timing
	cudaEventRecord(stop);
	glFlush();
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);
	frame_time += millis;

	if (frame_count == display_rate){
		char title[100];
		sprintf(title, "Exec & Render Total: %6.3f (FPS), %6.3f millis p/frame. | iters: %lu", display_rate / (frame_time / 1000.0f), frame_time / display_rate, iterations);
		glutSetWindowTitle(title);

		//reset
		frame_count = 0;
		frame_time = 0.0;
	}
	else{
		frame_count++;
	}
	checkGLError("display 4");

	glutSwapBuffers();
	glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key) {

	case(87) :
	case(119) :
			  // W == 87, w == 119
			  offset_y -= 1;
		//printf("offset_y=%f", offset_y);
		break;
	case(65) :
	case(97) :
			 // A == 65, a == 97
			 offset_x += 1;
		//printf("offset_x=%f", offset_x);
		break;
	case(83) :
	case(115) :
			  // S == 83, s == 115
			  offset_y += 1;
		//printf("offset_y=%f", offset_y);
		break;

	case(68) :
	case(100) :
			  // D == 68, d == 100
			  offset_x -= 1;
		//printf("offset_x=%f", offset_x);
		break;

	case(75) :
	case(107): // xoom in x 10 - k or K
		translate_z += 10.0;
		zoom_level += 10.0;
		printf("zoom in: translate_z=%f\n", translate_z);
		break;
	case(73) :
	case(105) :
			  // zoom in - I or i
			  translate_z += 1.0;
			  zoom_level += 1.0;
			  printf("zoom in: translate_z=%f\n", translate_z);
		break;

	case(76) :
	case(108): // zoom out x10 - l or L
		translate_z -= 10.0;
		zoom_level -= 10.0;
		printf("zoom out: translate_z=%f\n", translate_z);
		break;
	case(79) :
	case(111) :
			  // zoom out - O or o
			  translate_z -= 1.0;
			  zoom_level -= 1.0;
			  printf("zoom out: translate_z=%f\n", translate_z);
		break;

		// P == 80, p == 112
	case(80) :
	case(112) :
			  if (paused)
				  printf("un-paused\n");
			  else
				  printf("paused\n");
		paused = !paused;
		break;
		// Esc == 27
	case(27) :
		
		exit(EXIT_SUCCESS);
		// Space == 32
	case(GLUT_KEY_RIGHT) :
		singleIteration();
		fflush(stdout);
		break;
	}
	checkGLError("keyboard");
}


void close() {
	deleteVBO(&sphereVerts);
	deleteVBO(&vboIdSecretory);
	deleteVBO(&vboIdCiliary);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cleanup();
}

void special(int key, int x, int y){
}


////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
	checkGLError("mouse");
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4) {
		translate_z += (float)(dy * VIEW_DISTANCE * 0.001);
	}

	mouse_old_x = x;
	mouse_old_y = y;
	//checkGLError("motion");
}

void checkGLError(char* caller){
	int Error;
	if ((Error = glGetError()) != GL_NO_ERROR)
	{
		const char* Message = (const char*)gluErrorString(Error);
		fprintf(stderr, "OpenGL Error (caller:%s) [%d] : %s \n", caller, Error, Message);
	}
}
