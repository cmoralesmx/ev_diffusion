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
#include "shader_s.h"

extern float *h_xs;
extern float *h_ys;
#define FOVY 45.0

GLuint vboIdCiliary;
GLuint vaoIdCiliary;
GLuint vboIdSecretoryInitial, vboIdSecretoryDefault;
GLuint vaoIdSecretoryInitial, vaoIdSecretoryDefault;
cudaGraphicsResource *CGRsecretoryInitial, *CGRsecretoryDefault, *CGRciliary;
bool display_walls = true;
bool thick_walls = false;
bool evSpheres = true;
bool single_iteration_flag = false;
bool visualisation_enabled = true;

//Simulation output buffers/textures
cudaGraphicsResource_t EV_default_cgr, EV_initial_cgr, EV_collision_ev_default_cgr;
// vertex Shader
GLuint vertexShader;
GLuint fragmentShader;
GLuint shaderProgram;
GLuint vs_mapIndex;

GLuint secretoryInitial_vertexShader, secretoryDefault_vertexShader;
GLuint secretoryInitial_shaderProgram, secretoryDefault_shaderProgram;
GLuint boundaries_fragmentShader;
GLuint ciliary_vertexShader;
GLuint ciliary_shaderProgram;

// bo variables
GLuint sphereVerts, sphereInitialVerts, sphereCollisionEvDefaultVerts;

Shader *evPtsShader, *evGeoShader, *evInitialShader, *evCollisionEvDefaultShader;
Shader *secretoryInitialShader, *secretoryInitialThickShader, *secretoryDefaultShader, *secretoryDefaultThickShader;
Shader *ciliaryShader, *ciliaryThickShader;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
//float translate_z = -VIEW_DISTANCE;  // This parameter comes from the visualisation.h file

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

//GPU Kernels
__global__ void output_EV_agent_to_VBO(xmachine_memory_EV_list* agents, glm::vec3* vbo){

	//global thread index
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	vbo[index].x = agents->x[index];
	vbo[index].y = agents->y[index];
	vbo[index].z = agents->radius_um[index];
}

__global__ void output_SecretoryCell_agent_to_VBO(xmachine_memory_SecretoryCell_list* agents, float* vbo) {

	//global thread index
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int index2 = index * 8;

	vbo[index2] = agents->p1_x[index];
	vbo[index2 + 1] = agents->p1_y[index];
	vbo[index2 + 2] = agents->unit_normal_x[index];
	vbo[index2 + 3] = agents->unit_normal_y[index];

	vbo[index2 + 4] = agents->p2_x[index];
	vbo[index2 + 5] = agents->p2_y[index];
	vbo[index2 + 6] = 0; // agents->unit_normal_x[index];
	vbo[index2 + 7] = 0; // agents->unit_normal_y[index];
}

__global__ void output_CiliaryCell_agent_to_VBO(xmachine_memory_CiliaryCell_list* agents, float* vbo) {

	//global thread index
	int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int index2 = index * 8;

	vbo[index2] = agents->p1_x[index];
	vbo[index2 + 1] = agents->p1_y[index];
	vbo[index2 + 2] = agents->unit_normal_x[index];
	vbo[index2 + 3] = agents->unit_normal_y[index];

	vbo[index2 + 4] = agents->p2_x[index];
	vbo[index2 + 5] = agents->p2_y[index];
	vbo[index2 + 6] = 0; // agents->unit_normal_x[index];
	vbo[index2 + 7] = 0;
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

	// register callbacks
	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutCloseFunc(close);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	
	evPtsShader = new Shader("./shaders/ev_points_vs.glsl", "./shaders/ev_points_fs.glsl");
	evGeoShader = new Shader("./shaders/ev_spheres_vs.glsl", "./shaders/ev_spheres_fs.glsl", "./shaders/ev_spheres_gs.glsl");
	evInitialShader = new Shader("./shaders/ev_initial_vs.glsl", "./shaders/ev_spheres_fs.glsl", "./shaders/ev_spheres_gs.glsl");
	evCollisionEvDefaultShader = new Shader("./shaders/ev_collision_vs.glsl", "./shaders/ev_spheres_fs.glsl", "./shaders/ev_spheres_gs.glsl");
	
	secretoryInitialShader = new Shader("./shaders/cells_all_thin_vs.glsl", "./shaders/cells_secretory_initial_fs.glsl");
	secretoryInitialThickShader = new Shader("./shaders/cells_all_thick_vs.glsl", "./shaders/cells_secretory_initial_fs.glsl", "./shaders/cells_all_thick_gs.glsl");
	secretoryDefaultShader = new Shader("./shaders/cells_all_thin_vs.glsl", "./shaders/cells_secretory_default_fs.glsl");
	secretoryDefaultThickShader = new Shader("./shaders/cells_all_thick_vs.glsl", "./shaders/cells_secretory_default_fs.glsl", "./shaders/cells_all_thick_gs.glsl");

	ciliaryShader = new Shader("./shaders/cells_all_thin_vs.glsl", "./shaders/cells_ciliary_fs.glsl");
	ciliaryThickShader = new Shader("./shaders/cells_all_thick_vs.glsl", "./shaders/cells_ciliary_fs.glsl", "./shaders/cells_all_thick_gs.glsl");
	
	// create VBO's
	createVBO(&sphereVerts, get_agent_EV_MAX_count() * sizeof(glm::vec3));
	// registers a GraphicsGLResource with CUDA for interop access
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&EV_default_cgr, sphereVerts, cudaGraphicsMapFlagsNone));

	createVBO(&sphereCollisionEvDefaultVerts, get_agent_EV_MAX_count() * sizeof(glm::vec3));
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&EV_collision_ev_default_cgr, sphereCollisionEvDefaultVerts, cudaGraphicsMapFlagsNone));

	createVBO(&sphereInitialVerts, get_agent_EV_MAX_count() * sizeof(glm::vec3));
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&EV_initial_cgr, sphereInitialVerts, cudaGraphicsMapFlagsNone));

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

	float min_x = glm::min(glm::min(min_CiliaryCell_c_default_p1_x_variable(), min_CiliaryCell_c_default_p2_x_variable()),
		glm::min(min_SecretoryCell_s_default_p1_x_variable(), min_SecretoryCell_s_default_p2_x_variable()));
	float min_y = glm::min(glm::min(min_CiliaryCell_c_default_p1_y_variable(), min_CiliaryCell_c_default_p2_y_variable()),
		glm::min(min_SecretoryCell_s_default_p1_y_variable(), min_SecretoryCell_s_default_p2_y_variable()));
	float max_x = glm::max(glm::max(max_CiliaryCell_c_default_p1_x_variable(), max_CiliaryCell_c_default_p2_x_variable()),
		glm::max(max_SecretoryCell_s_default_p1_x_variable(), max_SecretoryCell_s_default_p2_x_variable()));
	float max_y = glm::max(glm::max(max_CiliaryCell_c_default_p1_y_variable(), max_CiliaryCell_c_default_p2_y_variable()),
		glm::max(max_SecretoryCell_s_default_p1_y_variable(), max_SecretoryCell_s_default_p2_y_variable()));
	glm::vec2 center;
	model = glm::mat4(1);
	if (max_y - min_y > max_x - min_x) {
		// the model is taller than wider, we need to rotate it to maximise screen space usage
		model = glm::rotate(model, glm::radians(-90.0f), glm::vec3(0, 0, 1.0f));
		center = glm::vec2((max_y - min_y)/2.0f, (max_x - min_x)/2.0f);
		printf("ROTATED Model coordinate ranges x:[%4.2f - %4.2f], y:[%4.2f - %4.2f]\n", min_y, max_y, min_x, max_x);
	}
	else {
		center = glm::vec2((max_x - min_x) / 2.0f, ((max_y - min_y) / 2.0f)*1.4f);
		printf("Model coordinate ranges x:[%4.2f - %4.2f], y:[%4.2f - %4.2f]\n", min_x, max_x, min_y, max_y);
	}
	model = glm::translate(model, glm::vec3(-center, 0));

	printf("Secretory cells: %d initial, %d default\n", get_agent_SecretoryCell_s_initial_count(), get_agent_SecretoryCell_s_default_count());
	printf("Ciliary cells: %d default, \n", get_agent_CiliaryCell_c_default_count());

	if (get_agent_SecretoryCell_s_initial_count() > 0) {
		size = get_agent_SecretoryCell_MAX_count() * 8 * sizeof(float);

		createVBO(&vboIdSecretoryDefault, size);
		gpuErrchk(cudaGraphicsGLRegisterBuffer(&CGRsecretoryDefault, vboIdSecretoryDefault, cudaGraphicsMapFlagsNone));
		checkGLError("createVBO - secretoryDefault");
		
		createVBO(&vboIdSecretoryInitial, size);
		gpuErrchk(cudaGraphicsGLRegisterBuffer(&CGRsecretoryInitial, vboIdSecretoryInitial, cudaGraphicsMapFlagsReadOnly));
		checkGLError("createVBO - secretoryInitial");

		float *secretory_dptr;
		size_t size_secretory;
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, &CGRsecretoryInitial, 0));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&secretory_dptr, &size_secretory, CGRsecretoryInitial));

		//cuda block size
		tile_size = (int)ceil((float)get_agent_SecretoryCell_s_initial_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

		output_SecretoryCell_agent_to_VBO << < grid, threads >> > (get_device_SecretoryCell_s_initial_agents(), secretory_dptr);
		cudaDeviceSynchronize();
		gpuErrchkLaunch();

		gpuErrchk(cudaGraphicsUnmapResources(1, &CGRsecretoryInitial, 0));


		gpuErrchk(cudaGraphicsMapResources(1, &CGRsecretoryDefault, 0));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&secretory_dptr, &size_secretory, CGRsecretoryDefault));

		//cuda block size
		tile_size = (int)ceil((float)get_agent_SecretoryCell_s_initial_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

		output_SecretoryCell_agent_to_VBO << < grid, threads >> > (get_device_SecretoryCell_s_initial_agents(), secretory_dptr);
		cudaDeviceSynchronize();
		gpuErrchkLaunch();

		gpuErrchk(cudaGraphicsUnmapResources(1, &CGRsecretoryDefault, 0));
	}

	if (get_agent_CiliaryCell_c_default_count() > 0) {
		size = get_agent_CiliaryCell_MAX_count() * 8 * sizeof(float);
		createVBO(&vboIdCiliary, size);
		gpuErrchk(cudaGraphicsGLRegisterBuffer(&CGRciliary, vboIdCiliary, cudaGraphicsMapFlagsReadOnly));
		checkGLError("createVBO - ciliary");

		float *ciliary_dptr;
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
	printf("There are %d initial, and %d default EVs\n", get_agent_EV_initial_count(), get_agent_EV_default_count());
	// start rendering mainloop
	glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
	if (!paused || (paused && single_iteration_flag)){
#ifdef SIMULATION_DELAY
		delay_count++;
		if (delay_count == SIMULATION_DELAY){
			delay_count = 0;
			singleIteration();
		}
#else
		singleIteration();
		++iterations;
		if(single_iteration_flag)
			single_iteration_flag = false;
#endif
	}

	//kernals sizes
	int threads_per_tile = 256;
	int tile_size;
	dim3 grid;
	dim3 threads;

	glm::vec3 *dptr, *dptr_initial, *dptr_collision_ev_default;

	if (get_agent_EV_default_count() > 0 && visualisation_enabled)
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
	if (get_agent_EV_initial_count() > 0 && visualisation_enabled)
	{
		size_t accessibleBufferSize = 0;
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, &EV_initial_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr_initial, &accessibleBufferSize, EV_initial_cgr));
		//cuda block size
		tile_size = (int)ceil((float)get_agent_EV_initial_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

		output_EV_agent_to_VBO << < grid, threads >> >(get_device_EV_initial_agents(), dptr_initial);
		gpuErrchkLaunch();
		// unmap buffer object
		gpuErrchk(cudaGraphicsUnmapResources(1, &EV_initial_cgr));
	}
	if (get_agent_EV_collision_default_count() > 0 && visualisation_enabled)
	{
		size_t accessibleBufferSize = 0;
		// map OpenGL buffer object for writing from CUDA
		gpuErrchk(cudaGraphicsMapResources(1, &EV_collision_ev_default_cgr));
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dptr_collision_ev_default, &accessibleBufferSize, EV_collision_ev_default_cgr));
		//cuda block size
		tile_size = (int)ceil((float)get_agent_EV_collision_default_count() / threads_per_tile);
		grid = dim3(tile_size, 1, 1);
		threads = dim3(threads_per_tile, 1, 1);

		output_EV_agent_to_VBO << < grid, threads >> >(get_device_EV_collision_default_agents(), dptr_collision_ev_default);
		gpuErrchkLaunch();
		// unmap buffer object
		gpuErrchk(cudaGraphicsUnmapResources(1, &EV_collision_ev_default_cgr));
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

	view = glm::translate(glm::mat4(1), glm::vec3(offset_x, offset_y, zoom_level));
	glm::mat4 mv = view * model;
	glm::mat4 mvp = projection * view * model;

	//Draw EV Agents in default state
	if (get_agent_EV_default_count() > 0) {
		if (evSpheres) {
			evGeoShader->use();
			evGeoShader->setMat4("mvp", mvp);
		}
		else {
			evPtsShader->use();
			evPtsShader->setMat4("mvp", mvp);
		}

		glBindBuffer(GL_ARRAY_BUFFER, sphereVerts);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glDrawArrays(GL_POINTS, 0, get_agent_EV_default_count());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	checkGLError("display 1b");
	if (get_agent_EV_initial_count() > 0) {
		evInitialShader->use();
		evInitialShader->setMat4("mvp", mvp);

		glBindBuffer(GL_ARRAY_BUFFER, sphereInitialVerts);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glDrawArrays(GL_POINTS, 0, get_agent_EV_initial_count());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	if (get_agent_EV_collision_default_count() > 0) {
		evCollisionEvDefaultShader->use();
		evCollisionEvDefaultShader->setMat4("mvp", mvp);

		glBindBuffer(GL_ARRAY_BUFFER, sphereCollisionEvDefaultVerts);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)(2 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glDrawArrays(GL_POINTS, 0, get_agent_EV_collision_default_count());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	// Walls
	if (display_walls) {
		if (get_agent_SecretoryCell_s_initial_count() > 0) {
			if (thick_walls) {
				secretoryInitialThickShader->use();
				secretoryInitialThickShader->setMat4("mvp", mvp);
			}
			else {
				secretoryInitialShader->use();
				secretoryInitialShader->setMat4("mvp", mvp);
			}

			glBindBuffer(GL_ARRAY_BUFFER, vboIdSecretoryInitial);
			// position
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
			glEnableVertexAttribArray(0);
			// unit normal
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
			glEnableVertexAttribArray(1);

			glDrawArrays(GL_LINES, 0, 4 * get_agent_SecretoryCell_s_initial_count());
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		if (get_agent_SecretoryCell_s_default_count() > 0) {
			if (thick_walls) {
				secretoryDefaultThickShader->use();
				secretoryDefaultThickShader->setMat4("mvp", mvp);
			}
			else {
				secretoryDefaultShader->use();
				secretoryDefaultShader->setMat4("mvp", mvp);
			}
			// Secretory_s_default_cgr
			glBindBuffer(GL_ARRAY_BUFFER, vboIdSecretoryDefault);
			// position
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
			glEnableVertexAttribArray(0);
			// unit normal
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
			glEnableVertexAttribArray(1);

			glDrawArrays(GL_LINES, 0, 4 * get_agent_SecretoryCell_s_default_count());
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
		

		if (get_agent_CiliaryCell_c_default_count() > 0) {
			if (thick_walls) {
				ciliaryThickShader->use();
				ciliaryThickShader->setMat4("mvp", mvp);
			}
			else {
				ciliaryShader->use();
				ciliaryShader->setMat4("mvp", mvp);
			}

			glBindBuffer(GL_ARRAY_BUFFER, vboIdCiliary);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
			glEnableVertexAttribArray(1);

			glDrawArrays(GL_LINES, 0, 4 * get_agent_CiliaryCell_c_default_count());
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
	}
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

	glutSwapBuffers();
	glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
#define KEY_A 65
// 68 D, 100 d
#define KEY_E 69
#define KEY_N 78
#define KEY_P 80
#define KEY_Q 81
#define KEY_S 83
#define KEY_T 84
#define KEY_V 86
#define KEY_W 87
#define KEY_X 88
#define KEY_Z 90
#define KEY_a 97
#define KEY_e 101
#define KEY_n 110
#define KEY_p 112
#define KEY_q 113
#define KEY_s 115
#define KEY_t 116
#define KEY_v 118
#define KEY_w 119
#define KEY_x 120
#define KEY_z 122
#define KEY_ESC 27
#define KEY_SPACE_BAR 32
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key) {
	case(KEY_s):
	case(KEY_S): 
		zoom_level += 10.0;
		break;
	case(KEY_a):
	case(KEY_A):
		zoom_level += 1.0;
		break;
	case(KEY_x):
	case(KEY_X):
		zoom_level -= 10.0;
		break;
	case(KEY_z):
	case(KEY_Z):
		zoom_level -= 1.0;
		break;
	case(KEY_SPACE_BAR):
		if (paused)
			printf("Simulation un-paused\n");
		else
			printf("Simulation paused\n");
		paused = !paused;
		break;
	case(KEY_V):
	case(KEY_v):
		visualisation_enabled = !visualisation_enabled;
		if (visualisation_enabled)
			printf("Visualisation enabled\n");
		else
			printf("Visualisation disabled\n");
		break;
	case(KEY_q):
	case(KEY_Q):
		evSpheres = !evSpheres;
		if (evSpheres)
			printf("Display EVs as fake spheres\n"); 
		else
			printf("Display EVs as points\n");
		break;
	case(KEY_w):
	case(KEY_W):
		display_walls = !display_walls;
		break;
	case(KEY_T):
	case(KEY_t):
		thick_walls = !thick_walls;
		break;
	case(KEY_ESC) :
		exit(EXIT_SUCCESS);
	case(KEY_N):
	case(KEY_n):
		single_iteration_flag = true;
		singleIteration();
		fflush(stdout);
		break;
	default:
		printf("key: %u\n", key);
		break;
	}
	checkGLError("keyboard");
}


void close() {
	deleteVBO(&sphereVerts);
	deleteVBO(&vboIdSecretoryInitial);
	deleteVBO(&vboIdSecretoryDefault);
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
		offset_y -= dy * 0.2f;
		offset_x += dx * 0.2f;
	}
	else if (mouse_buttons & 4) {
		zoom_level += (float)(dy * VIEW_DISTANCE * 0.01);
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
