
/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.s
 * 
 */

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#ifndef VISUALISATION
//#define INSTRUMENT_ITERATIONS 1
//#define INSTRUMENT_AGENT_FUNCTIONS 1
//#define INSTRUMENT_INIT_FUNCTIONS 1
//#define INSTRUMENT_STEP_FUNCTIONS 1
//#define INSTRUMENT_EXIT_FUNCTIONS 1
//#define OUTPUT_POPULATION_PER_ITERATION 1
#endif

#include <header.h>
#define _USE_MATH_DEFINES
#include <math_constants.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "myVisualisation.h"

int exiting_early = 0;

__device__ __host__ float degrees(float radians){
	return radians * (180 / M_PI);
}

__device__ __host__ float radians(float degrees){
	return degrees * (M_PI / 180.0);
}

__device__ float dotprod(float x1, float y1, float x2, float y2) {
	return x1*x2 + y1*y2;
}

__device__ __host__ float vlength(float x, float y) {
	return sqrtf(x*x + y*y);
}

// https://github.com/takagi/cl-cuda/blob/master/include/float4.h
__device__ float2 float2_add(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
__device__ float4 float4_add(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ float2 float2_sub(float2 a, float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
__device__ float4 float4_sub(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ float2 float2_scale(float2 a, float k)
{
	return make_float2(a.x * k, a.y * k);
}
__device__ float4 float4_scale(float4 a, float k)
{
	return make_float4(a.x * k, a.y * k, a.z * k, a.w * k);
}
__device__ float float2_dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}
__device__ float float4_dot(float4 a, float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/*
Scales b by scale and adds it to a
*/
__device__ float2 add_scaled(float2 a, float2 b, float scale) {
	return float2_add(a, float2_scale(b, scale));
}

/*
Creates a vector of lenght u parallel to vec
*/
__device__ float2 parallel(float vec_x, float vec_y, float u) {
	float factor = (u / vlength(vec_x, vec_y));
	return make_float2(vec_x * factor, vec_y * factor);
}
/*
	Returns the length of the projection of vector A in the direction of vector B
*/
__device__ float projection(float a_x, float a_y, float b_x, float b_y) {
	float length = vlength(a_x, a_y);
	float lengthVec = vlength(b_x, b_y);
	if (length == 0 || lengthVec == 0)
		return 0;
	return (a_x * b_x + a_y * b_y) / lengthVec;
}

__device__ float2 float2_project(float2 v1, float2 v2) {
	float mag = projection(v1.x, v1.y, v2.x, v2.y);
	return parallel(v2.x, v2.y, mag);
}

__device__ float2 project(float v1x, float v1y, float v2x, float v2y) {
	return float2_project(make_float2(v1x, v1y), make_float2(v2x, v2y));
}

/*
	Compute the distance between two agents using the coordinates of their centers
*/
__device__ float euclidean_distance(xmachine_memory_EV* agent, 
	float message_x, float message_y)
{
	return sqrtf((agent->x - message_x) * (agent->x - message_x) +
		(agent->y - message_y) * (agent->y - message_y));
}

__FLAME_GPU_STEP_FUNC__  void checkStopConditions() {
	if(exiting_early > 0){
		set_exit_early();
	}
	float dt = *get_dt();
	for(int i=0; i < get_agent_EV_default_count(); i++) {
		float x = get_EV_default_variable_x(i);
		float y = get_EV_default_variable_y(i);
		int id = get_EV_default_variable_id(i);
		if(!isfinite(x) || !isfinite(y)) {
			printf("EV id:%d has NaN or Inf coordinates.\n", id);
			++exiting_early;
		}
	}
}

__device__ int checkNanInfVelocity(xmachine_memory_EV* agent, float caller_id){
    return 0;
}

__device__ int checkNanInfLocation(xmachine_memory_EV* agent, float caller_id){
    return 0;
}

__device__ int checkValueNanInf(xmachine_memory_EV* agent, float value, float caller_id){
	return 0;
}

__device__ int checkValueNanInf(xmachine_memory_EV* agent, float2 value, float caller_id){
	return 0;
}

__FLAME_GPU_INIT_FUNC__ void precompute_values() {
	//srand(time(NULL));
	printf("Simulation timestep, dt=%f\n", *get_dt());
	if (*get_ev_collisions() > 0) {
		printf("Ev-Ev collisions are enabled\n");
	}
	else {
		printf("Ev-Ev collisions are disabled\n");
	}
	if (*get_boundaries() > 0) {
		printf("Collision with the boundaries are enabled\n");
	}
	else {
		printf("Collision with the boundaries are disabled\n");
	}
	
	if (*get_brownian_motion_1d() > 0) {
		printf("brownian_motion_1d is enabled\n");
	}
	else {
		printf("brownian_motion_1d is disabled\n");
	}
	if (*get_brownian_motion_2d() > 0) {
		printf("brownian_motion_2d is enabled\n");
		float bm_factor =  *get_dt() / *get_bm_frequency();
		set_bm_factor(&bm_factor);
		printf("brownian motion frequency in seconds: %f, factor:%f\n", *get_bm_frequency(), *get_bm_factor());
	}
	else {
		printf("brownian_motion_2d is disabled\n");
	}
	if (*get_apoptosis() > 0) {
		printf("Apoptosis is enabled, threshold: %f\n", *get_apoptosis_threshold());
	} else {
		printf("Apoptosis is disabled\n");
	}
	if (*get_ev_secretion() > 0) {
		printf("EV secretion is enabled, interval %f seconds, p. threshold of %f\n",
			*get_ev_secretion_interval(), *get_ev_secretion_threshold());
		printf("Min radius: %d, Max radius: %d\n", *get_min_ev_radius(), *get_max_ev_radius());
	}
	else {
		printf("EV secretion is disabled\n");
	}
	float val0 = 6 * M_PI * *get_const_water_dynamic_viscosity_pas();
	set_const_6_pi_dynamic_viscosity(&val0);
	printf("set_const_6_pi_dynamic_viscosity: %f\n", val0);
	float val1 = 6 * M_PI * *get_const_water_kinematic_viscosity_kg_ums();
	set_const_6_pi_kinematic_viscosity(&val1);
	printf("set_const_6_pi_kinematic_viscosity: %f\n", val1);
	float val2 = *get_const_Boltzmann() * *get_const_Temperature_K();
	printf("boltzmann: %.35f\n", *get_const_Boltzmann());
	set_const_boltzmann_x_temp(&val2);
	printf("set_const_boltzmann_x_temp: %.30f\n", val2);
	// min x, min y, max x, max y
	fvec4 minmax = fvec4(
		fmin(fmin(min_CiliaryCell_c_default_p1_x_variable(), min_CiliaryCell_c_default_p2_x_variable()),
			fmin(min_SecretoryCell_s_default_p1_x_variable(), min_SecretoryCell_s_default_p2_x_variable())) - 1,
		fmin(fmin(min_CiliaryCell_c_default_p1_y_variable(), min_CiliaryCell_c_default_p2_y_variable()),
			fmin(min_SecretoryCell_s_default_p1_y_variable(), min_SecretoryCell_s_default_p2_y_variable())) - 1,
		fmax(fmax(max_CiliaryCell_c_default_p1_x_variable(), max_CiliaryCell_c_default_p2_x_variable()),
			fmax(max_SecretoryCell_s_default_p1_x_variable(), max_SecretoryCell_s_default_p2_x_variable())) + 1,
		fmax(fmax(max_CiliaryCell_c_default_p1_y_variable(), max_CiliaryCell_c_default_p2_y_variable()),
			fmax(max_SecretoryCell_s_default_p1_y_variable(), max_SecretoryCell_s_default_p2_y_variable())) + 1
		);
	set_minMaxCoords(&minmax);
	printf("Limits on x:[%.2f, %.2f], y:[%.2f, %.2f]\n", minmax.x, minmax.z, minmax.y, minmax.w);
}

/*

*/
__FLAME_GPU_FUNC__ int secretory_cell_initialization(xmachine_memory_SecretoryCell* cell, 
	RNG_rand48* rand48){
		if(cell->min_ev_radius == -1)
		{
			if(rnd<CONTINUOUS>(rand48) <= 0.5){
				// exosome-size range
				cell->min_ev_radius = min_ev_radius;
				cell->max_ev_radius = threshold_ex_mv;
			} else {
				// microvesicles-size range
				cell->min_ev_radius = threshold_ex_mv;
				cell->max_ev_radius = max_ev_radius;
			}
		}
	return 0;
}

__FLAME_GPU_FUNC__ int ev_initialization_biogenesis(xmachine_memory_EV* ev){
	return 0;
}

__FLAME_GPU_FUNC__ int ev_initialization_default(xmachine_memory_EV* ev){
	return 0;
}

__FLAME_GPU_FUNC__ int ev_initialization_disabled(xmachine_memory_EV* ev){
	return 0;
}

__FLAME_GPU_STEP_FUNC__ int increase_iteration(){
	unsigned int iter = (*get_iteration() + 1);
	set_iteration(&iter);
	return 0;
}

/*
	EVs are subject to an appoptotic attempt per second during their lifespan.
	The apoptosis_threshold reflects the probability of the EV dying at a given time
*/
__FLAME_GPU_FUNC__ int ev_default_apoptosis(xmachine_memory_EV* agent, RNG_rand48* rand48){
	
	if(agent->apoptosis_timer > apoptosis_frequency){
		agent->apoptosis_timer = 0;
		if(rnd<CONTINUOUS>(rand48) < apoptosis_threshold){
			return 1;
		}
	} else {
		agent->apoptosis_timer += dt;
	}
	return 0;
}

/*
	Uses the Box-Mueller transform to generate a pair of normally distributed random numbers
	by sampling from two uniformly distributed RNG.
	In the original algorithm, the sampled values are transformed into cartesian coordinates,
	here they become the new velocity for the next step
	Due to limitations in FlameGPU, we sample both continuos numbers from the same RNG.
*/
__FLAME_GPU_FUNC__ int brownian_movement_1d(xmachine_memory_EV* agent, RNG_rand48* rand48) {	
	agent->bm_r.x = sqrtf(-2.0 * log(rnd<CONTINUOUS>(rand48)));
	agent->bm_r.y = agent->bm_r.x * agent->mdd125 * bm_factor;
	
	// the product of agent->bm_r * (cos|sin)(theta) becomes the displacement factor to use in this iteration
	agent->vx = agent->bm_r.y * cospif(2 * rnd<CONTINUOUS>(rand48));
	
	return 0;
}

__FLAME_GPU_FUNC__ int brownian_movement_2d(xmachine_memory_EV* agent, RNG_rand48* rand48) {
	float theta;
	agent->bm_r.x = sqrtf(-2*log(rnd<CONTINUOUS>(rand48)));
	agent->bm_r.y = agent->bm_r.x * agent->mdd125 * bm_factor;
	theta = 2 * rnd<CONTINUOUS>(rand48);
	
	agent->bm_vx = (agent->bm_r.y * cospif(theta));
	agent->bm_vy = (agent->bm_r.y * sinpif(theta));
	agent->vx = agent->bm_vx;
	agent->vy = agent->bm_vy;
	
	agent->last_bm = 0;
	return 0;
}

__device__ void ev_reset(xmachine_memory_EV* agent){
	agent->closest_ciliary_cell_id = -1;
	agent->closest_ciliary_cell_distance = 100;
	agent->closest_secretory_cell_id = -1;
	agent->closest_secretory_cell_distance = 100;
	agent->closest_ev_default_id = 0;
	agent->closest_ev_default_distance = 100;
	agent->closest_ev_biogenesis_id = 0;
	agent->closest_ev_biogenesis_distance = 100;
	agent->dbgSegC.x = 0;
	agent->dbgSegC.y = 0;
	agent->dbgSegC.z = 0;
	agent->dbgSegC.w = 0;
	agent->dbgSegCdispl.x = 0;
	agent->dbgSegCdispl.y = 0;
	agent->dbgSegCdispl.z = 0;
	agent->dbgSegCdispl.w = 0;
}

__device__ float displacement_sq(xmachine_memory_EV* ag){
	return (ag->x - ag->x_1) * (ag->x - ag->x_1) + (ag->y - ag->y_1) * (ag->y - ag->y_1);
}

__FLAME_GPU_FUNC__ int is_OOB(xmachine_memory_EV* ag){
	// 1000x1000 = 2,000,000, 3000x4000=25,000,000
	if (ag->x < minMaxCoords.x || ag->x > minMaxCoords.z || ag->y < minMaxCoords.y || ag->y > minMaxCoords.w)
		return 1;
	return 0;
}

__FLAME_GPU_FUNC__ int reset_state(xmachine_memory_EV* agent) {
	ev_reset(agent);
	return 0;
}

__FLAME_GPU_FUNC__ int reset_state_biogenesis(xmachine_memory_EV* agent) {
	ev_reset(agent);
	return 0;
}

__FLAME_GPU_FUNC__ int compute_displacement_cs(xmachine_memory_EV* agent){
	agent->displacementSq = displacement_sq(agent);
	return 0;
}
__FLAME_GPU_FUNC__ int compute_displacement_cc(xmachine_memory_EV* agent){
	agent->displacementSq = displacement_sq(agent);
	return 0;
}
__FLAME_GPU_FUNC__ int compute_displacement_cd(xmachine_memory_EV* agent){
	agent->displacementSq = displacement_sq(agent);
	return 0;
}
__FLAME_GPU_FUNC__ int compute_displacement_ci(xmachine_memory_EV* agent){
	agent->displacementSq = displacement_sq(agent);
	return 0;
}

__FLAME_GPU_FUNC__ int reset_state_collision_secretory(xmachine_memory_EV* agent) {
	ev_reset(agent);
	return 0;
}
__FLAME_GPU_FUNC__ int reset_state_collision_ciliary(xmachine_memory_EV* agent) {
	ev_reset(agent);
	return 0;
}
__FLAME_GPU_FUNC__ int reset_state_collision_default(xmachine_memory_EV* agent) {
	ev_reset(agent);
	return 0;
}
__FLAME_GPU_FUNC__ int reset_state_collision_biogenesis(xmachine_memory_EV* agent) {
	ev_reset(agent);
	return 0;
}
// These should not die but stay disabled
__FLAME_GPU_FUNC__ int disable_EV_cs(xmachine_memory_EV* agent) {
	agent->disabledAt = iteration;
	agent->disabledReason = 1;
	return 0;
}
__FLAME_GPU_FUNC__ int disable_EV_cc(xmachine_memory_EV* agent) {
	agent->disabledAt = iteration;
	agent->disabledReason = 2;
	return 0;
}
__FLAME_GPU_FUNC__ int disable_EV_cd(xmachine_memory_EV* agent) {
	agent->disabledAt = iteration;
	agent->disabledReason = 3;
	return 0;
}
__FLAME_GPU_FUNC__ int disable_EV_cb(xmachine_memory_EV* agent) {
	agent->disabledAt = iteration;
	agent->disabledReason = 4;
	return 0;
}

/*
	EVs in biogenesis state have the 'time_in_biogenesis_state' extended when colliding with
	an EV in default state. However, repeating this extensions several times
	can lead to undefined behaviour.
	To prevent this, the max extension allowed is teice the time in biogenesis state.
	Extensions longer than this value lead to apoptosis.
*/
__FLAME_GPU_FUNC__ int ev_biogenesis_apoptosis(xmachine_memory_EV* agent){
	agent->disabledAt = iteration;
	agent->disabledReason = 5;
	return 1;
}

__FLAME_GPU_FUNC__ int biogenesis_to_default(xmachine_memory_EV* agent) {
	agent->apoptosis_timer = 0;
	agent->defaultAt = iteration;
	return 0;
}

/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_EV. This represents a single agent instance and can be modified directly.
 
 */
 __device__ void moveEv(xmachine_memory_EV* agent){
	agent->x_1 = agent->x;
	agent->y_1 = agent->y;

	agent->x += agent->vx;
	agent->y += agent->vy;
	agent->age += dt;
 }
__FLAME_GPU_FUNC__ int moveDefault(xmachine_memory_EV* agent){
	moveEv(agent);
	agent->last_bm += dt;
    return 0;
}
__FLAME_GPU_FUNC__ int moveBiogenesis(xmachine_memory_EV* agent) {
	moveEv(agent);
	return 0;
}

/**
 * output_data FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_EV. Represents an agent instance, can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list.
 */
__FLAME_GPU_FUNC__ int output_location_ev_default(xmachine_memory_EV* agent, xmachine_message_location_ev_default_list* location_messages){
	add_location_ev_default_message(location_messages, agent->id, agent->x, agent->y,
		agent->z, agent->radius_um, agent->mass_ag, 
		agent->vx, agent->vy);
    return 0;
}
__FLAME_GPU_FUNC__ int output_location_ev_biogenesis(xmachine_memory_EV* agent, xmachine_message_location_ev_biogenesis_list* location_messages) {
	add_location_ev_biogenesis_message(location_messages, agent->id, agent->x, agent->y,
			agent->z, agent->radius_um, agent->mass_ag, agent->vx, agent->vy);
	return 0;
}

__FLAME_GPU_FUNC__ int output_ciliary_cell_location(xmachine_memory_CiliaryCell* agent, xmachine_message_ciliary_cell_location_list* location_messages) {

	add_ciliary_cell_location_message(location_messages, agent->id, agent->x, agent->y, 0,
		agent->p1_x, agent->p1_y, agent->p2_x, agent->p2_y, agent->direction_x, agent->direction_y,
		agent->direction_x_unit, agent->direction_y_unit, agent->direction_length, 
		agent->normal_x, agent->normal_y, agent->unit_normal_x, agent->unit_normal_y, agent->normal_length
	);
	return 0;
}

__FLAME_GPU_FUNC__ int output_secretory_cell_location(xmachine_memory_SecretoryCell* agent, xmachine_message_secretory_cell_location_list* location_messages) {
	add_secretory_cell_location_message(location_messages, agent->id, agent->x, agent->y, 0,
		agent->p1_x, agent->p1_y, agent->p2_x, agent->p2_y, agent->direction_x, agent->direction_y,
		agent->direction_x_unit, agent->direction_y_unit, agent->direction_length, 
		agent->normal_x, agent->normal_y, agent->unit_normal_x, agent->unit_normal_y, agent->normal_length
	);
	return 0;
}

__device__ float4 vectors_from_ev_to_wall_endpoints(xmachine_memory_EV* agent, const float & p1_x, const float & p1_y,
	const float & p2_x, const float & p2_y) {
	return make_float4(p1_x - agent->x, p1_y - agent->y, p2_x - agent->x, p2_y - agent->y);
}

__device__ float2 project_vectors_onto_wall(float4 ev_wall, float cell_direction_x, float cell_direction_y) {
	return make_float2(
		projection(ev_wall.x, ev_wall.y, cell_direction_x, cell_direction_y),
		projection(ev_wall.z, ev_wall.w, cell_direction_x, cell_direction_y));
}

__device__ float2 perpendicular_distance_from_ev_to_wall(float4 ev_wall, float proj1, float cell_direction_x_unit, float cell_direction_y_unit) {
	// add_scaled(ballp1, wdir.unit(), proj1*-1)
	return make_float2(
		ev_wall.x + cell_direction_x_unit * -proj1,
		ev_wall.y + cell_direction_y_unit * -proj1
	);
}

/**
	Computes the distance between a point (p0) and the segment of a line with end points (p1) and (p2).
	We get the vectors A = p0 - p1 and B = p2 - p1. 
	-param- is the scalar value that when multiplied by B gives you the point on the line closest to p0.
	A collision is possible if the point:
		- is projecting on the segment, 
		- is not yet proyecting on the segment but displacing towards the segment and within a shorter distance from the endpoints than its radius
	Modified from https://stackoverflow.com/a/6853926/3830240

	Returns
	float4 where:
		x - 1 if the point is projecting on the segment, 0 otherwise
		y - distance from the closest point on the segment to the EV position
		z - x coordinate of the closest point from the segment to the EV
		w - y coordinate of the closest point from the segment to the EV
*/
__device__ float4 point_to_line_segment_distance(float x, float y, float x1, float y1, float x2, float y2)
{
	float4 result;

	// vector components from point to one of the end points of the line segment
	float A = x - x1;
	float B = y - y1;
	// vector components of the line segment
	float C = x2 - x1;
	float D = y2 - y1;

	float dot = A * C + B * D; // vector projection
	float len_sq = C * C + D * D; // sqrd magnitude of the line segment
	float param = -1;
	float xx = 0.f, yy = 0.f, dx, dy;
	//float projecting_on_segment = 0.0f;

	if (len_sq != 0){ // in case of 0 length line
		param = dot / len_sq; // ratio of the projected vectors over the length of the line segment
	}
	
	if (param < 0){ // the closest point is p1, point not projecting on segment
		xx = x1;
		yy = y1;
	} 
	else if (param > 1){ // the closest point is p2, point not projecting on segment
		xx = x2;
		yy = y2;
	}
	else {   // the point is projecting on the segment
		// interpolate to find the closes point on the line segment
		//projecting_on_segment = 1.0f;
		xx = x1 + param * C;
		yy = y1 + param * D;
	}

	// dx / dy is the vector from p0 to the closest point on the segment
	dx = x - xx;
	dy = y - yy;

	result.x = param;
	result.y = sqrtf(dx * dx + dy * dy); // distance between p
	result.z = xx;
	result.w = yy;
	
	return result;
}

/**
	Checks any possible collision with the secretory cells on the boundaries. 
	If a collision occurred, a collision_message is generated and the agent will have the values:
		agent->closest_secretory_cell_id = closest_cell_id;
		agent->closest_secretory_cell_distance = closest_cell_distance;
	Otherwise, no message is produced and the ev agent will have the default values unchanged:
		agent->closest_secretory_cell_id = -1;
		agent->closest_secretory_cell_distance = -1.0f;
*/
__FLAME_GPU_FUNC__ int test_secretory_cell_collision(xmachine_memory_EV* agent, 
xmachine_message_secretory_cell_location_list* location_messages,
	xmachine_message_secretory_cell_location_PBM* partition_matrix)
{
	float4 res;
	xmachine_message_secretory_cell_location* message = get_first_secretory_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);
	agent->closest_secretory_cell_distance = agent->radius_um;
	while (message) {
		// we only check for collisions if the EV: -is displacing in opposite or perpendicular direction to the wall
		//float dotProduct =;

		// the dot product is:
		//	<0 for vectors in opposing directions
		//  =0 for those perpendicular to each other
		// If unormal and direction are perpendicular to each other
		// the displacement may be parallel or collinear to the wall
		if (dotprod(agent->vx, agent->vy, message->unit_normal_x, message->unit_normal_y) <= 0) {
			// for each agent's new position and a cell compute:
			// x - ratio of the segment where the position is projecting or 0,1 if outside
			// y - distance between position and closest point on the segment
			// z,w - coordinates where the position is projecting on the segment
			res = point_to_line_segment_distance(agent->x, agent->y,
				message->p1_x, message->p1_y, message->p2_x, message->p2_y);

			// if the ev new position is projecting over the segment
			if (res.y < agent->closest_secretory_cell_distance)
			{
				agent->closest_secretory_cell_id = message->id;
				agent->closest_secretory_cell_distance = res.y;
			}
		}

		// get the next message
		message = get_next_secretory_cell_location_message(message, location_messages, partition_matrix);
	}
	return 0;
}

__FLAME_GPU_FUNC__ int test_ciliary_cell_collision(xmachine_memory_EV* agent, 
xmachine_message_ciliary_cell_location_list* location_messages,
	xmachine_message_ciliary_cell_location_PBM* partition_matrix)
{
	float4 res;
	xmachine_message_ciliary_cell_location* message = get_first_ciliary_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);
	agent->closest_ciliary_cell_distance = agent->radius_um;
	while (message) {
		// we only check for collisions if the EV: -is displacing in opposite or perpendicular direction to the wall
		// the dot product will be negative for vectors in opposite directions and zero for those perpendicular to each other
		if (dotprod(agent->vx, agent->vy, message->unit_normal_x, message->unit_normal_y) <= 0) {
			// get the distance between agent and cell
			res = point_to_line_segment_distance(agent->x, agent->y,
				message->p1_x, message->p1_y, message->p2_x, message->p2_y);

				// save the reference if this is the first candidate collision
				// or if the agent is closer to the wall than the previous collision
				if (res.y < agent->closest_ciliary_cell_distance)
				{
					agent->closest_ciliary_cell_id = message->id;
					agent->closest_ciliary_cell_distance = res.y;
				}
		}
		message = get_next_ciliary_cell_location_message(message, location_messages, partition_matrix);
	}
	return 0;
}

/**
	Checks any possible collision with another EV
	If a collision occurred, a collision_message is generated and the agent will have the values:
		agent->closest_ev_id = closest_ev_id;
		agent->closest_ev_distance = closest_ev_distance;
	Otherwise, no message is produced and the ev agent will have the default values unchanged:
		agent->closest_ev_id = -1;
		agent->closest_ev_distance = -1.0f;
*/	
__FLAME_GPU_FUNC__ int test_collision_ev_default_ev_default(xmachine_memory_EV* agent,
	xmachine_message_location_ev_default_list* location_messages, 
	xmachine_message_location_ev_default_PBM* partition_matrix
	){
	
	float distance, overlap, max_overlap = 0;

	xmachine_message_location_ev_default* message = get_first_location_ev_default_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message)	{
		if (message->id != agent->id){
			// check for collision
			distance = euclidean_distance(agent, message->x, message->y);
			// smallest separation before a collision occurs
			overlap = (agent->radius_um + message->radius_um) - distance;
			//if (overlap > 1e-4){
				// the closer both EV locations are, the larger the overlap is
				// these 2 can become only 1 comparison? overlap > max_overlap
				if (//agent->closest_ev_default_id == 0 || 
					overlap > max_overlap){
						agent->closest_ev_default_id = message->id;
						agent->closest_ev_default_distance = distance;
						max_overlap = overlap;
				}
			//}
		}
		message = get_next_location_ev_default_message(message, location_messages, partition_matrix);
	}
	return 0;
}

__FLAME_GPU_FUNC__ int test_collision_ev_default_ev_biogenesis(xmachine_memory_EV* agent, 
	xmachine_message_location_ev_biogenesis_list* location_messages, 
	xmachine_message_location_ev_biogenesis_PBM* partition_matrix){
	
	float distance;
	float max_overlap = 0, overlap;

	//float4 new_values;

	xmachine_message_location_ev_biogenesis* message = get_first_location_ev_biogenesis_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	// This check could be improved by identifying and solving only the first collision involving this agent.
	while (message)	{
		if (message->id != agent->id){
			// check for collision
			distance = euclidean_distance(agent, message->x, message->y);
			overlap = (agent->radius_um + message->radius_um) - distance;
			if (overlap > max_overlap){
				agent->closest_ev_biogenesis_id = message->id;
				agent->closest_ev_biogenesis_distance = distance;
				max_overlap = overlap;
			}
		}
		message = get_next_location_ev_biogenesis_message(message, location_messages, partition_matrix);
	}
	return 0;
}
// tests and solves in one go
__FLAME_GPU_FUNC__ int test_collision_ev_biogenesis_ev_default(xmachine_memory_EV* agent, 
	xmachine_message_location_ev_default_list* location_messages, 
	xmachine_message_location_ev_default_PBM* partition_matrix){
	
	float distance;
	float max_overlap = 0, overlap;

	xmachine_message_location_ev_default* message = get_first_location_ev_default_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message)	{
		if (message->id != agent->id){
			// check for collision
			distance = euclidean_distance(agent, message->x, message->y);
			
			overlap = (agent->radius_um + message->radius_um) - distance;
			if (overlap > max_overlap){
				agent->closest_ev_default_id = message->id;
				agent->closest_ev_default_distance = distance;
				max_overlap = overlap;
				agent->last_ev_default_collision = iteration;

				agent->precol.x = agent->x;
				agent->precol.y = agent->y;
				agent->precol_v.x = agent->vx;
				agent->precol_v.y = agent->vy;
				
				agent->x = agent->x_1;
				agent->y = agent->y_1;
			}
		}
		message = get_next_location_ev_default_message(message, location_messages, partition_matrix);
	}
	if(agent->closest_ev_default_id != 0){
		agent->time_in_biogenesis_state += dt;
	}
	return 0;
}
// solves direct collisions between EV and segments
__device__  int solve_segment_collision(xmachine_memory_EV* agent, float cell_dir_x, 
	float cell_dir_y, float cell_direction_length, float perp_dist_x, float perp_dist_y,
	float cell_unit_normal_x, float cell_unit_normal_y) {
	
	// We compute the angle between velocity and wall direction based on
	// the cosine of the angle betwen two vectors which is the quotient of the
	// dot product of the vectors and the product of their magnitudes
	// 1. angle between velocity and wall direction
	float dp = dotprod(agent->vx, agent->vy, cell_dir_x, cell_dir_y);
	float lengths_product = (vlength(agent->vx, agent->vy) * cell_direction_length);
	float quotient = dp / lengths_product;
	
	// Due to rounding errors, the quotient can have values out of the
	// expected range for acosf() which produces NaN values
	if(quotient > 1.0) { quotient = 1.0; }
	else if (quotient < -1.0) { quotient = -1.0; }
	float angleBetween = acosf(quotient);

	// 2. reposition object
	// 2.1. start by computing deltaS
	float dist_dp_normal = dotprod(perp_dist_x, perp_dist_y, cell_unit_normal_x, cell_unit_normal_y);
	float sin_angle = sinf(angleBetween);
	float deltaS = 0;
	agent->dbgSegC.x = angleBetween;
	agent->dbgSegC.y = sin_angle;
	agent->dbgSegC.z = dist_dp_normal;

	// deltaS matches the value of 'closest_secretory_cell_distance'
	if (abs(sin_angle) < 1e-6) {
		// special case, orthogonal collisions produce angle=0, sin_angle=0
		// the distance to compensate affects only a single axis
		deltaS = agent->radius_um - abs(dist_dp_normal);
	} else {
		// *0.001 corrects the compensation required
		deltaS = ((agent->radius_um - abs(dist_dp_normal)) / sin_angle) * 0.001;
	}
	agent->dbgSegC.w = deltaS; // same value as the magnitude of displ?

	// 2.2. estimate the displacement vector needed for the correction
	float2 displ = parallel(agent->vx, agent->vy, deltaS);
	// distance, 
	agent->dbgSegCdispl.x = vlength(perp_dist_x, perp_dist_y);
	agent->dbgSegCdispl.y = vlength(displ.x, displ.y);// <- same value as deltaS?
	agent->dbgSegCdispl.z = displ.x;
	agent->dbgSegCdispl.w = displ.y;
	// 2.3. update position by subtracting the displacement
	agent->x -= displ.x;
	agent->y -= displ.y;

	// 6. decompose the velocity
	// velocity vector component perpendicular to wall just before impact
	float velocityProjection = projection(agent->vx, agent->vy, perp_dist_x, perp_dist_y);
	float2 normalVelocity = parallel(perp_dist_x, perp_dist_y, velocityProjection);
	
	// velocity vector component parallel to wall; unchanged by impact
	float tangentVelocity_x = agent->vx - normalVelocity.x;
	float tangentVelocity_y = agent->vy - normalVelocity.y;
	
	// velocity vector component perpendicular to wall just after impact
	float new_vx = tangentVelocity_x + (-normalVelocity.x);
	float new_vy = tangentVelocity_y + (-normalVelocity.y);
	if (isfinite(new_vx) && isfinite(new_vy)){
		agent->vx = new_vx;
		agent->vy = new_vy;
	} 
	return 0;
}

__device__  int solve_segment_end_point_collision(xmachine_memory_EV* agent, 
	float wall_pt_x, float wall_pt_y
) {
	// bounce off endpoint wall_p
	float distp_x = agent->x - wall_pt_x;
	float distp_y = agent->y - wall_pt_y;

	// move particle so that it just touches the endpoint			
	float L = agent->radius_um - vlength(distp_x, distp_y);
	float vrel = vlength(agent->vx, agent->vy);
	agent->x += agent->vx * (-L / vrel);
	agent->y += agent->vy * (-L / vrel);

	// normal velocity vector just before the impact
	float2 normalVelo = project(agent->vx, agent->vy, distp_x, distp_y);
	// tangential velocity vector
	float tangentVelo_x = agent->vx - normalVelo.x;
	float tangentVelo_y = agent->vy - normalVelo.y;
	// normal velocity vector after collision

	//// final velocity vector after collision
	agent->vx = -normalVelo.x + tangentVelo_x;
	agent->vy = -normalVelo.y + tangentVelo_y;
	return 0;
}

__device__ int solve_cell_collision(xmachine_memory_EV* agent, float p1_x, float p1_y, float p2_x, float p2_y, 
	float direction_x, float direction_y, float direction_x_unit, float direction_y_unit, float direction_length,
	float unit_normal_x, float unit_normal_y, float caller)
	{
		float4 ev_wall_vectors = vectors_from_ev_to_wall_endpoints(agent, p1_x, p1_y, p2_x, p2_y);
		float2 projections = project_vectors_onto_wall(ev_wall_vectors, direction_x, direction_y);
		float2 perp_dist = perpendicular_distance_from_ev_to_wall(ev_wall_vectors, projections.x, direction_x_unit, direction_y_unit);

		bool test_needed = ((abs(projections.x) < direction_length)
				&& (abs(projections.y) < direction_length));
		if (((perp_dist.x * perp_dist.x + perp_dist.y * perp_dist.y) < agent->radius_um_sq) && test_needed) {
			agent->precol.x = agent->x;
			agent->precol.y = agent->y;
			agent->precol_v.x = agent->vx;
			agent->precol_v.y = agent->vy;
			solve_segment_collision(agent, direction_x,	direction_y, direction_length,
				perp_dist.x, perp_dist.y, unit_normal_x, unit_normal_y);
		}
		else if (abs(ev_wall_vectors.x * ev_wall_vectors.x + ev_wall_vectors.y * ev_wall_vectors.y) < agent->radius_um_sq) {
			// collision with 1st end point
			agent->precol.x = agent->x;
			agent->precol.y = agent->y;
			agent->precol_v.x = agent->vx;
			agent->precol_v.y = agent->vy;
			solve_segment_end_point_collision(agent, p1_x, p1_y);
		}
		else if (abs(ev_wall_vectors.z * ev_wall_vectors.z + ev_wall_vectors.w * ev_wall_vectors.w) < agent->radius_um_sq) {
			// collision with 2nd end point
			agent->precol.x = agent->x;
			agent->precol.y = agent->y;
			agent->precol_v.x = agent->vx;
			agent->precol_v.y = agent->vy;
			solve_segment_end_point_collision(agent, p2_x, p2_y);
		}
		return 0;
}

/**
	Collision resolution algorithm modified from Physics for Javascript Games, Animation, and Simulation Ch, 11
	ID 3000
*/
__FLAME_GPU_FUNC__ int secretory_cell_collision_resolution(xmachine_memory_EV* agent,
	xmachine_message_secretory_cell_location_list* location_messages,
	xmachine_message_secretory_cell_location_PBM* partition_matrix)
{
	xmachine_message_secretory_cell_location* message = get_first_secretory_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message){
		// we verify the ev_id in the message matches this agent's id
		if (message->id == agent->closest_secretory_cell_id) {
			solve_cell_collision(agent, message->p1_x, message->p1_y, message->p2_x, message->p2_y,
				message->direction_x, message->direction_y, message->direction_x_unit, message->direction_y_unit,
				message->direction_length, message->unit_normal_x, message->unit_normal_y, 3000);
		}
		message = get_next_secretory_cell_location_message(message, location_messages, partition_matrix);
	}
	return 0;
}

/**
	This function resolves a single collision between an EV and a ciliary cell
	After this, the EV has new values for x, y, vx, and vy, compensated for the collision
	ID 4000
*/
__FLAME_GPU_FUNC__ int ciliary_cell_collision_resolution(xmachine_memory_EV* agent, 
	xmachine_message_ciliary_cell_location_list* location_messages,
	xmachine_message_ciliary_cell_location_PBM* partition_matrix)
{
	xmachine_message_ciliary_cell_location* message = get_first_ciliary_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);
	while (message) {
		// we verify the ev_id in the message matches this agent's id
		if (message->id == agent->closest_ciliary_cell_id) {
			solve_cell_collision(agent, message->p1_x, message->p1_y, message->p2_x, message->p2_y,
				message->direction_x, message->direction_y, message->direction_x_unit, message->direction_y_unit,
				message->direction_length, message->unit_normal_x, message->unit_normal_y, 4000);
		}
		message = get_next_ciliary_cell_location_message(message, location_messages, partition_matrix);
	}
	return 0;
}
__FLAME_GPU_FUNC__ int collision_default_default(xmachine_memory_EV* agent){
	agent->last_ev_default_collision = iteration;
	return 0;
}
__FLAME_GPU_FUNC__ int collision_default_biogenesis(xmachine_memory_EV* agent){
	agent->last_ev_biogenesis_collision = iteration;
	return 0;
}
__FLAME_GPU_FUNC__ int collision_secretory_cell(xmachine_memory_EV* agent){
	agent->last_cell_collision = iteration;
	return 0;
}
__FLAME_GPU_FUNC__ int collision_ciliary_cell(xmachine_memory_EV* agent){
	agent->last_cell_collision = iteration;
	return 0;
}

// must be executed by one of the involved EVs only
__FLAME_GPU_FUNC__ int collision_solver_ev_default_ev_default(xmachine_memory_EV* agent,
	xmachine_message_location_ev_default_list* location_messages,
	xmachine_message_location_ev_default_PBM* partition_matrix, 
	xmachine_message_resolved_collision_ev_default_list* resolved_collision_messages
	)
{
	xmachine_message_location_ev_default* message = get_first_location_ev_default_message(
		location_messages, partition_matrix, agent->x, agent->y, agent->z);
	while(message){
		if(agent->closest_ev_default_id == message->id){
			// solve the collision for both agents
			float2 ev1_velo = make_float2(agent->vx, agent->vy);
			float2 ev2_velo = make_float2(message->vx, message->vy);
			float2 distance = make_float2(agent->x - message->x, agent->y - message->y);
			
			// normal velocity vectors just before the impact
			float2 normal_velocity1 = float2_project(ev1_velo, distance);
			float2 normal_velocity2 = float2_project(ev2_velo, distance);
			// tangential velocity vector
			float2 tangent_velocity1 = float2_sub(ev1_velo, normal_velocity1);
			float2 tangent_velocity2 = float2_sub(ev2_velo, normal_velocity2);

			// move particles so that they just touch
			// If both normal components match, the value of pcd.vrel would be zero
			float2 normal_velo_subtracted = float2_sub(normal_velocity1, normal_velocity2);
			float vrel = vlength(normal_velo_subtracted.x, normal_velo_subtracted.y);
			if (vrel == 0 || vrel < 1e-8) {
				vrel = vlength(ev1_velo.x, ev1_velo.y);	
			}
			// correctionFactor = overlap / vrel
			float correctionFactor = (agent->radius_um + message->radius_um) - vlength(distance.x, distance.y) / vrel;
			float2 ev1_new_loc = add_scaled(make_float2(agent->x, agent->y), normal_velocity1, -correctionFactor);
			float2 ev2_new_loc = add_scaled(make_float2(message->x, message->y), normal_velocity2, -correctionFactor);			
			// normal velocity components after the impact
			float u1 = projection(normal_velocity1.x, normal_velocity1.y, distance.x, distance.y);
			float u2 = projection(normal_velocity2.x, normal_velocity2.y, distance.x, distance.y);
			float v1 = ((agent->mass_ag - message->mass_ag) * u1 + 2 * message->mass_ag * u2) / (agent->mass_ag + message->mass_ag);
			float v2 = ((message->mass_ag - agent->mass_ag) * u2 + 2 * agent->mass_ag * u1) / (agent->mass_ag + message->mass_ag);

			normal_velocity1 = parallel(distance.x, distance.y, v1);
			normal_velocity2 = parallel(distance.x, distance.y, v2);
			float2 ev1_new_v = float2_add(normal_velocity1, tangent_velocity1);
			float2 ev2_new_v = float2_add(normal_velocity2, tangent_velocity2);
			
			// update the current agent
			agent->precol.x = agent->x;
			agent->precol.y = agent->y;
			agent->precol_v.x = agent->vx;
			agent->precol_v.y = agent->vy;
			agent->x = ev1_new_loc.x;
			agent->y = ev1_new_loc.y;
			agent->vx = ev1_new_v.x;
			agent->vy = ev1_new_v.y;

			// create a message with the updated values for the other involved agent
			add_resolved_collision_ev_default_message(resolved_collision_messages,
				agent->id, message->x, message->y, message->z, 
				message->id, ev2_new_loc.x, ev2_new_loc.y, ev2_new_v.x, ev2_new_v.y
			);
		}
		message = get_next_location_ev_default_message(message, location_messages, partition_matrix);
	}
	return 0;
}
// executed by the second EV involved in the previous collision
__FLAME_GPU_FUNC__ int post_collision_update_ev_default_ev_default(xmachine_memory_EV* agent,
	xmachine_message_resolved_collision_ev_default_list* resolved_collision_messages,
	xmachine_message_resolved_collision_ev_default_PBM* partition_matrix
	)
{
		xmachine_message_resolved_collision_ev_default* message = get_first_resolved_collision_ev_default_message(
			resolved_collision_messages, partition_matrix, agent->x, agent->y, agent->z);
		while(message){
			if(agent->id == message->ev2_id){
				agent->precol.x = agent->x;
				agent->precol.y = agent->y;
				agent->precol_v.x = agent->vx;
				agent->precol_v.y = agent->vy;

				agent->x = message->ev2_new_x;
				agent->y = message->ev2_new_y;
				agent->vx = message->ev2_new_vx;
				agent->vy = message->ev2_new_vy;
			}
			message = get_next_resolved_collision_ev_default_message(
				message, resolved_collision_messages, partition_matrix);
		}
		return 0;
}

__FLAME_GPU_FUNC__ int collision_solver_ev_default_ev_biogenesis(xmachine_memory_EV* agent,
	xmachine_message_location_ev_biogenesis_list* location_messages, 
	xmachine_message_location_ev_biogenesis_PBM* partition_matrix){

	xmachine_message_location_ev_biogenesis* message = get_first_location_ev_biogenesis_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);
	while(message){
		// The EV in default state must be relocated so that it just touches the EV in biogenesis state
		// To do so, we correct it's position by the overlap distance
		// EV_default's position is corrected using it's velocity before collision
		if(agent->closest_ev_biogenesis_id == message->id){
			agent->precol.x = agent->x;
			agent->precol.y = agent->y;
			agent->precol_v.x = agent->vx;
			agent->precol_v.y = agent->vy;
			float2 new_values;
			
			float2 agent_vel = make_float2(agent->vx, agent->vy);
			float2 dist = make_float2(message->x - agent->x, message->y - agent->y);
			// normal velocity vectors just before the impact
			float2 normal_velocity1 = float2_project(agent_vel, dist);
			float2 normal_velocity2 = float2_project(make_float2(message->vx, message->vy), dist);
			// tangential velocity vector, not affected by the collision resolution
			float2 tangent_velocity1 = float2_sub(agent_vel, normal_velocity1);

			// overlap / original velocity before collision
			agent->dbgSegCdispl.x = vlength(dist.x, dist.y);
			agent->dbgSegC.z = vlength(agent->vx, agent->vy);
			// if the overlap is larger than the agent's radius it goes back to the previous location
			agent->dbgSegC.w = (agent->radius_um + message->radius_um) - agent->dbgSegCdispl.x;
			if(agent->dbgSegC.w < agent->radius_um){
				float correctionFactor = agent->dbgSegC.w / agent->dbgSegC.z;
				// If the EVs displace in opposite direction, the correction must subtract from the displacement
				if(dotprod(agent->vx, agent->vy, message->vx,  message->vy) < 0){
					correctionFactor = -correctionFactor;
				}
				agent->dbgSegCdispl.y = correctionFactor;
				//pcdi.correctionApplied = float2_scale(normal_velocity1, correctionFactor);
				new_values = add_scaled(make_float2(agent->x, agent->y), normal_velocity1, correctionFactor);
				agent->dbgSegCdispl.z = normal_velocity1.x * correctionFactor;
				agent->dbgSegCdispl.w = normal_velocity1.y * correctionFactor;
			} else {
				agent->x = agent->x_1;
				agent->y = agent->y_1;
			}
			// normal velocity components after the impact
			float u1 = projection(normal_velocity1.x, normal_velocity1.y, dist.x, dist.y);
			float u2 = projection(normal_velocity2.x, normal_velocity2.y, dist.x, dist.y);
			float v1 = (((agent->mass_ag - message->mass_ag) * u1 + 2 * message->mass_ag * u2)
						/ (agent->mass_ag + message->mass_ag));
			normal_velocity1 = parallel(dist.x, dist.y, v1);
			new_values = float2_add(normal_velocity1, tangent_velocity1);
			agent->vx = new_values.x;
			agent->vy = new_values.y;
		}
		message = get_next_location_ev_biogenesis_message(message, location_messages, partition_matrix);
	}
	return 0;
}

__FLAME_GPU_FUNC__ int collision_solver_ev_biogenesis_ev_default(xmachine_memory_EV* agent){
	
	return 0;
}

__FLAME_GPU_FUNC__ int secrete_ev(xmachine_memory_SecretoryCell* secretoryCell, xmachine_memory_EV_list* EVs, RNG_rand48* rand48){
	// compute the next interval of secretion
	if(secretoryCell->time_to_next_secretion_attempt < 0)
	{
		// compute the potential time to next secretion attempt
		secretoryCell->time_to_next_secretion_attempt = (ev_secretion_interval/2 + 
				(rnd<CONTINUOUS>(rand48) * ev_secretion_interval)) + dt;

		// value saved in the secretoryCell for debugging
		secretoryCell->probability_of_secretion = rnd<CONTINUOUS>(rand48);
				
		if( secretoryCell->probability_of_secretion > ev_secretion_threshold)
		{
			int rand_i = (int)(rnd<CONTINUOUS>(rand48) * secretoryCell->max_ev_radius);
			// our simulations work with EVs with diameters in the range 80-320 nm
			// therefore, the radius must be in the range 40-160 nm
			float radius_nm = (rand_i % secretoryCell->max_ev_radius) + secretoryCell->min_ev_radius;
			float radius_um = (radius_nm) / 1000; // faster than doing /1000

			// compute the volume
			//float volume = const_pi_4div3 * radius_nm_3;
			//float mass_g = const_mass_per_volume_unit * volume;
			float mass_ag = (const_mass_p_vol_u_x_4div3_pi * radius_nm * radius_nm * radius_nm) / 1e-18;

			// to convert:
			// N square metres to square micrometre: multiply N * 1e+12
			// N metres to micrometres: multiply N * 1e+6
			// Here D is in squared meters/second
			float D_ms = const_Boltzmann_x_Temp_K / (const_6_pi_dynamic_viscosity * (radius_um / 1E6));
			// A more usual units for D is cm^2/s^-1, however, we use um^2 / sec
			float diffusion_rate_ums = D_ms * 1e+12; // square micrometre
			// Mean Squared Displacement (MSD) x^2 = 2 * dof * D * t
			// mean displacement distance (MDD) = sqrt(MSD)
			float MDD_01s = sqrtf(2 * dof * diffusion_rate_ums * bm_frequency);
			// MDD is the mean distance a particle would diffuse in time t.
			// After experimentation, we found that sampling from a normal distribution
			// with parameters: mean = 0 and std = MDD@0.1sec / 1.25, produces the
			// expected mean displacement for the corresponding diffusion rate
			float mdd125 = MDD_01s * 0.8;
			float bm_rx = sqrtf(-2*log(rnd<CONTINUOUS>(rand48)));
			float bm_ry =  bm_rx * mdd125 * bm_factor;
			// decompose velocity
			float vx = bm_ry * secretoryCell->unit_normal_x;
			float vy = bm_ry * secretoryCell->unit_normal_y;

			// choose a random starting point
			int rand_i2 = (int)(rnd<CONTINUOUS>(rand48) * secretoryCell->source_points);
			// if the last source point secreting an Ev is selected again and the cell has
			// more than 1 source point, we select another one manually
			if (rand_i2 == secretoryCell->last_source_point_secreting && secretoryCell->source_points > 1) {
				if(rand_i2+1 < secretoryCell->source_points){
					rand_i2++;
				}
				else {
					rand_i2--;
				}
			}

			secretoryCell->last_source_point_secreting = rand_i2;
			int ssp = rand_i2 * xmachine_memory_SecretoryCell_MAX;
			float x = secretoryCell->source_points_xs[ssp];
			float y = secretoryCell->source_points_ys[ssp];
			// last secreted ev counter * 1e6 + agent->id;
			secretoryCell->last_ev_id++;
			unsigned int id = secretoryCell->last_ev_id * 1e6 + ((unsigned) secretoryCell->id);
			// displace the startgint coordinates backwards by the diameter of the EV
			x -= secretoryCell->unit_normal_x * radius_um;
			y -= secretoryCell->unit_normal_y * radius_um;

			float time_in_biogenesis = ((radius_um * 2.) / bm_ry) * dt;

			// if the cell has only 1 source point, it cannot secrete another EV until
			// the previous has been fully released, we make sure of that
			if(secretoryCell->source_points < 2 && secretoryCell->time_to_next_secretion_attempt <= time_in_biogenesis * 1.5){
				secretoryCell->time_to_next_secretion_attempt = time_in_biogenesis * 1.5;
			}

			// EV_secretoryCell_list, id, x, y, z, x_1, y_1, vx, vy, bm_vx, bm_vy, bm_r, last_bm,
			add_EV_agent(EVs, id, x, y, 0, x - vx, y - vy, vx, vy, 0, 0, fvec2(bm_rx, bm_ry), 0,
				// mass_ag, radius_um, radius_um^2
				mass_ag, radius_um, radius_um * radius_um, (radius_um * 1.5) * (radius_um * 1.5),
				// diffusion_rate_um, MDD_01s, mdd125
				diffusion_rate_ums, MDD_01s, mdd125,
				// closest: ev_d_id, d_dist, ev_i_id, i_dist,
				0, 100, 0, 100,
				// closest: secretory, dist, ciliary, dist, last_[ev_d|ev_b|cell]_collision, age
				-1, 100, -1, 100, 0, 0, 0, 0,
				// (initial) apoptosis timer, time_in_biogenesis_state
				time_in_biogenesis * 2, time_in_biogenesis,
				// pre-collision: x, y, vx, vy
				fvec2(0.0f, 0.0f), fvec2(0.0f, 0.0f),
				// dbgSegC
				fvec4(0,0,0,0),
				fvec4(0.0f, 0.0f, 0.0f, 0.0f),
				// defaultAt, disabledAt, disabledReason, displacementSq
				0, 0, 0, 0.0f
				);
		}
	}
	secretoryCell->time_to_next_secretion_attempt -= dt;
	return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
