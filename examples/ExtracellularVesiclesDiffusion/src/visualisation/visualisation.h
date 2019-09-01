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
 * on www.flamegpu.com website.
 * 
 */
#ifndef __VISUALISATION_H
#define __VISUALISATION_H

//#define SIMULATION_DELAY 1
#define PAUSE_ON_START 0

// constants
const unsigned int WINDOW_WIDTH = 1200;
const unsigned int WINDOW_HEIGHT = 900;

//frustrum
const double NEAR_CLIP = 0.005;
const double FAR_CLIP = 4000;

//Circle model fidelity
const int SPHERE_SLICES = 8;
const int SPHERE_STACKS = 8;
//const double SPHERE_RADIUS = 1;
#define SPHERE_RADIUS 1
#define SPHERE_RADIUS_OFFSET 0.25

//Viewing Distance
const double VIEW_DISTANCE = 50;

//light position
GLfloat LIGHT_POSITION[] = {50.f, 50.f, 50.f, 50.f};

#endif //__VISUALISATION_H
