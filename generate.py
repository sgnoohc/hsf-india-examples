#!/bin/env python

import LSTMath as m
import random
import plotly
import plotly.graph_objs as go
import plotly.express as px
import math
import numpy as np
import matplotlib.pyplot as plt
import sys

generate_n_particles = 10000


#==============================================================================
# Below is to generate an exponentially falling distribution to sample pt from
#==============================================================================
def custom_pdf(x):
    # Define your custom PDF here
    # For example, a simple triangular distribution between 0 and 1
    return math.exp(-50*x)
def rejection_sampling(pdf, bounds, num_samples, max_pdf_value):
    samples = []
    while len(samples) < num_samples:
        # Generate a candidate sample from a uniform distribution within bounds
        x = np.random.uniform(bounds[0], bounds[1])
        # Generate a uniform random number between 0 and max_pdf_value
        u = np.random.uniform(0, max_pdf_value)
        # Accept the sample if u < pdf(x)
        if u < pdf(x):
            samples.append(x)
    return np.array(samples)
# Define parameters
bounds = (0, 2)
num_samples = generate_n_particles
max_pdf_value = 2
# Generate samples
samples = rejection_sampling(custom_pdf, bounds, num_samples, max_pdf_value)
#==============================================================================






#==============================================================================
# Generate a random track
#==============================================================================
def get_random_helix_pairs(pt):
    eta = random.uniform(-2.5, 2.5)
    phi = random.uniform(-math.pi, math.pi)
    vx = random.uniform(-0.5, 0.5)
    vy = random.uniform(-0.5, 0.5)
    vz = random.uniform(-0.5, 0.5)
    pos = m.construct_helix_from_kinematics(pt, eta, phi, vx, vy, vz, 1)
    neg = m.construct_helix_from_kinematics(pt, eta, phi, vx, vy, vz, -1)
    return [pos, neg, pt, eta, phi, vx, vy, vz]
#==============================================================================






#==============================================================================
# Basic detector geometry
#==============================================================================
# Radius of where the tracker lies
radii = [20, 35, 50, 70, 85, 110]
# Gaps of the doublet layer
gaps = [0.2, 0.2, 0.2, 0.4, 0.4, 0.4]
# Number of phi slices in each layer
# (computed by doing 2*pi*/(2*pi*R / 10 cm))
rads = [10./20, 10./40, 10./50, 10./70, 10./85, 10./110]
# Z slice width
zgap = 10 # slice in z by every 10 cm
#==============================================================================







#==============================================================================
# Output dataset
#==============================================================================
hits_data = open("hits.csv", "w")
hits_data.write(f"trkid,layer,det_id,x,y,z\n")
trk_data = open("tracks.csv", "w")
trk_data.write(f"trkid,pt,eta,phi,vx,vy,vz,q\n")
#==============================================================================






# Processing
for i in range(generate_n_particles):

    if i % 100 == 0:
        print(i)

    pos, neg, pt, eta, phi, vx, vy, vz = get_random_helix_pairs((samples[i])*20 + 0.2)

    if random.uniform(0, 1) > 0.5:
        chosen = pos
        trk_data.write(f"{i},{pt},{eta},{phi},{vx},{vy},{vz},1\n")
    else:
        chosen = neg
        trk_data.write(f"{i},{pt},{eta},{phi},{vx},{vy},{vz},-1\n")

    for ilayer, (ir, g) in enumerate(zip(radii, gaps)):

        # Get the cross section between track and the cylindrical track
        lower_hit = m.get_helix_point_from_radius(chosen, ir)
        upper_hit = m.get_helix_point_from_radius(chosen, ir+g)

        # Clean up bad hits (sometimes the calculation isn't great and gives weird points)
        if abs(lower_hit[3] - ir)/ir > 0.01:
            continue
        if abs(lower_hit[2]) > 120:
            continue

        # "detid" computation detail is here
        # Basically
        #    100 * (# of Z-slice from -z side) * phi_idx + zidx) + 10 * ilayer + isupper
        # Last digit will be whether it is upper or lower
        # Second to last digit will be which layer
        # The rest is z and phi slice information
        zidx = int((lower_hit[2]+120) / zgap)
        phidix = int((m.get_etaphi(lower_hit)[1]+math.pi) / rads[ilayer])
        detid_upper = int(phidix * (240 / zgap) + zidx) * 100  + (ilayer + 1) * 10 + 1
        detid_lower = int(phidix * (240 / zgap) + zidx) * 100  + (ilayer + 1) * 10 + 0

        hits_data.write(f"{i},{ilayer},{detid_upper},{upper_hit[0]},{upper_hit[1]},{upper_hit[2]}\n")
        hits_data.write(f"{i},{ilayer},{detid_lower},{lower_hit[0]},{lower_hit[1]},{lower_hit[2]}\n")

