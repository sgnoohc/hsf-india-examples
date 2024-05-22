#!/bin/env python

import LSTMath as m
import random
import plotly
import plotly.graph_objs as go
import plotly.express as px
import math
import numpy as np

# Get list of hits
hit_x = []
hit_y = []
hit_z = []
f = open("hits.csv")
lines = [ x.strip() for x in f.readlines()[1:] ]
for line in lines:
    ls = line.split(",")
    hit_x.append(float(ls[3]))
    hit_y.append(float(ls[4]))
    hit_z.append(float(ls[5]))

# Create Plotly scatter object
hits_to_draw = go.Scatter3d(
    x = hit_z,
    y = hit_x,
    z = hit_y,
    mode='markers',
    marker=dict(
        symbol='circle',
        size=1.5,
        color='gray',
        colorscale='Viridis',
        opacity=0.5,
        ),
    hoverinfo='none',
    name='lower hits',
    )

# Get list of mds
md_x = []
md_y = []
md_z = []
f = open("mds.csv")
lines = [ x.strip() for x in f.readlines()[1:] ]
for line in lines:
    ls = line.split(",")
    md_x.append(float(ls[0]))
    md_y.append(float(ls[1]))
    md_z.append(float(ls[2]))

# Create Plotly scatter object
mds_to_draw = go.Scatter3d(
    x = md_z,
    y = md_x,
    z = md_y,
    mode='markers',
    marker=dict(
        symbol='circle',
        size=1.5,
        color='cyan',
        colorscale='Viridis',
        opacity=0.5,
        ),
    hoverinfo='none',
    name='mds',
    )

# Get the list of points on each simulated tracks
# The list of points computed from pt,eta,phi,vx,vy,vz,q
# will be used to draw track lines
sim_track_x = []
sim_track_y = []
sim_track_z = []
g = open("tracks.csv")
lines = [ x.strip() for x in g.readlines()[1:] ]
for iline, line in enumerate(lines):
    ls = line.strip().split(",")
    # if the track has pt < 0.8 skip
    if float(ls[1]) < 1.5:
        continue
    # printing progress
    if iline % 100 == 0:
        print(iline)
    # points contain list of points to draw curved line
    points = m.get_helix_points(m.construct_helix_from_kinematics(float(ls[1]), float(ls[2]), float(ls[3]), float(ls[4]), float(ls[5]), float(ls[6]), float(ls[7])))
    for x, y, z in zip(points[0], points[1], points[2]):
        if abs(z) < 120:
            sim_track_x.append(x)
            sim_track_y.append(y)
            sim_track_z.append(z)
    # By inserting "None" points in between plotly
    # divides the line between each different tracks
    sim_track_x.append(None)
    sim_track_y.append(None)
    sim_track_z.append(None)

# Create plotly object
sims_to_draw = go.Scatter3d(
        x = sim_track_z,
        y = sim_track_x,
        z = sim_track_y,
        mode='lines',
        line=dict(
            color='red',
            width=2,
        ),
        opacity=0.5,
        hoverinfo='none',
        name='Sim Track',
)

# draw the figure
# fig = go.Figure([hits_to_draw, sims_to_draw])
fig = go.Figure([hits_to_draw, sims_to_draw, mds_to_draw])

# Set the window settings
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgb(0,0,0,1)",
    scene = dict(
        xaxis = dict(nticks=10, range=[-300,300],),
        yaxis = dict(nticks=10, range=[-200,200],),
        zaxis = dict(nticks=10, range=[-200,200],),
        aspectratio=dict(x=1, y=0.666, z=0.666),
    ),
    width=1400,
    height=800,
    margin=dict(r=20, l=10, b=10, t=10));

# Write to file
fig.write_html("event.html")
