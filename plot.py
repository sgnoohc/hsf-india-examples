

import LSTMath as m
import random
import plotly
import plotly.graph_objs as go
import plotly.express as px
import math
import numpy as np

f = open("hits.txt")

lines = [ x.strip() for x in f.readlines() ]

lower_hit_x = []
lower_hit_y = []
lower_hit_z = []
upper_hit_x = []
upper_hit_y = []
upper_hit_z = []

for line in lines:
    ls = line.split(",")
    if int(ls[1]) == 0:
        lower_hit_x.append(float(ls[4]))
        lower_hit_y.append(float(ls[5]))
        lower_hit_z.append(float(ls[6]))
    else:
        upper_hit_x.append(float(ls[4]))
        upper_hit_y.append(float(ls[5]))
        upper_hit_z.append(float(ls[6]))


lh_to_draw = go.Scatter3d(
    x = lower_hit_z,
    y = lower_hit_x,
    z = lower_hit_y,
    mode='markers',
    marker=dict(
        symbol='circle',
        size=1.5,
        color='cyan',
        colorscale='Viridis',
        opacity=0.5,
        ),
    hoverinfo='none',
    name='lower hits',
    )

rh_to_draw = go.Scatter3d(
    x = upper_hit_z,
    y = upper_hit_x,
    z = upper_hit_y,
    mode='markers',
    marker=dict(
        symbol='circle',
        size=1.5,
        color='orange',
        colorscale='Viridis',
        opacity=0.5,
        ),
    hoverinfo='none',
    name='upper hits',
    )

Xsim = []
Ysim = []
Zsim = []
Rsim = []

g = open("tracks.txt")
for iline, line in enumerate(g.readlines()):
    ls = line.strip().split(",")
    if float(ls[1]) < 0.8:
        continue
    if iline % 100 == 0:
        print(iline)
    points = m.get_helix_points(m.construct_helix_from_kinematics(float(ls[1]), float(ls[2]), float(ls[3]), float(ls[4]), float(ls[5]), float(ls[6]), float(ls[7])))
    for x, y, z in zip(points[0], points[1], points[2]):
        if abs(z) < 120:
            Xsim.append(x)
            Ysim.append(y)
            Zsim.append(z)
            Rsim.append(math.sqrt(x**2 + y**2))
    Xsim.append(None)
    Ysim.append(None)
    Zsim.append(None)
    Rsim.append(None)

sims_to_draw = go.Scatter3d(
        x = Zsim,
        y = Ysim,
        z = Xsim,
        mode='lines',
        line=dict(
            color='red',
            width=2,
        ),
        opacity=0.1,
        hoverinfo='none',
        name='Sim Track',
)

fig = go.Figure([lh_to_draw, rh_to_draw, sims_to_draw])

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

fig.write_html("htmls/hits.html")
