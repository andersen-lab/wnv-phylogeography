import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
from io import StringIO as sio
import imp
import matplotlib.colorbar as cb
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import math
import re
from scipy import stats
plt.rcParams.update({'font.size': 24, "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 22})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
import fiona
from pyproj import Proj, transform
from matplotlib.patches import Polygon, Circle, PathPatch
from matplotlib.collections import PatchCollection, LineCollection
from shapely.geometry import shape
from matplotlib import animation
from itertools import product
from datetime import datetime as dt
import time
import random

bt = imp.load_source('baltic', './baltic/baltic.py')
regions = pd.read_csv("region_trait.tsv", sep="\t")
region_list = regions["location"].unique().tolist()
region_list = ['San-Diego', 'South-CA', 'Kern', 'South-Central-Valley', 'Sacramento-Yolo', 'North-Central-Valley']
county_regions = pd.read_csv("county_regions.txt", sep = "\t", names = ["county", "region", "fips"])

# Read trees
tree_dir="./mcc/"
tip_regex = "_([0-9-]+)"
tree_str = [sio(open(tree_dir+i).read()) for i in os.listdir(tree_dir)]
trees = [bt.loadNexus(i, tip_regex = tip_regex) for i in tree_str]

for i in trees:
    # Add full location to traits
    for o in i.Objects:
        m = max(o.traits["location.set.prob"])
        l = o.traits["location.set"][o.traits["location.set.prob"].index(m)]
        o.traits["location"] = l
    i.treeStats()

# Generate epochs and calculate transition times
# Function to calculate decimal date. Source: https://stackoverflow.com/questions/6451655/python-how-to-convert-datetime-dates-to-decimal-years
def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def get_quad_bezier_control(p1, p2, d):
    mid = [float(p1[0]+p2[0])/2, float(p1[1]+p2[1])/2]
    slope = float(p2[1] - p1[1])/(p2[0] - p1[0])
    theta = np.arctan(slope)
    p = [mid[0] + d * np.cos(np.pi/2 + theta), mid[1] + d * np.sin(np.pi/2 + theta)]
    return p

def quadBrezPoints(P0, P2, P1, nSamples):
    ans = np.zeros((nSamples+1,2))
    for i in range(nSamples):
        t = (i+0.0)/nSamples
        ans[i,0] = (1-t)**2 * P0[0] + 2*t*(1-t)*P1[0] + t**2 * P2[0]
        ans[i,1] = (1-t)**2 * P0[1] + 2*t*(1-t)*P1[1] + t**2 * P2[1]
    ans[nSamples] = P2
    return np.array([ans[:, 0], ans[:,1]])

def plot_trees_animate(ax, tree, y_offset, current_time):
    # region_ax = plt.subplot(gs[1])
    cmap = cm.Set1
    segments = []
    seg_colors = []
    points = {
        "x": [],
        "y": []
    }
    point_colors = []
    for node in tree.Objects:
        if current_time < node.absoluteTime:
            continue
        elif current_time >= node.absoluteTime and node.branchType == "leaf":
            points["x"].append(node.absoluteTime)
            points["y"].append(node.y + y_offset)
            c = cmap(region_list.index(node.traits["location"]))
            point_colors.append(c)
            continue
        for child in node.children:
            c = cmap(region_list.index(child.traits["location"]))
            current_x = current_time
            if current_time > child.absoluteTime:
                current_x = child.absoluteTime
            seg_colors.extend([c,c])
            segments.append([[node.absoluteTime, node.y+y_offset], [node.absoluteTime, child.y+y_offset]])
            segments.append([[node.absoluteTime, child.y+y_offset], [current_x, child.y+y_offset]])
    return segments, seg_colors, points, point_colors

def plot_trees(ax, tree, y_offset, bg=None):
    # region_ax = plt.subplot(gs[1])
    cmap = cm.Set1
    x_attr=lambda k: k.absoluteTime ## branch x position is determined by height
    b_func=lambda k: 1
    s_func=lambda k: 10
    su_func=lambda k: 20
    c_func= lambda k: cmap(region_list.index(k.traits["location"])) if bg == None else bg
    ct_func=lambda k: cmap(region_list.index(k.traits["location"])) if bg == None else bg
    cu_func=lambda k: 'k'
    z_func=lambda k: 100
    zu_func=lambda k: 99
    y_func = lambda k: k.y + y_offset
    tree.plotTree(ax,x_attr=x_attr,branchWidth=b_func,colour_function=c_func,y_attr = y_func) ## plot tree
    tree.plotPoints(ax,x_attr=x_attr,size_function=s_func,colour_function=ct_func,zorder_function=z_func, y_attr = y_func)

    for i in range(math.floor(tree.root.absoluteTime), 2019):
        ax.axvspan(i, i+0.5, zorder = 0, color="#ECECEC")

def decimalDateToMonthYear(decimalDate):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mIndex = int(np.floor((decimalDate%np.floor(decimalDate)) * 12))
    return months[mIndex]+" "+str(int(np.floor(decimalDate)))

in_projection = Proj(init = "epsg:4269")
out_projection = Proj(init = "epsg:3857")

fig = plt.figure(figsize=(13,7))
gs = gridspec.GridSpec(1,2, width_ratios = [1,1])
ax = plt.subplot(gs[0])
treeax = plt.subplot(gs[1])
# Draw California
ll_lng = -125.137244
ll_lat = 32.018864
ur_lat = 42.403590
ur_lng = -113.600355
ll = transform(in_projection, out_projection, ll_lng, ll_lat)
ur = transform(in_projection, out_projection, ur_lng, ur_lat)
ca_shp = fiona.open("./ca_regions_shp/ca_regions.shp")

poly_lw = []
poly_colors = []
poly_patches = []
poly_names = []
centroids = {}
for feat in ca_shp:
    centroids[feat["properties"]["REGION_COD"]] = shape(feat["geometry"]).centroid.xy
    if feat["geometry"]["type"] == "Polygon":
        coords = feat["geometry"]["coordinates"][0]
        coords = [transform(in_projection, out_projection, *i) for i in coords]
        poly_patches.append(Polygon(coords))
        r = feat["properties"]["REGION_COD"]
        poly_names.append(r)
        if r == "NONE":
            poly_colors.append("#707070")
            poly_lw.append(0)
        else:
            c = plt.cm.Set1(region_list.index(r))
            poly_colors.append(c)
            poly_lw.append(2)
    else:
        for k in feat["geometry"]["coordinates"]:
            coords = k[0]
            coords = [transform(in_projection, out_projection, *i) for i in coords]
            poly_patches.append(Polygon(coords))
            r = feat["properties"]["REGION_COD"]
            poly_names.append(r)
            if r == "NONE":
                poly_colors.append("#707070")
                poly_lw.append(0)
            else:
                c = plt.cm.Set1(region_list.index(r))
                poly_colors.append(c)
                poly_lw.append(2)

map_patch_collection = PatchCollection(poly_patches, facecolor=poly_colors, edgecolor="#000000", lw = 0, zorder = 2, alpha = 1)
ax.add_collection(map_patch_collection)
ax.set_ylim([ll[1], ur[1]])
ax.set_xlim([ll[0], ur[0]])
ax.axis("off")

lowest_tip = min([i.absoluteTime for j in trees for i in j.Objects])
highest_tip = max([i.absoluteTime for j in trees for i in j.Objects])

branch_collection = LineCollection([], zorder = 101)
treeax.add_collection(branch_collection)

_frames = 1500
_buffer = 10
time_intervals = np.linspace(lowest_tip, highest_tip, _frames+_buffer)
y_offset = 0
for ctr, i in enumerate(trees):
    plot_trees(treeax, i, y_offset, "#707070")
    y_offset += len(i.getExternal()) + 50

timeline, = treeax.plot([2003, 2003], [y_offset, y_offset], lw=1, color='#000000')

curve_collection =  [LineCollection([]) for tree in trees for i in range(len(tree.Objects))]
td = [None] * len(curve_collection)

time_text = ax.annotate(str(decimalDateToMonthYear(lowest_tip)), ((ll[0]+ur[0])/2, ll[1]), size=25, horizontalalignment='right', verticalalignment = "top")

for i in curve_collection:
    i.set_capstyle("round")
    i.set_zorder(4)
    ax.add_collection(i)

# Plot for points in front of curve
curve_points = ax.scatter([],[], s = 0, c= "#FFFFFF", zorder = 5, edgecolor='none')
tree_bg_points = treeax.scatter([], [], s = 0, c = "#000000", zorder = 102, edgecolor='none')
tree_points = treeax.scatter([], [], s = 0, c = "#707070", zorder = 103, edgecolor='none')


def animate(i):
    current_time = time_intervals[i]
    print(current_time)
    x = []
    y = []
    sizes = []
    pc = []
    for region in poly_names:
        if region == "NONE":
            pc.append("#707070")
            continue
    for k in curve_collection:
        k.set_segments([])
        k.set_linewidth(0)
    flist = []
    tlist = []
    ctr = 0
    for tree in trees:
        for node in tree.Objects:
            if node.absoluteTime > current_time or node.branchType == "leaf":
                ctr += 1
                continue
            for child in node.children:
                if child.absoluteTime < current_time or (node.traits["location"] == child.traits["location"]):
                    continue
                f = [i[0] for i in centroids[node.traits["location"]]]
                f = transform(in_projection, out_projection, *f)
                t = [i[0] for i in centroids[child.traits["location"]]]
                t = transform(in_projection, out_projection, *t)
                _td = 250000
                if child.traits["location"] in tlist and node.traits["location"] in flist:
                    _td = -250000
                if ctr >= len(td):
                    print(ctr)
                if td[ctr] == None:
                    td[ctr] = _td
                flist.append(node.traits["location"])
                tlist.append(child.traits["location"])
                parent_time = node.absoluteTime
                child_time = child.absoluteTime
                curve_time = child_time - parent_time - 50 * (time_intervals[1] - time_intervals[0])
                frac_curve_end = (current_time - parent_time)/curve_time
                frac_curve_start = (time_intervals[i-min(i, 50)] - parent_time)/curve_time
                points = quadBrezPoints(f, t, get_quad_bezier_control(f, t, td[ctr]), 100)
                points = points.T.reshape(-1, 1, 2)
                curve_start_index = np.floor(frac_curve_start * len(points)).astype(int)
                curve_end_index = np.floor(frac_curve_end * len(points)).astype(int)
                seg_points = points[curve_start_index: curve_end_index+1]
                segments = np.concatenate([seg_points[:-1], seg_points[1:]], axis=1)
                widths = np.logspace(np.log10(2),np.log10(6), len(points) - 1,endpoint=True)
                seg_widths = widths[curve_start_index: curve_end_index + 1]
                curve_collection[ctr].set_segments(segments)
                curve_collection[ctr].set_linewidth(seg_widths)
                curve_collection[ctr].set_color("#000000")
                curve_collection[ctr].set_zorder(4)
                if len(segments) >= 1:
                    x.append(segments[-1][-1][0])
                    y.append(segments[-1][-1][1])
                    sizes.append(seg_widths[-1])
            ctr += 1
    curve_points.set_offsets(np.dstack([x, y])[0])
    curve_points.set_sizes(sizes)
    curve_points.set_zorder(5)
    curve_points.set_color("#FFFFFF")
    timeline.set_data([current_time, current_time], [0, y_offset])
    patches = [curve_points]
    patches.extend(curve_collection)
    patches.append(timeline)
    branch_segments = []
    branch_colors = []
    animate_y_offset = 0
    x = []
    y = []
    point_colors = []
    for tree in trees:
        s,c,p, pc = plot_trees_animate(treeax, tree, animate_y_offset, current_time)
        branch_segments.extend(s)
        branch_colors.extend(c)
        x.extend(p["x"])
        y.extend(p["y"])
        point_colors.extend(pc)
        animate_y_offset += len(tree.getExternal()) + 50
    tree_bg_points.set_offsets(np.dstack([x, y])[0])
    tree_bg_points.set_sizes([20] * len(x))
    tree_bg_points.set_zorder(102)
    tree_bg_points.set_color("#000000")
    tree_points.set_offsets(np.dstack([x, y])[0])
    tree_points.set_sizes([10] * len(x))
    tree_points.set_zorder(103)
    tree_points.set_color(point_colors)
    patches.append(tree_points)
    patches.append(tree_bg_points)
    branch_collection.set_segments(branch_segments)
    branch_collection.set_color(branch_colors)
    branch_collection.set_linewidth(1)
    branch_collection.set_zorder(101)
    branch_collection.set_capstyle("round")
    time_text.set_text(decimalDateToMonthYear(current_time))
    patches.append(branch_collection)
    patches.append(time_text)
    return patches

treeax.get_yaxis().set_visible(False)
for i in ["top", "left", "right"]:
    ax.spines[i].set_visible(False)

anim = animation.FuncAnimation(plt.gcf(), animate, frames=_frames+_buffer, interval = 20, blit=True, repeat=False)
# plt.show()
# plt.close()
gs.tight_layout(fig)
treeax.set_ylim([-50,y_offset])
treeax.set_xlim([2003, 2018])

FFwriter = animation.FFMpegWriter(fps=45, bitrate=3500)
anim.save('./wnv_ca_phylogeography.mp4',writer=FFwriter, dpi = 300)
plt.close()
