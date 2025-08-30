import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import pandas as pd
import scipy

rightSections = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 
2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
leftSections = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

frame_interval = 5

for s in range(len(rightSections)):
    if s == 0 or s == len(rightSections) - 1:
        pass
    else:
        if rightSections[s - 1] == rightSections[s + 1] and rightSections[s] != rightSections[s - 1]:
            rightSections[s] = rightSections[s - 1]

for s in range(len(leftSections)):
    if s == 0 or s == len(leftSections) - 1:
        pass
    else:
        if leftSections[s - 1] == leftSections[s + 1] and leftSections[s] != leftSections[s - 1]:
            leftSections[s] = leftSections[s - 1]

plt.plot(rightSections)
plt.plot(leftSections)
plt.show()

time = np.arange(len(rightSections))

section_line = LineString(np.column_stack((time, rightSections)))

o_one_positive = np.column_stack((np.arange(0, len(rightSections) -1 ),np.full_like(np.arange(0, len(rightSections) -1 ), 1.05, dtype=np.float64)))
o_one_positive = LineString(o_one_positive)
o_one_negative = np.column_stack((np.arange(0, len(rightSections) -1 ),np.full_like(np.arange(0, len(rightSections) -1 ), 0.95, dtype=np.float64)))
o_one_negative = LineString(o_one_negative)

intersection_o_one_positive = section_line.intersection(o_one_positive)
intersection_o_one_negative = section_line.intersection(o_one_negative)

o_one_positive_intersections = []
if intersection_o_one_positive.geom_type == 'MultiPoint':
    #plt.plot(*LineString(intersection_left.geoms).xy, 'o')
    o_one_positive_intersections = sorted(((LineString(intersection_o_one_positive.geoms).xy)[0]).tolist())
elif intersection_o_one_positive.geom_type == 'Point':
    #plt.plot(*intersection_left.xy, 'o')
    o_one_positive_intersections = [intersection_o_one_positive.xy[0].tolist()[0]]

o_one_positive_intersections = sorted(o_one_positive_intersections)

#if len(o_one_positive_intersections) % 2 != 0:
 #   o_one_positive_intersections.append(len(rightSections))

o_one_negative_intersections = []
if intersection_o_one_negative.geom_type == 'MultiPoint':
    #plt.plot(*LineString(intersection_left.geoms).xy, 'o')
    o_one_negative_intersections = sorted(((LineString(intersection_o_one_negative.geoms).xy)[0]).tolist())
elif intersection_o_one_negative.geom_type == 'Point':
    #plt.plot(*intersection_left.xy, 'o') 
    o_one_negative_intersections = [intersection_o_one_negative.xy[0].tolist()[0]]

o_one_negative_intersections = sorted(o_one_negative_intersections)

#if len(o_one_negative_intersections) % 2 != 0:
#    o_one_negative_intersections.append(len(rightSections))

for x in range(len(o_one_positive_intersections)):
    o_one_positive_intersections[x] = o_one_positive_intersections[x]*frame_interval#*10# / 3
for x in range(len(o_one_negative_intersections)):
    o_one_negative_intersections[x] = o_one_negative_intersections[x]*frame_interval#*10# / 3

if rightSections[0] > 1:
    o_one_positive_intersections.insert(0,0)

if rightSections[0] < 1:
    o_one_negative_intersections.insert(0,0)

if rightSections[-1] > 1:
    o_one_positive_intersections.append(len(rightSections)*frame_interval)

if rightSections[-1] < 1:
    o_one_negative_intersections.append(len(rightSections)*frame_interval)

for x in range(len(o_one_positive_intersections)):
    o_one_positive_intersections[x] = o_one_positive_intersections[x] / 30

for x in range(len(o_one_negative_intersections)):
    o_one_negative_intersections[x] = o_one_negative_intersections[x] / 30

print('For right hand:')
print(o_one_positive_intersections)
print(o_one_negative_intersections)

#

time = np.arange(len(leftSections))

section_line = LineString(np.column_stack((time, leftSections)))

o_one_positive = np.column_stack((np.arange(0, len(leftSections) -1 ),np.full_like(np.arange(0, len(leftSections) -1 ), 1.05, dtype=np.float64)))
o_one_positive = LineString(o_one_positive)
o_one_negative = np.column_stack((np.arange(0, len(leftSections) -1 ),np.full_like(np.arange(0, len(leftSections) -1 ), 0.95, dtype=np.float64)))
o_one_negative = LineString(o_one_negative)

intersection_o_one_positive = section_line.intersection(o_one_positive)
intersection_o_one_negative = section_line.intersection(o_one_negative)

o_one_positive_intersections = []
if intersection_o_one_positive.geom_type == 'MultiPoint':
    #plt.plot(*LineString(intersection_left.geoms).xy, 'o')
    o_one_positive_intersections = sorted(((LineString(intersection_o_one_positive.geoms).xy)[0]).tolist())
elif intersection_o_one_positive.geom_type == 'Point':
    #plt.plot(*intersection_left.xy, 'o')
    o_one_positive_intersections = [intersection_o_one_positive.xy[0].tolist()[0]]

o_one_positive_intersections = sorted(o_one_positive_intersections)

#if len(o_one_positive_intersections) % 2 != 0:
 #   o_one_positive_intersections.append(len(leftSections))

o_one_negative_intersections = []
if intersection_o_one_negative.geom_type == 'MultiPoint':
    #plt.plot(*LineString(intersection_left.geoms).xy, 'o')
    o_one_negative_intersections = sorted(((LineString(intersection_o_one_negative.geoms).xy)[0]).tolist())
elif intersection_o_one_negative.geom_type == 'Point':
    #plt.plot(*intersection_left.xy, 'o') 
    o_one_negative_intersections = [intersection_o_one_negative.xy[0].tolist()[0]]

o_one_negative_intersections = sorted(o_one_negative_intersections)

#if len(o_one_negative_intersections) % 2 != 0:
#    o_one_negative_intersections.append(len(leftSections))

for x in range(len(o_one_positive_intersections)):
    o_one_positive_intersections[x] = o_one_positive_intersections[x]*frame_interval#*frame_interval# / 3
for x in range(len(o_one_negative_intersections)):
    o_one_negative_intersections[x] = o_one_negative_intersections[x]*frame_interval#*frame_interval# / 3

if leftSections[0] > 1:
    o_one_positive_intersections.insert(0,0)

if leftSections[0] < 1:
    o_one_negative_intersections.insert(0,0)

if leftSections[-1] > 1:
    o_one_positive_intersections.append(len(leftSections)*frame_interval)

if leftSections[-1] < 1:
    o_one_negative_intersections.append(len(leftSections)*frame_interval)

for x in range(len(o_one_positive_intersections)):
    o_one_positive_intersections[x] = o_one_positive_intersections[x] / 30

for x in range(len(o_one_negative_intersections)):
    o_one_negative_intersections[x] = o_one_negative_intersections[x] / 30

print('For left hand:')
print(o_one_positive_intersections)
print(o_one_negative_intersections)