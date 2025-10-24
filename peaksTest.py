o_one_positive_intersections_left = [2.5083333333333333, 3.158333333333333, 15.841666666666667, 19.825, 27.675, 29.825, 30.175, 34.65833333333333, 35.84166666666667, 38.833333333333336]

left_values_to_remove = []
for x in range(len(o_one_positive_intersections_left)):
    if (x + 1) % 2 == 0 and x != len(o_one_positive_intersections_left) - 1:
        if o_one_positive_intersections_left[x+1] - o_one_positive_intersections_left[x] < 0.5:
            left_values_to_remove.append(o_one_positive_intersections_left[x])
            left_values_to_remove.append(o_one_positive_intersections_left[x+1])

for value in left_values_to_remove:
    o_one_positive_intersections_left.remove(value)

print(o_one_positive_intersections_left)