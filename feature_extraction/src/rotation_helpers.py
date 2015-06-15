from math import sin, cos, radians as rad

def rotated(x, y, angle_deg):
    return (x * cos(rad(angle_deg)) - y * sin(rad(angle_deg)),
            y * cos(rad(angle_deg)) + x * sin(rad(angle_deg)))

def rotated_rect(x1, y1, x2, y2, angle):
    x1_, y1_ = rotated(x1, y1, angle)
    x2_, y2_ = rotated(x2, y2, angle)
    x3_, y3_ = rotated(x1, y2, angle)
    x4_, y4_ = rotated(x2, y1, angle)
    return (x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_)

def rect_in_rotated_rect( rect_coords,
                          rot_rect_coords,
                          angle ):
    origin_x = rot_rect_coords[0] + (rot_rect_coords[2] - rot_rect_coords[0])/2
    origin_y = rot_rect_coords[1] + (rot_rect_coords[3] - rot_rect_coords[1])/2

    rect_coords = [rect_coords[0] - origin_x, rect_coords[1] - origin_y,
                   rect_coords[2] - origin_x, rect_coords[3] - origin_y]

    (x1, y1, x2, y2) = tuple(rot_rect_coords)
    (x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_) = rotated_rect(*rect_coords, angle=angle)

    x1_ += origin_x; x2_ += origin_x; x3_ += origin_x; x4_ += origin_x
    y1_ += origin_y; y2_ += origin_y; y3_ += origin_y; y4_ += origin_y

    return (x1 <= x1_ <= x2) and (x1 <= x2_ <= x2) and \
        (y1 <= y1_ <= y2) and (y1 <= y2_ <= y2) and \
        (x1 <= x3_ <= x2) and (x1 <= x4_ <= x2) and \
        (y1 <= y3_ <= y2) and (y1 <= y4_ <= y2)
