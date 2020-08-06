# _____________________________________________________________________________
# perspective.py
# _____________________________________________________________________________
import numpy
import cv2

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def shi_tomasi(im, val):
    maxCorners = max(val, 1)
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = True
    k = 0.04
    #Process the image before finding points.
    im = cv2.GaussianBlur(im, (3, 3), 0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Apply corner detection
    corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel,
                                      minDistance, None, blockSize=blockSize,
                                      gradientSize=gradientSize,
                                      useHarrisDetector=useHarrisDetector, k=k)
    return corners


def find_perspective_transform(src, dst):
    # Taken from : Computer Vision (CE316 and CE866): Vision in a 3D World, Adrian F. Clark
    n = len(pos)
    if n != 4:
        raise Exception("Wrong number of matches.")
    a = numpy.zeros((8, 8), dtype=numpy.float32)
    b = numpy.zeros((8), dtype=numpy.float32)
    m = numpy.zeros((3, 3), dtype=numpy.float32)
    for i in range(0, 4):
        a[i, 0] = a[i + 4, 3] = src[i, 0]
        a[i, 1] = a[i + 4, 4] = src[i, 1]
        a[i, 2] = a[i + 4, 5] = 1
        a[i, 3] = a[i, 4] = a[i, 5] = a[i + 4, 0] = a[i + 4, 1] = a[i + 4, 2] = 0
        a[i, 6] = -src[i, 0] * dst[i, 0]
        a[i, 7] = -src[i, 1] * dst[i, 0]
        a[i + 4, 6] = -src[i, 0] * dst[i, 1]
        a[i + 4, 7] = -src[i, 1] * dst[i, 1]
        b[i] = dst[i, 0]
        b[i + 4] = dst[i, 1]
    # Solve the set of equations and copy the solution into a transformation
    # matrix, which we return.
    x = numpy.linalg.solve(a, b)
    m[0, 0] = x[0]
    m[1, 0] = x[3]
    m[2, 0] = x[6]
    m[0, 1] = x[1]
    m[1, 1] = x[4]
    m[2, 1] = x[7]
    m[0, 2] = x[2]
    m[1, 2] = x[5]
    m[2, 2] = 1.0
    return m
# -----------------------------------------------------------------------------
# Main program.
# -----------------------------------------------------------------------------

# Size of the output image
n = 595
m = 420

# Define the image containing the region to extract and the boundaries
# of the region.
pos = numpy.zeros((4, 2), dtype=numpy.float32)
fn = "images\\paper.jpg"
im = cv2.imread(fn)
corners = shi_tomasi(im, 4)


corner = []
for c in corners:
    corner.append(c[0])
corner = numpy.array(corner)


# Draw a polygon from each corner.
copy = numpy.copy(im)
cv2.polylines(copy, numpy.int32([corner]), False, (0, 255, 0), thickness=10)
copy = cv2.resize(copy, (m, n), interpolation=cv2.INTER_AREA) 
# Sort corners for - x first then y.
x_sort = corner[corner[:, 0].argsort()]
y_sort = corner[corner[:, 1].argsort()]
x_right = x_sort[2:, :]
x_right = x_right[x_right[:, 1].argsort()]
x_left = x_sort[:2, :]
x_left = x_left[x_left[:, 1].argsort()]

pos[0, :] = x_left[0]  # Upper Left x, Upper Left y
pos[1, :] = x_left[1]  # Lower Left x, Lower Left y
pos[2, :] = x_right[1]  # Lower Right x, Lower Right y
pos[3, :] = x_right[0]  # Upper Right x, Upper Right y


opos = numpy.zeros((4, 2), dtype=numpy.float32)
opos[0, :] = [0, 0]
opos[1, :] = [0, n-1]
opos[2, :] = [m-1, n-1]
opos[3, :] = [m-1, 0]
# Transformation from the match-points.
xform = find_perspective_transform(pos, opos)
# Correct the perspective transform.
warp = cv2.warpPerspective(im, xform, (m, n))

cv2.imwrite("images\\rectifiedPaper.jpg", warp)
both = numpy.hstack((copy, warp))   # stack images
cv2.imwrite("images\\result.jpg", both)
cv2.waitKey()

# -----------------------------------------------------------------------------
# End of perspective.py
# -----------------------------------------------------------------------------


# _____________________________________________________________________________
# TITLE - perspective.py
# AUTHOR - Ogulcan Ozer.
# C_DATE - 19 Jun 2020
# U_DATE - 06 Aug 2020
# _____________________________________________________________________________
