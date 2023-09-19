#! /usr/bin/python3

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import math

# Plot L1 norm
# figure, axes = plt.subplots()
# Drawing_diamond = patches.RegularPolygon((0, 0), 4, 1)
# axes.set_aspect( 1 )
# axes.add_artist( Drawing_diamond )
# plt.xlim(-1.2, 1.2)
# plt.ylim(-1.2, 1.2)
# plt.title( 'L1 Norm' )
# plt.show()

# # Plot L2 norm
# figure, axes = plt.subplots()
# Drawing_circle = plt.Circle( (0, 0 ),
#                                       1 ,
#                                       fill = True )
 
# axes.set_aspect( 1 )
# axes.add_artist( Drawing_circle )
# plt.xlim(-1.2, 1.2)
# plt.ylim(-1.2, 1.2)
# plt.title( 'L2 Norm' )
# plt.show()

# Plot L_inf norm
figure, axes = plt.subplots()
Drawing_square = plt.Rectangle((-1, -1), 2, 2)
axes.set_aspect( 1 )
axes.add_artist( Drawing_square )
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.title( 'L_inf Norm' )
plt.show()