import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


fig = plt.figure()


img = mpimg.imread('test_images/straight_lines1.jpg')
plt.imshow(img)
coords = []

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print ('x = %d, y = %d'%(
        ix, iy))

    global coords
    coords.append((ix, iy))

    if len(coords) == 8:
        fig.canvas.mpl_disconnect(cid)

    return coords
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

'''obtained following data'''
'''
x = 541, y = 490
x = 747, y = 490
x = 254, y = 681
x = 1049, y = 681
'''
