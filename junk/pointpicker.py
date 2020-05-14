import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def pointpicker(image,points):
    plt.close()
    fig, ax = plt.subplots()

    plt.subplots_adjust(bottom=0.2)

    ax.imshow(image)

    red = (1,0,0,1)
    green = (0,1,0,1)
    orange = (0.6,0.2,0.1,1)
    global mode
    mode = red
    exts = []
    pois = []
    mults = []
    colorcode = {red: exts,
            green: pois,
            orange: mults}
    c = ["C0"]*points.shape[0]

    def extended(event):
        global mode
        mode = red
        #l.set_title('Selecting Extended Sources')

    def point(event):
        global mode
        mode = green
        #l.title('Selecting Point Sources')
    def multi(event):
        global mode
        mode = orange
    def onclick(event):
        i = event.ind
        l._facecolors[i,:] = mode
        fig.canvas.draw()

    axpoint = plt.axes([0.7, 0.05, 0.1, 0.075])
    axext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bext = Button(axext, 'Extended')
    bext.on_clicked(extended)
    bpoint = Button(axpoint, 'Point')
    bpoint.on_clicked(point)
    axmul = plt.axes([0.61, 0.05, 0.1, 0.075])
    bmul = Button(axmul, 'Mutli')
    bmul.on_clicked(multi)

    l = ax.scatter(points[:,0],points[:,1],c=c,picker=True)

    fig.canvas.mpl_connect('pick_event', onclick)

    plt.show()

    for i,fc in enumerate(l._facecolors):
        fc = tuple(fc)
        if fc in colorcode.keys():
            colorcode[fc].append(i)
    assignments = dict()
    assignments['point'] = colorcode[green]
    assignments['extended'] = colorcode[red]
    assignments['multi'] = colorcode[orange]

    print("Point: ",pois)
    print("Extended: ",exts)
    print("Mutli: ",mults)
    return assignments
