import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def pointshifter(image,points):
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(image)
    l = ax.scatter(points[:,0],points[:,1],picker=True)


    def onkey(event):
        l.set_offset_position('data')
        offsets = l.get_offsets()
        if event.key == "up":
            offsets[:,1]+=1
        if event.key == "down":
            offsets[:,1]-=1
        if event.key == "left":
            offsets[:,0]-=1
        if event.key == "right":
            offsets[:,0]+=1
        l.set_offsets(offsets)
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
    return l.get_offsets()

