import numpy as np
import matplotlib.pyplot as plt
import bilby
from astropy.constants import h,c,k_B
from synthetic_cube import BlackbodyCubeSampler

class BlackBody(bilby.Likelihood):
    def __init__(self, wl, images, weights, seeing):
        super().__init__(parameters={
            'T': None,
            'A': None,
            'x': None,
            'y': None,
            })
        self.wl = wl
        self.images = images
        weights[weights<1e-5]=1e-6
        self.stds = 1/weights**0.5
        self.seeing = seeing
        _,sizex,sizey = images.shape
        self.bcs = BlackbodyCubeSampler(wl, sizex, sizey, seeing=seeing)
        #self.N = np.prod(images.shape)
    def log_likelihood(self):
        T = self.parameters['T']
        A = self.parameters['A']
        x = self.parameters['x']
        y = self.parameters['y']
        wl = self.wl
        images = self.images
        stds = self.stds
        seeing = self.seeing
        bcs = self.bcs
        _, sizex, sizey = images.shape
        maxy, miny = int(min(sizey, round(y + sizey/2 + bcs.size/2))), int(max(0, round(y + sizey/2 - bcs.size/2)))
        maxx, minx = int(min(sizex, round(x + sizex/2 + bcs.size/2))), int(max(0, round(x + sizex/2 - bcs.size/2)))
        cimages = images[:, miny:maxy, minx:maxx]
        #cstds = stds[:, miny:maxy, minx:maxx]
        cstds = 1e-3*np.ones_like(cimages) #stds[:, miny:maxy, minx:maxx]
        bcss = bcs.sample(T,A,x,y)
        res = (cimages - bcss)/cstds
        #plt.imshow(images.sum(axis=0))
        #plt.scatter(x+31.5,y+31.5,label="%.2d,%.2d"%(x+31.5,y+31.5))
        #plt.legend()
        #plt.show()
        #plt.imshow(cimages.sum(axis=0))
        #plt.show()
        #plt.imshow(bcss.sum(axis=0))
        #plt.show()
        #l1 = (1-pb)/np.sqrt(2*np.pi*sy**2) * np.exp(-(y - m*x - b)**2 / 2 / sy**2)
        #l2 = pb/np.sqrt(2*np.pi*(vb+sy**2))* np.exp(-(y - yb)**2 / 2 / (vb+sy**2))
        #l = 1/np.sqrt(2*np.pi*sy**2) * np.exp(-(y - A*blackbody(x,T))**2 / 2 / sy**2)
        #res = (images - blackbodycube(T, A, x, y, wl, size, seeing))/stds
        #return -0.5 * np.sum(res ** 2 + np.log(2 * np.pi * stds ** 2))
        #logl = -0.5 * np.sum(res ** 2 + np.log(2 * np.pi * cstds ** 2))
        logl = -0.5 * np.sum(res ** 2) + -0.5*np.sum(np.log(2 * np.pi * cstds ** 2))
        logl += np.log(np.sum(cimages))
        #print(x,y,logl)
        return logl
weights = np.load('weights.npy')
size = weights.shape[1]
wl = np.load('wl.npy')
images = np.load('images.npy')
seeing = 2

def test_log_likelihood(T,A,x,y):
    _, sizex, sizey = images.shape
    bcs = BlackbodyCubeSampler(wl, sizex, sizey, seeing=seeing)
    stds = 1/weights**0.5
    maxy, miny = int(min(sizey, round(y + sizey/2 + bcs.size/2))), int(max(0, round(y + sizey/2 - bcs.size/2)))
    maxx, minx = int(min(sizex, round(x + sizex/2 + bcs.size/2))), int(max(0, round(x + sizex/2 - bcs.size/2)))
    cimages = images[:, miny:maxy, minx:maxx]
    cstds = 1e-3*cimages #stds[:, miny:maxy, minx:maxx]
    bcss = bcs.sample(T,A,x,y)
    res = (cimages - bcss)/cstds
    plt.imshow(images.sum(axis=0))
    plt.scatter(x+31.5,y+31.5,label="%.2d,%.2d"%(x+31.5,y+31.5))
    plt.legend()
    plt.show()
    plt.imshow(cimages.sum(axis=0))
    plt.show()
    plt.imshow(bcss.sum(axis=0))
    plt.show()
    plt.imshow(res.sum(axis=0))
    plt.show()
    #l1 = (1-pb)/np.sqrt(2*np.pi*sy**2) * np.exp(-(y - m*x - b)**2 / 2 / sy**2)
    #l2 = pb/np.sqrt(2*np.pi*(vb+sy**2))* np.exp(-(y - yb)**2 / 2 / (vb+sy**2))
    #l = 1/np.sqrt(2*np.pi*sy**2) * np.exp(-(y - A*blackbody(x,T))**2 / 2 / sy**2)
    #res = (images - blackbodycube(T, A, x, y, wl, size, seeing))/stds
    #return -0.5 * np.sum(res ** 2 + np.log(2 * np.pi * stds ** 2))
    logl = -0.5 * np.sum(res ** 2) + -0.5*np.sum(np.log(2 * np.pi * cstds ** 2))
    #print(x,y,logl)
    return logl
likelihood = BlackBody(wl, images, weights, seeing)
uniform = bilby.core.prior.Uniform
priors = dict(
        T=uniform(name='T',minimum=1e3,maximum=1e4),
        A=uniform(name='A',minimum=0,maximum=1e3),
        x=uniform(name='x',minimum=-size/2+2,maximum=size/2-2),
        y=uniform(name='y',minimum=-size/2+2,maximum=size/2-2),
        #x=uniform(name='x',minimum=-size/2,maximum=-29),
        #y=uniform(name='y',minimum=-3,maximum=3),
        )
result = bilby.run_sampler(likelihood=likelihood,priors=priors,
        sampler='dynesty',npoints=5000,walks=20,outdir='samples',label='bbtest',dlogz=1e2,plot=True)
result.plot_corner()
