import cv2
import numpy as np

def histogram_lbp(img):
    m,n = img.shape
    out = np.zeros((m-2, n-2, 8), dtype=np.uint8)
    disp = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    center = img[1:-1,1:-1]

    for i,d in enumerate(disp):
        out[:,:,i] = img[d[0]+1:d[0]+m-1, d[1]+1:d[1]+n-1] >= center
        out[:,:,i] *= 2**i

    lbp = np.sum(out,axis=2)
    hist = np.histogram(lbp,256,density=True)[0]
    return hist, lbp

def histogram_gabor(img):
    kernels = [
        cv2.getGaborKernel((11,11),3,np.pi/4,11,1),
        cv2.getGaborKernel((11,11),3,-np.pi/4,11,1),
        cv2.getGaborKernel((11,11),2,np.pi/4,5,1),
        cv2.getGaborKernel((11,11),2,-np.pi/4,5,1)
    ]

    img = img / 255.0
    binaries = [(cv2.filter2D(img, cv2.CV_32F, k) > 0).astype(np.uint8) for k in kernels]
    texture_map = sum(b * (2**i) for i,b in enumerate(binaries))
    hist = np.histogram(texture_map, bins=16, range=(0,16), density=True)[0]
    return hist, texture_map