import sys
import cv2
import calibrate
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.util import img_as_ubyte
from matplotlib.axes._axes import _log as matplotlib_axes_logger


matplotlib_axes_logger.setLevel('ERROR')

pointX = [calibrate.HorizontalRes*calibrate.VerticalRes]
pointY = [calibrate.HorizontalRes*calibrate.VerticalRes]

def filler(pixelX, pixelY, snow):
    #filler records relevant plots with snow and calibrates them in accordance to the top down image's coordinates
    if(snow):
        GroundX = (calibrate.AOI[1][2]-calibrate.AOI[0][2]) / (calibrate.AOI[1][0] - calibrate.AOI[0][0]) * pixelX - (calibrate.AOI[1][2] - calibrate.AOI[0][2]) / (calibrate.AOI[1][0] - calibrate.AOI[0][0]) * calibrate.AOI[0][0] + calibrate.AOI[0][2]
        GroundY = (calibrate.AOI[3][3] - calibrate.AOI[1][3]) / (calibrate.AOI[3][1] - calibrate.AOI[1][1]) * pixelY - (calibrate.AOI[3][3] - calibrate.AOI[1][3]) / (calibrate.AOI[3][1] - calibrate.AOI[1][1]) * calibrate.AOI[0][1] + calibrate.AOI[0][3]
        TopAOI = (calibrate.AOI[3][0] - calibrate.AOI[2][0])
        BottomAOI = (calibrate.AOI[1][0] - calibrate.AOI[0][0])
        maxFactor = (TopAOI / BottomAOI)
        
        factor = (maxFactor-1) / (calibrate.AOI[3][1] - calibrate.AOI[1][1]) * pixelY #+ 1 - (1- maxFactor) / (calibrate.AOI[3][1] - calibrate.AOI[1][1]) * calibrate.AOI[1][1]
        
        b = 1 - (maxFactor-1) / (calibrate.AOI[3][1] - calibrate.AOI[1][1]) * calibrate.AOI[1][1]
        
        factor = factor + b
        
        GroundX = GroundX / factor
        pointX.append(GroundX)
        pointY.append(GroundY)            
        return

def feed_filler(array):
    #feed_filler iterates through the thresholded image array and feeds it to filler function
    arrayXlen = array.shape[0]
    arrayYlen = array.shape[1]
    for i in range(0, arrayXlen):
        for j in range(0, arrayYlen):
            if (array[i][j].all() != 0):
                xCoord = array[i][j][0]
                yCoord = array[i][j][1]
                filler(yCoord, xCoord, True)

def detect_snow(imgFirst):
    #convert image to grayscale as thresholds only work in that context
    image = cv2.cvtColor(imgFirst, cv2.COLOR_BGR2GRAY)
    #apply otsu threshold
    thresh = threshold_otsu(image)
    binary = image > thresh
 
    binary = img_as_ubyte(binary)
    binaryX = binary.shape[0] #1080
    binaryY = binary.shape[1] #1920
    
    #create array of values that stores the coordinates of each pixel that is identified as white/snow
    binaryCoordinates = np.zeros((binaryX,binaryY), dtype=(int,2))
    snowCount = 0
    for i in range(0, binaryX):
        for j in range(0, binaryY):
            if (binary[i][j] == 255):
                snowCount = snowCount + 1
                binaryCoordinates[i][j] = (i,j)
         
    snowCoverage = int(100*(snowCount/(binaryX*binaryY)))
    print("Finished making coordinate system")
    feed_filler(binaryCoordinates)
    return snowCoverage

def plotsnow(snowCoverage):
    #plotsnow plots the calibrated coordinates of snow onto the topdown view picture
    plt.xlim(calibrate.AOI[0][2], calibrate.AOI[1][2]) 
    plt.ylim(calibrate.AOI[0][3], calibrate.AOI[3][3])
    area = np.pi * 1
    colors = (0,0,0)
    img = plt.imread("topDown.png")
    snowCoverage = str(snowCoverage)
    snowLabel = "Snow plot for coverage of " + snowCoverage + "%" 
    plt.imshow(img, extent=[calibrate.AOI[0][2], calibrate.AOI[1][2],calibrate.AOI[0][3], calibrate.AOI[3][3]])
    
    plt.scatter(pointX, pointY, s=area, c=colors, alpha=0.5)
    plt.title(snowLabel)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('test.png')
    print("Picture Generated")

def main(img):
    #load image and resize to 1080 x 1920...we need these dimensions for plotting
    img2 = cv2.imread(img)
    dim = (1920, 1080)
    resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
    snowPercentage = detect_snow(resized)
    plotsnow(snowPercentage)
    cv2.waitKey(0)
    
main(sys.argv[1])
