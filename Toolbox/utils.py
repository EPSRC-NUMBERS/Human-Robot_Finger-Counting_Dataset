"""
This file contains a collection of routines that help with hand silhouetting.
"""

import cv2
import numpy as np
import math

from pandas import DataFrame
import xml.etree.ElementTree as ET
from pathlib import Path, PurePath

def detectObject( imgBk, img, threshold ):

    difference = cv2.absdiff(imgBk, img)
    differenceS = cv2.split(difference)
    object = cv2.inRange(differenceS[2], threshold, 255)
    #object = cv2.medianBlur(object,5)
    mask = cv2.inRange(differenceS[0], 50, 255)
    mask = cv2.bitwise_or(object, mask)
    #mask = cv2.medianBlur(mask,5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = cv2.erode(mask, kernel, iterations = 2)
    return object, mask


def combineImages(img, imgBk, mask):

    # img1 = img * (255 * mask[:,:,None].astype(img.dtype))
    # mask = cv2.bitwise_not(mask);
    # img2 = imgBk * (255 * mask[:,:,None].astype(img.dtype))
    # res = img1 + img2;
    locs = np.where(mask != 0)
    res = imgBk
    res[locs[0], locs[1], :] = img[locs[0], locs[1], :]
    return res



# def getAngleABC(a, b, c):
#
#     ab = np.array([b[0] - a[0], b[1] - a[1]])
#     cb = np.array([b[0] - c[0], b[1] - c[1]])
#
#     // dot product
#     dot = ab.dot(cb) #(ab[0] * cb[0] + ab[1] * cb[1]);
#
#     // length square of both vectors
#     abSqr = ab.dot(ab)
#     cbSqr = cb.dot(cb)
#
#     // square of cosine of the needed angle
#     cosSqr = dot * dot / abSqr / cbSqr;
#
#     // this is a known trigonometric equality:
#     // cos(alpha * 2) = [ cos(alpha) ]^2 * 2 - 1
#     cos2 = 2 * cosSqr - 1;
#
#     // Here's the only invocation of the heavy function.
#     // It's a good idea to check explicitly if cos2 is within [-1 .. 1] range
#
#     const float pi = 3.141592f;
#
#     float alpha2 =
#         (cos2 <= -1) ? pi :
#         (cos2 >= 1) ? 0 :
#         acosf(cos2);
#
#     float rslt = alpha2 / 2;
#
#     float rs = rslt * 180. / pi;
#
#
#     // Now revolve the ambiguities.
#     // 1. If dot product of two vectors is negative - the angle is definitely
#     // above 90 degrees. Still we have no information regarding the sign of the angle.
#
#     // NOTE: This ambiguity is the consequence of our method: calculating the cosine
#     // of the double angle. This allows us to get rid of calling sqrt.
#
#     if (dot < 0)
#         rs = 180 - rs;
#
#     // 2. Determine the sign. For this we'll use the Determinant of two vectors.
#
#     float det = (ab.x * cb.y - ab.y * cb.y);
#     if (det < 0)
#         rs = -rs;
#
#     return (int) abs(floor(rs + 0.5));
#
#
#
# def filterContour(countoursIn):
#
#     if len(contoursIn) > 4:
#         countoursOut = []
#         countoursOut.append(countoursIn[0])
#         countoursOut.append(countoursIn[1])
#         angles = []
#         angles.append(0);
#         angles.append(0);
#     int angle = 0;
#     int angleThreshold;
#
#         for i in range(2, len(contoursIn)):
#             angle = getAngleABC(countoursIn[i-2], countoursIn[i-1], countoursIn[i])
#             angles.push_back(angle);
#
#
#     Scalar tempVal1, tempVal2;
#     meanStdDev( angles, tempVal1, tempVal2 );
#     int meanA = (int)tempVal1.val[0];
#     int stdA = (int)tempVal2.val[0];
#     angleThreshold = meanA - stdA*0.5;
#
#     for (int i = 2; i<nContour; i++)
#     {
#         if (angles[i] > angleThreshold)
#         {
#             countoursOut.push_back(countoursIn[i]);
#         }
#     }
#
#     return countoursOut, angles


def angleBetween(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def angleOf(p1):
    p2 = (0, 1)
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def filterContour(contoursIn):
    return contoursIn

def defineContours(contours, aproximation, numberMinPointsThreshold, numberMaxPointsThreshold):

    contoursResult = []
    for i in range(len(contours)):
        contourAP = cv2.approxPolyDP(contours[i], aproximation, 1)
        if contourAP.size > numberMinPointsThreshold:
            contoursResult.append(contourAP)

    # To add contour filtering
    contoursFiltered = filterContour(contoursResult)

    return contoursResult, contoursFiltered

def boundingBox(contours, parameter = 'area'):

    if parameter == 'area':
        value = 0
    else:
        value = 100000

    index = 0
    x, y, w, h, index = 0, 0, 0, 0, -1
    for i in range(len(contours)):
        valueA = cv2.contourArea(contours[i])
        if parameter == 'area':
            if valueA > value:
                value = valueA
                x, y, w, h = cv2.boundingRect(contours[i])
                index = i
        else:
            x_, y_, w_, h_ = cv2.boundingRect(contours[i])
            valueC = y_ + (h_ / 2)

            if (valueC < value) & (valueA > 100):
                value = valueC
                x, y, w, h = cv2.boundingRect(contours[i])
                index = i
            # print (str(value) + '  ' + str(valueC))

    return x, y, w, h, index

def redefineObject(object, contour):
    kernel = np.ones((5, 5), np.uint8)

    maskObject = np.zeros((object.shape), np.uint8)
    i = 0
    ratio = 0
    while ratio < 0.5:
        cv2.drawContours(maskObject, [contour[0]], -1, (255,255,255), -1, cv2.LINE_AA)
        maskObject = cv2.erode(maskObject, kernel, iterations = 1)
        object, contour, hierarchy = cv2.findCont1ours(maskObject,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        area = cv2.contourArea(contour[0])
        hull = cv2.convexHull(contour[0])
        hull_area = cv2.contourArea(contour[0])
        ratio = float(area)/hull_area
        print(ratio)
        cv2.imshow("hull", hull)
        cv2.imshow("object", object)
        cv2.imshow("maskObject", maskObject)
        cv2.waitKey(0)

    return maskObject

def features(img, maskObject, contour):
    # Features has been extracted of
    # https://docs.opencv.org/3.4.1/dd/d49/tutorial_py_contour_features.html
    # https://docs.opencv.org/3.4.1/d1/d32/tutorial_py_contour_properties.html

    # Centroid
    M = cv2.moments(maskObject)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # Area
    area = cv2.contourArea(contour)

    # Perimeter
    perimeter = cv2.arcLength(contour,1)

    # Orientation
    if len(contour) > 5:
        #print(contour)
        (xE,yE),(ma,MA),angle = cv2.fitEllipse(contour)
        try:
            xE;yE;ma;MA;angle
        except NameError :
            xE,yE,ma,MA,angle = 0,0,0,0,0
        if MA > 0:
            aspectRatio = ma/MA
        else:
            aspectRatio = 1000
    else:
        angle = 0
        aspectRatio = 0
        xE,yE,ma,MA,angle = 0,0,0,0,0

    #Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    try:
        solidity = float(area)/hull_area
    except ZeroDivisionError:
        solidity = 1

    # weighted centroid acoording with gray level luminosity
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imageMask = cv2.bitwise_and(imgG,maskObject)
    x = range(0, imageMask.shape[0])
    y = range(0, imageMask.shape[1])

    (X,Y) = np.meshgrid(x,y)
    X = np.transpose(X)
    Y = np.transpose(Y)

    x_coord = (X*imageMask).sum() / imageMask.sum().astype("float")
    y_coord = (Y*imageMask).sum() / imageMask.sum().astype("float")

    #Mean Color
    mean_val = cv2.mean(img, mask = maskObject)

    area = M['m00']/255 # area expressed as momentum order 0 is accurate

    vector = (cx - y_coord, cy - x_coord)
    angleIllumination = angleOf(vector)
    distanceIllumination = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])

    return [cx, cy, area, perimeter, angle, aspectRatio, solidity, int(y_coord), int(x_coord), mean_val[0], mean_val[1], mean_val[2], int(ma/2), int(MA/2), int(xE), int(yE), angleIllumination, distanceIllumination]


def saveFileXML(filename, resultFolder, path, heigth, width, channels, name, x, y, w, h):
    xmlName = Path(PurePath(filename).with_suffix(".xml")).resolve()
    xmlPath = Path.resolve(path / resultFolder / xmlName)
    xmlFile = open (xmlPath, 'w')
    xmlFile.write ("<annotation>" + "\n")
    xmlFile.write ("\t" + "<folder>" + str(resultFolder) + "</folder>" + "\n")
    xmlFile.write ("\t" + "<filename>" + str(filename) + "</filename>" + "\n")
    xmlFile.write ("\t" + "<path>" + str(path) + "/" + str(filename) + "</path>\n")
    xmlFile.write ("\t" + "<source>\n\t\t<database>Unknown</database>\n\t</source>\n")
    xmlFile.write ("\t<size>\n\t\t<width>" + str(width) + "</width>\n\t\t<height>" + str(heigth) + "</height>\n\t\t<depth>" + str(channels) + "</depth>\n\t</size>\n")
    xmlFile.write ("\t<segmented>0</segmented>\n\t<object>\n")
    xmlFile.write ("\t\t<name>" + name + "</name>\n")
    xmlFile.write ("\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n")
    xmlFile.write ("\t\t<bndbox>\n\t\t\t<xmin>" + str(x) + "</xmin>\n")
    xmlFile.write ("\t\t\t<ymin>" + str(y) + "</ymin>\n")
    xmlFile.write ("\t\t\t<xmax>" + str(x + w) + "</xmax>\n")
    xmlFile.write ("\t\t\t<ymax>" + str(y + h) + "</ymax>\n")
    xmlFile.write ("\t\t</bndbox>\n\t</object>\n")
    xmlFile.write ("</annotation>")

    xmlFile.close()
    return


def plotData(dataX, dataY, axisXRange = [], axisYRange = [], color = (128,255,128), base =[], steps = [], type = "line", title = "",  axisColor = (255,255,255)):

    if base==[]:
        base = np.zeros((480,640,3), np.uint8)

    h, w, _ = base.shape

    if axisXRange==[]:
        axisXRange=[np.amin(dataX), np.amax(dataX)]

    if axisYRange==[]:
        axisYRange=[np.amin(dataY), np.amax(dataY)]

    axisRange = np.append(axisXRange, axisYRange)

    if steps==[]:
        steps = [int(round(w/100)), int(round(h/100))]

    # print(dataX)
    # print(dataY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    xMargin = 45
    yMargin = 25
    y1Margin = 10
    #xMin = np.amin(data[:,0])
    #xMax = np.amax(data[:,0])
    #yMin = np.amin(data[:,1])
    #yMax = np.amax(data[:,1])
    xMin = axisRange[0]
    xMax = axisRange[1]
    yMin = axisRange[2]
    yMax = axisRange[3]
    # print(xMin)
    # print(xMax)
    # print(yMin)
    # print(yMax)
    scaleX = (w-(2*xMargin))/(xMax-xMin)
    scaleY = (h-(2*(yMargin+y1Margin)))/(yMax-yMin)
    xStep = steps[0]
    xCoor = np.arange(xMargin,w-xMargin,int((w- xMargin - xMargin)/xStep))
    xCoor = np.append(xCoor, w-xMargin)
    inc = ((xMax-xMin)/xStep)
    # if inc == 0:
    #     inc = 1
    #xValues = list(range(xMin,xMax,int(inc)))
    xValues = np.arange(float(xMin),float(xMax),inc)
    xValues = np.append(xValues, xMax)
    yStep = steps[1]
    yCoor = np.arange(yMargin + y1Margin,h-yMargin-y1Margin,int((h- 2*(yMargin + y1Margin))/yStep))
    yCoor = np.append(yCoor, h-yMargin-y1Margin)
    #print (yCoor)
    inc = ((yMax-yMin)/yStep)
    # if inc == 0:
    #     inc = 1
    #yValues = list(range(float(yMax),float(yMin),-inc))
    yValues = np.arange(float(yMax),float(yMin),-inc)
    yValues = np.append(yValues, yMin)
    for xC,xV in zip(xCoor, xValues):
        cv2.putText(base,str(round(xV,2)),(xC-10 ,h-yMargin+5), font, 0.4, axisColor,1,cv2.LINE_AA)
        cv2.line(base, (xC ,h-yMargin-y1Margin+4), (xC ,h-yMargin-y1Margin), axisColor, 1)
    for yC,yV in zip(yCoor, yValues):
        cv2.putText(base,str(round(yV,2)),(5, yC + 4), font, 0.4, axisColor,1,cv2.LINE_AA)
        cv2.line(base, (xMargin-4 , yC), (xMargin , yC), axisColor, 1)

    # cv2.putText(base,str(xMin),(xMargin,h-yMargin), font, 0.4, axisColor,1,cv2.LINE_AA)
    # cv2.putText(base,str(xMax),(w-xMargin,h-yMargin), font, 0.4, axisColor,1,cv2.LINE_AA)
    #
    # cv2.putText(base,str(yMin),(5,h-yMargin-y1Margin), font, 0.4, axisColor,1,cv2.LINE_AA)
    # cv2.putText(base,str(yMax),(5,yMargin+y1Margin), font, 0.4, axisColor,1,cv2.LINE_AA)

    #offsetX = [xMin] - np.amin(dataX)
    #print(offsetX)
    dataX = (dataX - xMin)*scaleX + xMargin
    dataY = h - yMargin - y1Margin - (dataY - yMin)*scaleY

    zeroValueY = int(h - yMargin - y1Margin + yMin*scaleY)

    if type == "line":
        pts = np.vstack((dataX,dataY)).astype(np.int32).T
        cv2.polylines(base, [pts], isClosed=False, color=color)
    elif type == "rectangle":
        for i, yVal in enumerate(dataY):
            incDataX = int(dataX[1] - dataX[0])
            xVal = int(dataX[i])
            tl = ( xVal,int(yVal))
            br = ( xVal + incDataX, zeroValueY)
            cv2.rectangle(base, tl, br, color, 1, cv2.LINE_AA, 0)
    else:
        pts = np.vstack((dataX,dataY)).astype(np.int32).T
        for i in pts:
            cv2.circle(base,(i[0],i[1]),3,color, 1, cv2.LINE_AA, 0)

    cv2.line(base, (xMargin,h-yMargin-y1Margin), (w-xMargin,h-yMargin-y1Margin), axisColor, 1)
    cv2.line(base, (xMargin,h-yMargin-y1Margin), (xMargin, yMargin+y1Margin), axisColor, 1)
    cv2.putText(base,title,(xMargin,yMargin), font, 0.5,axisColor,1,cv2.LINE_AA)


    return base

def plotHistogram(dataH, bins = 'auto', roundFactor = 1, figure = [], axisXRange = [], axisYRange = [], color = (0,255,255), title = "Histogram"):

    # # print(np.amax(dataH))
    minData = np.amin(dataH)
    maxData = np.amax(dataH)
    # if minData > 1:
    #     minData = int(round(np.amin(dataH)/roundFactor)*roundFactor)
    #     maxData = int(round(np.amax(dataH)/roundFactor)*roundFactor)
    # BinWidth = (maxData-minData)/NumBins
    # bins=np.arange(float(minData), float(maxData), BinWidth)
    # binsE = np.append(bins,[maxData])
    # # print(bins)
    # # print(binsE)

    hist, bin_edges = histArea = np.histogram(dataH, bins)
    # print(title)
    # print(histArea[0])
    # print(histArea[1])
    # print(np.sum(histArea[0]))
    # print(len(dataH))

    if axisXRange==[]:
        axisXRange=[minData, maxData]

    if axisYRange==[]:
        axisYRange=[0, np.amax(histArea[0])]

    # print(axisRange)
    figure = plotData(bin_edges, hist, axisXRange, axisYRange, color, figure, [], "rectangle", title)
    #figure = plotData(bin_edges[0:len(bin_edges-2)], hist, axisXRange, axisYRange, color, figure, [int(len(bin_edges)/2),10], "rectangle", title)
    #figureArea = plotData(bins, histArea[0], axisXRange, axisYRange, color, figure, [int(NumBins/2),10], "rectangle", title)
    #figureArea = plotData(bins, histArea[0], axisRange, color, figure, [(NumBins),10], "line", title)
    return figure

def drawFeatures(img, contoursPoints, contoursFiltered, boundinBox, data):

    pt1 = (boundinBox[0], boundinBox[1])
    pt2 = (boundinBox[2], boundinBox[3])

    cv2.drawContours(img, [contoursPoints], -1, (128,255,255), 1, cv2.LINE_AA)
    cv2.drawContours(img, [contoursFiltered], -1, (255,128,255), 1, cv2.LINE_AA)
    cv2.rectangle(img, pt1, pt2, (255,255,128), 1, cv2.LINE_AA, 0)
    cv2.circle(img,(data[2],data[3]),10,(255,128,128), 1, cv2.LINE_AA, 0)
    cv2.circle(img,(data[9],data[10]),10,(128,255,128), 1, cv2.LINE_AA, 0)
    cv2.ellipse(img, (data[16],data[17]), (data[14],data[15]), data[6], startAngle=0, endAngle=360, color=(128, 128, 255), thickness = 1)#, cv2.LINE_AA, 0)
    cv2.arrowedLine(img, (data[2],data[3]), (data[9],data[10]), color=(200, 200, 200), thickness=1)#, int line_type=8, int shift=0, double tipLength=0.1)¶
    font = cv2.FONT_HERSHEY_SIMPLEX
    height = np.size(img, 0)
    cv2.putText(img,str(data[4])[:7],(5, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[5])[:7],(65, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[6])[:5],(125, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[7])[:5],(175, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[8])[:5],(225, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[11])[:5],(275, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[12])[:5],(325, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[13])[:5],(375, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[14])[:5],(430, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[15])[:5],(480, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[18])[:5],(530, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[19])[:5],(580, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(img,str(data[1])[:5],(625, height-10), font, 0.4,(255,255,255),1,cv2.LINE_AA)
    legend = " area   perimeter angle  aRatio solidity meanB meanG meanR minAxis majAxis angleIlu distIlu ID"
    cv2.putText(img,legend,(10, height-25), font, 0.4,(255,255,255),1,cv2.LINE_AA)

    return img

def drawFigures(allData, color, figPos, figAngle, figArea, figPerimeter, figAspectRatio, figSolidity, figBmean, figGmean, figRmean, figAIlu, figDIlu, figDAIlu):

    # POSITION
    figPos = fn.plotData(allData[:,0], allData[:,1], [0,w], [0,h], color, figurePos, [4,4], "circle", "Position " + legend)
    # ANGLE
    figAngle = fn.plotHistogram(allData[:,4], title = "Histogram Angle", figure = figureAngle, color = color, bins=10, axisXRange =[0, 180], axisYRange =[0, 100])
    # AREA
    figArea = fn.plotHistogram(allData[:,2], roundFactor = 1000, title = "Histogram Area", figure = figureArea, axisXRange =[8000, 80000], axisYRange =[0, 45], color = color)
    # PERIMETER
    figPerimeter = fn.plotHistogram(allData[:,3], roundFactor = 100, title = "Histogram Perimeter", figure = figurePerimeter, axisXRange =[500, 2500], axisYRange =[0, 45], color = color)
    # ASPECT RATIO,
    figAspectRatio = fn.plotHistogram(allData[:,5], title = "Histogram AspectRatio", figure = figureAspectRatio, axisXRange =[0.2, 1], color = color, axisYRange =[0, 45])
    # SOLIDITY,
    figSolidity = fn.plotHistogram(allData[:,6], title = "Histogram Solidity", figure = figureSolidity, axisXRange =[0.5, 1], color = color, axisYRange =[0, 45])
    # mean B values (color)
    figBmean = fn.plotHistogram(allData[:,9], title = "Histogram B mean Values", figure = figureBmean, axisXRange =[0, 128], axisYRange =[0, 45], color = color)
    # mean G values (color)
    figGmean = fn.plotHistogram(allData[:,10], title = "Histogram G mean Values", figure = figureGmean, axisXRange =[0, 128], axisYRange =[0, 45], color = color)
    # mean R values (color)
    figRmean = fn.plotHistogram(allData[:,11], title = "Histogram R mean Values", figure = figureRmean, axisXRange =[0, 128], axisYRange =[0, 45], color = color)
    # ANGLE ILUMINATION
    figAIlu = fn.plotHistogram(allData[:,16], bins = 10, title = "Histogram Angle Ilumination", figure = figureAIlu, color = color, axisXRange = [0, 360], axisYRange =[0, 150])
    # DISTANCE ILUMINATION
    figDIlu = fn.plotHistogram(allData[:,17], title = "Histogram Distance Ilumination", figure = figureDIlu, axisXRange =[0, 80], axisYRange =[0, 45], color = color)
    # DISTANCE/ANGLE ILUMINATION
    figDAIlu = fn.plotData(allData[:,16], allData[:,17], [0, 360], [0, 80], color, figureDAIlu, [4,4], "circle", "Angle / Distance Ilumination")

    return figPos, figAngle, figArea, figPerimeter, figAspectRatio, figSolidity, figBmean, figGmean, figRmean, figAIlu, figDIlu, figDAIlu


def xml_to_csv(pathlibPath):
    xml_list = []
    for xml_file in pathlibPath.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = DataFrame(xml_list, columns=column_name)
    print("Converting xml to cvs -> '{}'".format(pathlibPath))
    output_path = Path.resolve(pathlibPath / Path('all_labels.csv'))
    xml_df.to_csv(output_path, index=None)
    print('Successfully converted xml to csv.')
