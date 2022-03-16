from PIL import Image
import time
import copy

image1 = Image.open("minecraft_void_background.jpg")

rgbValues = []

for x in range(image1.size[0]):
    for y in range(image1.size[1]):
        rgbValues.append(image1.getpixel((x,y)))

for index, value in enumerate(rgbValues):
    r = value[0]
    g = value[1]
    b = value[2]

    luminance = int((r+g+b)/(3))
    maximum = max(r,g,b)
    minimum = min(r,g,b)
    if maximum != 0:
        saturation = (maximum-minimum)/(maximum+minimum)
    else:
        saturation = 0
    
    maximum /= 255
    minimum /= 255
    r /= 255
    g /= 255
    b /= 255

    if saturation != 0:
        if maximum == r:
            hue = (g-b)/(maximum-minimum)
        elif maximum == g:
            hue = 2 + (b-r)/(maximum-minimum)
        elif maximum == b:
            hue = 4 + (r-g)/(maximum-minimum)
    else:
        hue = 0
    hue *= 60

    hue /= 360
    luminance /= 255

    rgbValues[index] = (hue, saturation, luminance)

    def getSpecificColor(temporaryColor, temporary1, temporary2):
        if 6*temporaryColor < 1:
            newColor = temporary2 + (temporary1 - temporary2) * 6 * temporaryColor
        elif 2*temporaryColor < 1:
            newColor = temporary1
        elif 3*temporaryColor < 2:
            newColor = temporary2 + (temporary1 - temporary2) * (0.666 - temporaryColor) * 6
        else:
            newColor = temporary2
        return(newColor)

    def getRGB(hue, saturation, luminance):
        if luminance < 0.5:
            temp1 = luminance * (1 + saturation)
        elif luminance >= 0.5:
            temp1 = luminance + saturation - (luminance * saturation)
        temp2 = 2 * luminance - temp1

        tempR = hue + 0.333
        if tempR > 1:
            tempR -= 1
        tempG = hue
        tempB = hue - 0.333
        if tempB < 0:
            tempB += 1

        r = getSpecificColor(tempR, temp1, temp2)
        g = getSpecificColor(tempG, temp1, temp2)
        b = getSpecificColor(tempB, temp1, temp2)
        return(r,g,b)

imageList = []
for k in range(30):
    i = 0
    for x in range(image1.size[0]):
        for y in range(image1.size[1]):
            hue, saturation, luminance = rgbValues[i]
            hue += k/30
            hue %= 1
            R, G, B = getRGB(hue, saturation, luminance)

            R = int(R * 255)
            G = int(G * 255)
            B = int(B * 255)

            image1.putpixel((x,y),(R,G,B))
            i += 1
    copyIm = copy.deepcopy(image1)
    imageList.append(copyIm)

imageList[0].save('hue_shift.gif', save_all=True, append_images=imageList[1:], optimize=False, duration=100, loop=0)
