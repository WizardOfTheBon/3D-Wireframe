from PIL import Image

image1 = Image.open("minecraft_void_background.jpg")

rgbValues = []

for x in range(image1.size[0]):
    for y in range(image1.size[1]):
        rgbValues.append(image1.getpixel((x,y)))

for index, value in enumerate(rgbValues):
    r = value[0]
    g = value[1]
    b = value[2]

    brightness = int((r+g+b)/(3))
    maximum = max(r,g,b)
    minimum = min(r,g,b)
    if maximum != 0:
        saturation = (maximum-minimum)/(maximum+minimum)
    else:
        saturation = 0

    rgbValues[index] = (brightness, saturation)

i = 0
for x in range(image1.size[0]):
    for y in range(image1.size[1]):
        image1.putpixel((x,y),(rgbValues[i][0], int(rgbValues[i][0]*(1-rgbValues[i][1])), rgbValues[i][0]))
        i += 1

image1.save('alternative_background5.jpg')