import pygame as pg
import time
import math
import numpy

pg.init()

def rollMatrix(theta):
    matrix = numpy.array([(math.cos(theta), -math.sin(theta), 0), (math.sin(theta), math.cos(theta), 0), (0,0,1)])
    return(matrix)

def yawMatrix(theta):
    matrix = numpy.array([(math.cos(theta), 0, math.sin(theta)), (0,1,0), (-math.sin(theta), 0, math.cos(theta))])
    return(matrix)

def pitchMatrix(theta):
    matrix = numpy.array([(1,0,0), (0, math.cos(theta), -math.sin(theta)), (0, math.sin(theta), math.cos(theta))])
    return(matrix)
'''
def Rotation3D(position, orientation, isCamera):
    ax, ay, az = position
    tx, ty, tz = orientation
    if isCamera:
        inverted = -1
    else:
        inverted = 1
    startMatrix = numpy.array([ax,ay,az])
    zRotationMatrix = numpy.array([(math.cos(tz), -inverted*math.sin(tz), 0), (inverted*math.sin(tz), math.cos(tz), 0), (0,0,1)])
    yRotationMatrix = numpy.array([(math.cos(ty), 0, inverted*math.sin(ty)), (0,1,0), (-inverted*math.sin(ty), 0, math.cos(ty))])
    xRotationMatrix = numpy.array([(1,0,0), (0, math.cos(tx), -inverted*math.sin(tx)), (0, inverted*math.sin(tx), math.cos(tx))])

    zMatrix = numpy.matmul(startMatrix, zRotationMatrix)
    yzMatrix = numpy.matmul(zMatrix, yRotationMatrix)
    xyzMatrix = numpy.matmul(yzMatrix, xRotationMatrix)

    dx, dy, dz = xyzMatrix

    return((dx, dy, dz))
'''
def Rotation3DMatrix(position, orientation, isInverted):
    ax, ay, az = position
    rotationMatrix = orientation
    finalMatrix = numpy.matmul(rotationMatrix, numpy.array([ax,ay,az]))
    if isInverted:
        finalMatrix = numpy.transpose(finalMatrix)
    return(finalMatrix)

class Vertex:
    def __init__(self, position):
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        self.dx, self.dy, self.dz = (0,0,0)

    def updateCameraPerspective(self, camera):
        ax = self.x - camera.x
        ay = self.y - camera.y
        az = self.z - camera.z

        startMatrix = numpy.array([ax,ay,az])
        finalMatrix = numpy.matmul(camera.rotationMatrix, startMatrix)

        self.dx = finalMatrix[0]
        self.dy = finalMatrix[1]
        self.dz = finalMatrix[2]

    def getProjectedPosition(self, camera):
        self.updateCameraPerspective(camera)

        ex = camera.displayX
        ey = camera.displayY
        ez = camera.displayZ

        x = ((ez / self.dz) * self.dx) + ex
        y = ((ez / self.dz) * self.dy) + ey

        if abs(x) == math.inf or abs(y) == math.inf:
            x = None
            y = None

        return((x,y))

    def getProjectedPosition2(self, camera, position):
        dx, dy, dz = position

        ex = camera.displayX
        ey = camera.displayY
        ez = camera.displayZ

        x = ((ez / dz) * dx) + ex
        y = ((ez / dz) * dy) + ey

        if abs(x) == math.inf or abs(y) == math.inf:
            x = None
            y = None

        return((x,y))

class Camera:
    def __init__(self, x, y, z, thetaX, thetaY, thetaZ, displayX, displayY, displayZ):
        self.x = x
        self.y = y
        self.z = z

        self.rotationMatrix = numpy.identity(3)

        # These display values are the position of the display relative to the camera position and orientation (like a projector screen).
        self.displayX = displayX
        self.displayY = displayY
        self.displayZ = displayZ

    def movePosition(self, deltaX, deltaY, deltaZ):
        self.x += deltaX
        self.y += deltaY
        self.z += deltaZ

    def moveOrientation(self, matrix):
        self.rotationMatrix = numpy.matmul(matrix, self.rotationMatrix)

class RectangularPrism:
    def __init__(self, corner, oppositeCorner):
        x1, y1, z1 = corner
        x2, y2, z2 = oppositeCorner
        self.vertices = [(x1,y1,z1),(x1,y1,z2),(x1,y2,z1),(x1,y2,z2),
                         (x2,y1,z1),(x2,y1,z2),(x2,y2,z1),(x2,y2,z2)]
        self.edges = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)] # These are the indexes for the list of vertices

    def getShape(self):
        return(self.vertices, self.edges)

class Shape:
    def __init__(self, baseShape, orientation):
        '''The baseShape is a tuple containing the vertices and the edges. The vertices are a list of tuples, each containing x, y, and z. 
        The edges are a list of tuples, each containing 2 indices to reference the list of vertices. The tuple of indices can turn into a tuple of vertices, which is how a line is drawn.
        The orientation is a tuple containing the angles for x rotation, y rotation, and z rotation. These are Tait-Bryan angles for orientation in 3D space.'''
        vertices = baseShape[0]
        self.edges = baseShape[1]
        self.thetaX, self.thetaY, self.thetaZ = orientation
        yzMatrix = numpy.matmul(pitchMatrix(self.thetaY), rollMatrix(self.thetaZ))
        self.rotationMatrix = numpy.matmul(yawMatrix(self.thetaX), yzMatrix)

        for index, vertex in enumerate(vertices):
            vertices[index] = Vertex(vertex)
        self.vertices = vertices

        self.center = [0,0,0]
        self.updateCenter()

    def scale(self, scaleFactor):
        self.updateCenter()
        for index, vertex in enumerate(self.vertices):
            x = (scaleFactor - 1) * (vertex.x - self.center[0])
            y = (scaleFactor - 1) * (vertex.y - self.center[1])
            z = (scaleFactor - 1) * (vertex.z - self.center[2])
            self.vertices[index][1].x += x
            self.vertices[index][1].y += y
            self.vertices[index][1].z += z

    def updateCenter(self):
        self.center = [0,0,0]
        for vertex in self.vertices:
            self.center[0] += vertex.x
            self.center[1] += vertex.y
            self.center[2] += vertex.z
        self.center = tuple((1/len(self.vertices))*x for x in self.center)

    def getOrientedVertices(self):
        orientedVertices = [0]*len(self.vertices)
        tx = self.thetaX
        ty = self.thetaY
        tz = self.thetaZ
        for index, vertex in enumerate(self.vertices):    
            ax = vertex.x - self.center[0]
            ay = vertex.y - self.center[1]
            az = vertex.z - self.center[2]

            dx, dy, dz = Rotation3DMatrix((ax, ay, az), self.rotationMatrix, False)

            orientedVertices[index] = Vertex((dx + self.center[0], dy + self.center[1], dz + self.center[2]))

        return(orientedVertices)

class Screen:
    def __init__(self, camera):
        self.screenSize = (1280, 720)
        self.screen = pg.display.set_mode(self.screenSize)
        self.background = pg.image.load("minecraft_void_background.jpg")
        self.background = pg.transform.scale(self.background, self.screenSize)
        self.screen.blit(self.background, (0,0))
        self.currentCamera = camera
        self.currentCamera.displayZ = 500

    def cartesianConvertToDisplay(self, position):
        x = position[0]
        y = position[1]
        if x != None and y != None:
            y = -y
            x += self.screenSize[0]/2
            y += self.screenSize[1]/2
            return((x,y))

    def updateCanvas(self, elapsedTime):
        self.screen.blit(self.background, (0,0))
        self.displayShape(cube1)
        self.displayShape(cube2)
        pg.display.update()

    def displayShape(self, shape):
        shapeSurface = pg.Surface(self.screenSize, pg.SRCALPHA)
        for index, edge in enumerate(shape.edges):
            vertex1 = shape.getOrientedVertices()[edge[0]]
            vertex2 = shape.getOrientedVertices()[edge[1]]
            point1 = self.cartesianConvertToDisplay(vertex1.getProjectedPosition(self.currentCamera))
            point2 = self.cartesianConvertToDisplay(vertex2.getProjectedPosition(self.currentCamera))

            if vertex1.dz > 0 and vertex2.dz > 0:
                if index == 0:
                    pg.draw.line(shapeSurface, (255,255,255), point1, point2, 1)
                else:
                    pg.draw.line(shapeSurface, (255,255-23*index,23*index), point1, point2, 1)

        self.screen.blit(shapeSurface, (0,0))

    def start(self):
        startTime = time.time()
        while True:
            time.sleep(0.001)
            elapsedTime = time.time() - startTime
            xInput = 0
            yInput = 0
            zInput = 0
            if pg.key.get_pressed()[pg.K_w]:
                zInput += 0.05
            if pg.key.get_pressed()[pg.K_a]:
                xInput -= 0.05
            if pg.key.get_pressed()[pg.K_s]:
                zInput -= 0.05
            if pg.key.get_pressed()[pg.K_d]:
                xInput += 0.05
            if pg.key.get_pressed()[pg.K_SPACE]:
                yInput += 0.05
            if pg.key.get_pressed()[pg.K_LSHIFT]:
                yInput -= 0.05
            if pg.key.get_pressed()[pg.K_LEFT]:
                self.currentCamera.moveOrientation(yawMatrix(0.02))
            if pg.key.get_pressed()[pg.K_RIGHT]:
                self.currentCamera.moveOrientation(yawMatrix(-0.02))
            if pg.key.get_pressed()[pg.K_UP]:
                self.currentCamera.moveOrientation(pitchMatrix(0.02))
            if pg.key.get_pressed()[pg.K_DOWN]:
                self.currentCamera.moveOrientation(pitchMatrix(-0.02))
            if pg.key.get_pressed()[pg.K_q]:
                self.currentCamera.moveOrientation(rollMatrix(-0.02))
            if pg.key.get_pressed()[pg.K_e]:
                self.currentCamera.moveOrientation(rollMatrix(0.02))
            if pg.key.get_pressed()[pg.K_r]:
                self.currentCamera.rotationMatrix = numpy.identity(3)

            inputX = Rotation3DMatrix((xInput, 0, 0), numpy.transpose(self.currentCamera.rotationMatrix), False)
            inputY = Rotation3DMatrix((0, yInput, 0), numpy.transpose(self.currentCamera.rotationMatrix), False)
            inputZ = Rotation3DMatrix((0, 0, zInput), numpy.transpose(self.currentCamera.rotationMatrix), False)

            inputX, inputY, inputZ = inputX + inputY + inputZ

            self.currentCamera.x += inputX
            self.currentCamera.y += inputY
            self.currentCamera.z += inputZ

            self.updateCanvas(elapsedTime)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit()

cube1 = Shape(RectangularPrism((-3,-1,2),(3,1,8)).getShape(), (0,0,0))
cube2 = Shape(RectangularPrism((-1,1,4),(1,3,6)).getShape(), (0,0,math.pi/4))

screen1 = Screen(Camera(0,0,0,0,0,0,0,0,0))

screen1.start()