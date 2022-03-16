from pyexpat import XML_PARAM_ENTITY_PARSING_UNLESS_STANDALONE
import pygame as pg
import time
import math
import numpy

pg.init()

def applyFriction(velocities, friction):
    magnitude = 0
    for velocity in velocities:
        magnitude += velocity**2
    magnitude = math.sqrt(magnitude)
    
    if magnitude != 0:
        ratios = []
        for velocity in velocities:
            ratios.append(velocity/magnitude)

        if magnitude > 0:
            if magnitude < friction:
                magnitude = 0
            else:
                if magnitude > 0:
                    magnitude -= friction
                else:
                    magnitude += friction
        
        newVelocities = []

        for ratio in ratios:
            newVelocities.append(ratio*magnitude)
    else:
        newVelocities = []
        for i in velocities:
            newVelocities.append(0)

    return(newVelocities)

def rollMatrix(theta):
    matrix = numpy.array([(math.cos(theta), -math.sin(theta), 0), (math.sin(theta), math.cos(theta), 0), (0,0,1)])
    return(matrix)

def yawMatrix(theta):
    matrix = numpy.array([(math.cos(theta), 0, math.sin(theta)), (0,1,0), (-math.sin(theta), 0, math.cos(theta))])
    return(matrix)

def pitchMatrix(theta):
    matrix = numpy.array([(1,0,0), (0, math.cos(theta), -math.sin(theta)), (0, math.sin(theta), math.cos(theta))])
    return(matrix)

def getYawPitchRoll(matrix):
    alpha = math.atan2(matrix[1][0], matrix[0][0])
    beta = math.atan2(-matrix[2][0], math.hypot(matrix[2][1], matrix[2][2]))
    gamma = math.atan2(matrix[2][1], matrix[2][2])
    return(alpha, beta, gamma)

def combineYawPitchRoll(yaw, pitch, roll):
    '''This function is broken and doesnt work right'''
    matrix1 = numpy.matmul(pitchMatrix(pitch), rollMatrix(roll))
    matrix2 = numpy.matmul(yawMatrix(yaw), matrix1)
    return(matrix2)

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

        if self.dz == 0:
            self.dz = -1

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

        self.xVel = 0
        self.yVel = 0
        self.zVel = 0

        self.yawVel = 0
        self.pitchVel = 0
        self.rollVel = 0

        self.rotationMatrix = numpy.identity(3)

        # These display values are the position of the display relative to the camera position and orientation (like a projector screen).
        self.displayX = displayX
        self.displayY = displayY
        self.displayZ = displayZ

    def setPosition(self, deltaX, deltaY, deltaZ):
        self.x += deltaX
        self.y += deltaY
        self.z += deltaZ

    def updatePosition(self):
        self.x += self.xVel
        self.y += self.yVel
        self.z += self.zVel
    
    def changeVelocity(self, deltaX, deltaY, deltaZ):
        xVel = Rotation3DMatrix((deltaX, 0, 0), numpy.transpose(self.rotationMatrix), False)
        yVel = Rotation3DMatrix((0, deltaY, 0), numpy.transpose(self.rotationMatrix), False)
        zVel = Rotation3DMatrix((0, 0, deltaZ), numpy.transpose(self.rotationMatrix), False)

        xVel, yVel, zVel = xVel + yVel + zVel
        
        self.xVel += xVel
        self.yVel += yVel
        self.zVel += zVel
    
    def changeRotationalVelocity(self, yawVelocity, pitchVelocity, rollVelocity):
        self.yawVel += yawVelocity
        self.pitchVel += pitchVelocity
        self.rollVel += rollVelocity
    
    def setVelocity(self, x, y, z):
        self.xVel = x
        self.yVel = y
        self.zVel = z

    def moveOrientation(self, matrix):
        self.rotationMatrix = numpy.matmul(matrix, self.rotationMatrix)
    
    def updateOrientation(self):
        self.rotationMatrix = numpy.matmul(yawMatrix(self.yawVel), self.rotationMatrix)
        self.rotationMatrix = numpy.matmul(pitchMatrix(self.pitchVel), self.rotationMatrix)
        self.rotationMatrix = numpy.matmul(rollMatrix(self.rollVel), self.rotationMatrix)

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
        for index, vertex in enumerate(self.vertices):    
            ax = vertex.x - self.center[0]
            ay = vertex.y - self.center[1]
            az = vertex.z - self.center[2]

            dx, dy, dz = Rotation3DMatrix((ax, ay, az), self.rotationMatrix, False)

            orientedVertices[index] = Vertex((dx + self.center[0], dy + self.center[1], dz + self.center[2]))

        return(orientedVertices)

class InfoUI:
    def __init__(self, screenSize, location):
        self.screenSize = screenSize
        self.center = (self.screenSize[0]/2, self.screenSize[1]/2)
        self.location = location
        self.surface = pg.Surface(self.screenSize)

    def drawBackground(self):
        self.surface.fill("black")
        pg.draw.rect(self.surface, "gray", (0,0,self.screenSize[0],self.screenSize[1]), 1, 1)
        
        pg.draw.line(self.surface, (50,50,50), (self.center[0], 40), (self.center[0], self.screenSize[1] - 40), 6)
        pg.draw.line(self.surface, (50,50,50), (40, self.center[1]), (self.screenSize[0] - 40, self.center[1]), 6)
        pg.draw.line(self.surface, (50,50,50), (self.screenSize[0]-20, 40), (self.screenSize[0]-20, self.screenSize[1]-40), 6)
    
    def drawVelocities(self, velocities):
        lengths = []
        for velocity in velocities:
            lengths.append(velocity*50)
        pg.draw.line(self.surface, (255,0,0), self.center, (self.center[0], self.center[1] - lengths[2]), 3)
        pg.draw.line(self.surface, (255,0,0), self.center, (self.center[0] + lengths[0], self.center[1]), 3)
        pg.draw.line(self.surface, (255,0,0), (self.screenSize[0]-20, self.center[1]), (self.screenSize[0]-20, self.center[1] - lengths[1]), 3)

    def update(self, velocities):
        self.drawBackground()
        self.drawVelocities(velocities)

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
        cube2.rotationMatrix = numpy.matmul(yawMatrix(math.cos(elapsedTime)/43), cube2.rotationMatrix)
        cube2.rotationMatrix = numpy.matmul(rollMatrix(math.sin(elapsedTime)/50), cube2.rotationMatrix)
        self.displayShape(cube2)
        self.displayShape(cube3)
        compass.update(numpy.matmul(self.currentCamera.rotationMatrix, (self.currentCamera.xVel, self.currentCamera.yVel, self.currentCamera.zVel)))
        self.screen.blit(compass.surface, compass.location)
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
        lastPressedKeys = []
        while True:
            time.sleep(0.001)
            elapsedTime = time.time() - startTime
            xInput = 0
            yInput = 0
            zInput = 0

            pitchInput = 0
            yawInput = 0
            rollInput = 0

            friction = 0
            rotationalFriction = 0
            if pg.key.get_pressed()[pg.K_v]:
                friction = 0.001
            if pg.key.get_pressed()[pg.K_b]:
                rotationalFriction = 0.001
            acceleration = 0.002
            rotationalAcceleration = 0.0005
            if pg.key.get_pressed()[pg.K_LCTRL]:
                acceleration *= 4
            if pg.key.get_pressed()[pg.K_c]:
                acceleration /= 4
                rotationalAcceleration /= 4
                friction = 0.0002
                rotationalFriction = 0.0001
            if pg.key.get_pressed()[pg.K_w]:
                zInput += acceleration
            if pg.key.get_pressed()[pg.K_s]:
                zInput -= acceleration
            if pg.key.get_pressed()[pg.K_a]:
                xInput -= acceleration
            if pg.key.get_pressed()[pg.K_d]:
                xInput += acceleration
            if pg.key.get_pressed()[pg.K_SPACE]:
                yInput += acceleration
            if pg.key.get_pressed()[pg.K_LSHIFT]:
                yInput -= acceleration

            if pg.key.get_pressed()[pg.K_LEFT]:
                yawInput += rotationalAcceleration
            if pg.key.get_pressed()[pg.K_RIGHT]:
                yawInput -= rotationalAcceleration
            if pg.key.get_pressed()[pg.K_UP]:
                pitchInput += rotationalAcceleration
            if pg.key.get_pressed()[pg.K_DOWN]:
                pitchInput -= rotationalAcceleration
            if pg.key.get_pressed()[pg.K_q]:
                rollInput -= rotationalAcceleration
            if pg.key.get_pressed()[pg.K_e]:
                rollInput += rotationalAcceleration
            
            if pg.key.get_pressed()[pg.K_r]:
                self.currentCamera.rotationMatrix = numpy.identity(3)
                self.currentCamera.x = 0
                self.currentCamera.y = 0
                self.currentCamera.z = 0

            self.currentCamera.yawVel, self.currentCamera.pitchVel, self.currentCamera.rollVel = applyFriction((self.currentCamera.yawVel, self.currentCamera.pitchVel, self.currentCamera.rollVel), rotationalFriction)
            self.currentCamera.changeRotationalVelocity(yawInput, pitchInput, rollInput)
            self.currentCamera.updateOrientation()
            
            self.currentCamera.xVel, self.currentCamera.yVel, self.currentCamera.zVel = applyFriction((self.currentCamera.xVel, self.currentCamera.yVel, self.currentCamera.zVel), friction)
            self.currentCamera.changeVelocity(xInput, yInput, zInput)
            self.currentCamera.updatePosition()
            
            self.updateCanvas(elapsedTime)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    exit()

cube1 = Shape(RectangularPrism((-3,-1,2),(3,1,8)).getShape(), (0,0,0))
cube2 = Shape(RectangularPrism((-1,1,4),(1,3,6)).getShape(), (0,0,math.pi/4))
cube3 = Shape(RectangularPrism((-100,-100,-100),(100,100,100)).getShape(), (0,0,0))

screen1 = Screen(Camera(0,0,0,0,0,0,0,0,0))
compass = InfoUI((200,200), (20, screen1.screenSize[1] - 220))

screen1.start()