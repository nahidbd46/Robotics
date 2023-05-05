import math
import pygame

class Envo:
    def __init__(self,dimensions):
        self.black = (0,0,0)
        self.white = (255,255,255)
        self.green = (0,255,0)
        self.blue = (0,0,255)
        self.red = (255,0,0)
        self.yellow = (255,255,0)

        self.length = dimensions[0]
        self.width = dimensions[1]

        pygame.display.set_caption("Differential Drive Robot")
        self.win = pygame.display.set_mode((self.length,self.width))

        self.font = pygame.font.Font('freesansbold.ttf',35)
        self.text = self.font.render('default',True,self.white,self.black)
        self.textrect = self.text.get_rect()
        self.textrect.center=(dimensions[1]-175, dimensions[0]-500)
        self.traj_set=[]

    def write (self, velL, velR, theta):
        txt = f"Vel L = {velL/3779.52:.3f} Vel R ={velR/3779.52:.3f} theta = {int(math.degrees(theta))}"
        self.text=self.font.render(txt,True,self.white,self.black)
        self.win.blit(self.text,self.textrect)

    def trajectory(self,pos):
        for i in range(0,len(self.traj_set)-1):
            pygame.draw.line(self.win,self.red,(self.traj_set[i][0],self.traj_set[i][1]),(self.traj_set[i+1][0],self.traj_set[i+1][1]))
        if self.traj_set.__sizeof__()>300000:
            self.traj_set.pop(0)
        self.traj_set.append(pos)


class Robot:
    def __init__(self,startpoint, robotimg, width):
        self. met2pix = 3779.52
        self.wd= width
        self.x= startpoint[0]
        self.y=startpoint[1]
        self.theta=0
        self.velL=0.01 * self.met2pix
        self.velR=0.01 * self.met2pix
        self.max =0.02* self.met2pix
        self.min = 0.005* self.met2pix

        self.imge=pygame.image.load(robotimg)
        self.rotate=self.imge
        self.rect=self.rotate.get_rect(center=(self.x,self.y))

    def draw(self,win):
        win.blit(self.rotate,self.rect)

    def press(self, event= None):
        if event is not None:

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_KP1:
                    self.velL+= 0.001* self.met2pix
                elif event.key == pygame.K_KP2:
                    self.velL-= 0.001* self.met2pix
                elif event.key == pygame.K_KP3:
                    self.velR+= 0.001* self.met2pix
                elif event.key == pygame.K_KP4:
                    self.velR-= 0.001* self.met2pix

        self.x += ((self.velL+ self.velR)/2) * math.cos(self.theta)*dt
        self.y -= ((self.velL+ self.velR)/2) * math.sin(self.theta)*dt
        self.theta += (self.velR-self.velL)/self.wd*dt

        self.rotate = pygame.transform.rotozoom(self.imge,math.degrees(self.theta),1)
        self.rect = self.rotate.get_rect(center= (self.x,self.y))


pygame.init()

start=(400,200)
rob = Robot(start,r"C:\Users\Nehid\PycharmProjects\Robotics\Robot.png", 0.01*3779.52)
dim= (1200,800)
env = Envo(dim)
dt= 0
prevtime= pygame.time.get_ticks()

run= True

while run:
    /*=for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        rob.press(event)
    dt = (pygame.time.get_ticks()- prevtime)/1000
    prevtime = pygame.time.get_ticks()
    pygame.display.update()
    env.win.fill(env.white)
    rob.press()
    rob.draw(env.win)
    env.trajectory((rob.x,rob.y))
    env.write(int(rob.velL),int(rob.velR),int(rob.theta))












