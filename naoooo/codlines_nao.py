import random
import time
from naoqi import ALProxy #importar librerias de nao
'''import sys
print("Ejecutando desde Python 2.7")
print("Argumentos:", sys.argv)'''

#Simulador
'''IP="10.190.103.36"  #IP Nao
PROXY=51940    #Port del NAO'''

#Nyx
IP="127.0.0.1"  #IP Nao
PROXY=59242    #Port del NAO'''

#Zyra
'''IP="192.168.108.90"  #IP Nao
PROXY=9559    #Port del NAO'''

tts = ALProxy("ALTextToSpeech", IP, PROXY)
postura = ALProxy("ALRobotPosture", IP, PROXY)
motion = ALProxy("ALMotion", IP, PROXY)

a = random.uniform(0,1)
b = random.uniform(0,1)

names = ["LShoulderRoll","RShoulderPitch"]
times = [[1],[1]]

#motion.moveInit()
start = time.time()
postura.goToPosture("StandInit", 0.7)
end = time.time()
tiempo_execution = end-start
print(tiempo_execution)
start = time.time()
motion.angleInterpolation(names,[a,0.0],times,True)
motion.angleInterpolation(names,[a,b],times,True)
tts.say("Hola")
end = time.time()

tiempo_execution = end-start
print(tiempo_execution)
#postura.goToPosture("StandInit", 2.0)
#motion.moveInit()
#motion.post.moveTo(0.2,0.0,0.0)

#time.sleep(3)
#tts.say("hola mundo")