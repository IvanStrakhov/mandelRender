from PIL import Image, ImageDraw, ImageFont
import math
from math import *
import numpy as np
from timeit import default_timer as timer
from numba import jit,cuda
from itertools import product
import multiprocessing
import multiprocessing.pool
from threading import Thread
from console_progressbar import ProgressBar
from multiprocessing import Pool
import functools

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class mandelRender():
	'''
	The mandelRender is for creating mandelbrot set images

	Parameters
	----------
	
	Attributes
	----------
	
	Methods
	----------
	setScale
	getScale
	setImageSize
	getImageSize
	setCenter
	getCenter
	setPngSaveName	
	getPngSaveName	
	setGifSaveName	
	getGifSaveName	
	setNumOfFrames		
	getNumOfFrames
	getGamma		
	setGamma	
	render
	renderGIF
	'''
	
	def __init__(self):
		self.xcent=-1.4801308697
		self.ycent=-0.001092450
		
		
		#xcent=0.13000002968
		#ycent=-0.59050002976
		#xcent=-1.47983996
		#ycent=-0.0006001
		#xcent=-1.45986996
		#ycent=-0.0
    	
    	
		#nitr=1500.
    	
		self.frin=0.0001
		
    	
		self.nitr=np.float64(-115.592077184*log(self.frin)+207.219973)
	
	    
		self.axx=600
		self.ayy=600
		self.name = "mandelbrot"
		self.gifName = "gifmandelbrot"
		self.gamma="sunset"
		self.numOfFrames = 10
	    
		self.p_lin=np.linspace(self.xcent-self.axx/self.ayy*self.frin,self.axx/self.ayy*self.frin+self.xcent, self.axx)
		self.q_lin=np.linspace(self.ycent-self.frin,self.ycent+self.frin, self.ayy)
		
		self.percent=0
		self.frameNo=0
        
		self.image = Image.new('RGBA', (self.axx, self.ayy), (255,0,0))
		self.draw = ImageDraw.Draw(self.image) 
		self.pb = ProgressBar(total=100,prefix='', suffix='', decimals=3, length=50, fill='X', zfill='-')
		

	def setImageSize(self,x_size,y_size):
		"""sets size of image
		
		size of image in pixels
				
		Parameters:
		-----------
		x_size : int
			Positive number
		y_size : int
			Positive number
	    
		Returns:
		-------
		None
	    
		"""
		self.axx=x_size
		self.ayy=y_size
		self.image = Image.new('RGBA', (self.axx, self.ayy), (255,0,0))
		self.draw = ImageDraw.Draw(self.image) 
		self.p_lin=np.linspace(self.xcent-self.axx/self.ayy*self.frin,self.axx/self.ayy*self.frin+self.xcent, self.axx)
		self.q_lin=np.linspace(self.ycent-self.frin,self.ycent+self.frin, self.ayy)
		
	def getImageSize(self):
		return (self.axx, self.ayy)
		
	def setCenter(self,xcent,ycent):
		"""sets center of image
		
		center of image will be at this point (x,y)
				
		Parameters:
		-----------
		xcent : float
			Positive number
		ycent : float
			Positive number
	    
		Returns:
		-------
		None
		
		Advices:
		(xcent=-1.4801308697,ycent=-0.001092450)
		(xcent=0.13000002968,ycent=-0.59050002976)
		(xcent=-1.47983996, ycent=-0.0006001)
		(xcent=-1.45986996, ycent=-0.0)
	    
		"""
		self.xcent=xcent
		self.ycent=ycent
		self.p_lin=np.linspace(self.xcent-self.axx/self.ayy*self.frin,self.axx/self.ayy*self.frin+self.xcent, self.axx)
		self.q_lin=np.linspace(self.ycent-self.frin,self.ycent+self.frin, self.ayy)
		self.nitr=np.float64(-115.592077184*log(self.frin)+207.219973)
	def getCenter(self):
		"""sets center of image
		
		center of image will be at this point (x,y)
				
		Parameters:
		-----------
		xcent : float
			Positive number
		ycent : float
			Positive number
	    
		float: 
			xcent, first parameter
		float : 
			ycent, second parameter
        
    
	    
		"""
		return (self.xcent,self.ycent)
	def setScale(self,scale):
		"""sets scale of image
		
		image pixels will be inside of frame with half of side equal to this scale
				
		Parameters:
		-----------
		scale : float
			Positive number, 1.0e-14 < scale < 6.0
	    
		Returns:
		-------
		None
	    
		"""
	    
		self.frin=scale
		if self.frin<9.e-15:
				print("To small value;")
				self.frin=9.e-15
				#self.frin=np.longdouble(scale)
		self.p_lin=np.linspace(self.xcent-self.axx/self.ayy*self.frin,self.axx/self.ayy*self.frin+self.xcent, self.axx)
		self.q_lin=np.linspace(self.ycent-self.frin,self.ycent+self.frin, self.ayy)
		self.nitr=np.float64(-115.592077184*log(self.frin)+207.219973)
		
	def getScale(self):
		"""get scale of image
	    
	    gets scale of image
	    image pixels will be inside of frame with half of side equal to this scale
	    
	    Returns:
	    -------
	    float
	        
	    
	    """
		return (self.frin)
		
	def setPngSaveName(self,string):
		"""set prefix for png image
		
		string prefix for png image
				
		Parameters:
		-----------
		string : str
			
		
		Returns:
		-------
		None
	    
		"""
		self.name=string
		
	def getPngSaveName(self):
		return self.name
		
	def setGifSaveName(self,string):
		"""set prefix for gif image
		
		string prefix for gif image
				
		Parameters:
		-----------
		string : str
			
		
		Returns:
		-------
		None
	    
		"""
		self.gifName=string
		
	def getGifSaveName(self):
		return self.gifName
		
	def setNumOfFrames(self,number):
		"""set number of frames
		
		number of frames in GIF animation
				
		Parameters:
		-----------
		number : int
			Positive number
			
		
		Returns:
		-------
		None
	    
		"""
		self.numOfFrames = number
		
	def getNumOfFrames(self):
		return self.numOfFrames
	def getGamma(self):
		return self.gamma
		
	def setGamma(self,string):
		"""set gamma for image
		
		color palette for rendering image
				
		
		Parameters:
		-----------
		string : str
			name of gamma, possible values: "sunset", "acid","azure","bloody grass","bw"
			
		
		Returns:
		-------
		None
	    
		"""
		if string=="sunset" or string=="acid" or string=="azure" or string=="bloody grass" or string=="bw" or string=="exper":
			self.gamma=string
		else:
			print("wrong gamma")
			return
		
		
	def render(self):
		"""start rendering of image
		
		this rendering takes time
				
		
		Parameters:
		-----------
		None
			
		
		Returns:
		-------
		None
	    
		"""
		start=timer()
		#lam=lambda x,y: ycycle(x,y,axx,ayy,frin,nitr,xcent,ycent,draw)
	
		#list(map(lam,p_mesh,q_mesh))
		self.frameNo=0
		frames = self.numOfFrames
        
		self.numOfFrames=3
		print("Image size: ",self.axx,self.ayy)
		print("Scale: ",self.frin)
		print("Center: ",self.getCenter())
		print("Gamma: ", self.getGamma())
		print("Name: ",self.name+'.png')
		
		self.pb = ProgressBar(total=100,prefix='Frames: 1/1', suffix='', decimals=3, length=50, fill='X', zfill='-')
		self.percent=0
		
		
		copier = functools.partial(drawcycle, nitr = self.nitr,axx=self.axx,ayy=self.ayy,frin=self.frin,xcent=self.xcent,ycent=self.ycent,gamma=self.gamma)
		self.pb.print_progress_bar(self.percent)
		
		result = list(map(copier,list(product(self.p_lin[:int(1/3*len(self.p_lin))], self.q_lin[:int(1/3*len(self.p_lin))]))))
		for i in result:
			self.draw.rectangle(((i[0],i[1]),(i[0]+1,i[1]-1)),fill = (i[2],i[3],i[4]))
		self.percent = 11
		
		self.pb.print_progress_bar(self.percent)
		result = list(map(copier,list(product(self.p_lin[int(1/3*len(self.p_lin)):int(2/3*len(self.p_lin))], self.q_lin[:int(1/3*len(self.p_lin))]))))
		for i in result:
			self.draw.rectangle(((i[0],i[1]),(i[0]+1,i[1]-1)),fill = (i[2],i[3],i[4]))
		self.percent = 22
		
		self.pb.print_progress_bar(self.percent)
		result = list(map(copier,list(product(self.p_lin[int(2/3*len(self.p_lin)):], self.q_lin[:int(1/3*len(self.p_lin))]))))
		for i in result:
			self.draw.rectangle(((i[0],i[1]),(i[0]+1,i[1]-1)),fill = (i[2],i[3],i[4]))
		self.percent = 33
		
		self.pb.print_progress_bar(self.percent)
		
		
		result = list(map(copier,list(product(self.p_lin[:int(1/3*len(self.p_lin))], self.q_lin[int(1/3*len(self.p_lin)):int(2/3*len(self.p_lin))]))))
		for i in result:
			self.draw.rectangle(((i[0],i[1]),(i[0]+1,i[1]-1)),fill = (i[2],i[3],i[4]))
		self.percent = 44
		
		self.pb.print_progress_bar(self.percent)
		
		result = list(map(copier,list(product(self.p_lin[int(1/3*len(self.p_lin)):int(2/3*len(self.p_lin))], self.q_lin[int(1/3*len(self.p_lin)):int(2/3*len(self.p_lin))]))))
		for i in result:
			self.draw.rectangle(((i[0],i[1]),(i[0]+1,i[1]-1)),fill = (i[2],i[3],i[4]))
		self.percent = 55
		
		self.pb.print_progress_bar(self.percent)
		
		result = list(map(copier,list(product(self.p_lin[int(2/3*len(self.p_lin)):], self.q_lin[int(1/3*len(self.p_lin)):int(2/3*len(self.p_lin))]))))
		for i in result:
			self.draw.rectangle(((i[0],i[1]),(i[0]+1,i[1]-1)),fill = (i[2],i[3],i[4]))
		
		self.percent = 66
		
		self.pb.print_progress_bar(self.percent)
		
		result = list(map(copier,list(product(self.p_lin[:int(1/3*len(self.p_lin))], self.q_lin[int(2/3*len(self.p_lin)):]))))		
		for i in result:
			self.draw.rectangle(((i[0],i[1]),(i[0]+1,i[1]-1)),fill = (i[2],i[3],i[4]))
		self.percent = 77
		
		self.pb.print_progress_bar(self.percent)
		
		
		result = list(map(copier,list(product(self.p_lin[int(1/3*len(self.p_lin)):int(2/3*len(self.p_lin))], self.q_lin[int(2/3*len(self.p_lin)):]))))		
		for i in result:
			self.draw.rectangle(((i[0],i[1]),(i[0]+1,i[1]-1)),fill = (i[2],i[3],i[4]))
		self.percent = 88
		
		self.pb.print_progress_bar(self.percent)
		
		
		result = list(map(copier,list(product(self.p_lin[int(2/3*len(self.p_lin)):], self.q_lin[int(2/3*len(self.p_lin)):]))))
		for i in result:
			self.draw.rectangle(((i[0],i[1]),(i[0]+1,i[1]-1)),fill = (i[2],i[3],i[4]))
		self.percent = 100
		
		self.pb.print_progress_bar(self.percent)
		
		
		
		#list(map(self._drawcycle,list(product(self.p_lin, self.q_lin))))
        
		self.numOfFrames=frames
		print("Time: ",timer()-start)
        
		self.image.save(self.name+'.png')
		#self.image.show()
		
		
		fig = plt.figure()

		img=mpimg.imread(self.name+'.png')
		
		def onclick(event):
			if event.button==1:
				ix, iy = event.xdata, event.ydata
				print("previous center: ",self.getCenter())
			
				self.setCenter(self.p_lin[int(ix)], self.q_lin[-int(iy)])
				print("current center: ",self.getCenter())
				
				self.percent=0
			if event.button==2:
				print("previous scale: ",self.getScale())
				self.setScale(self.frin*5)
				print("current scale: ",self.getScale())
			if event.button==3:
				print("previous scale: ",self.getScale())
				self.setScale(self.frin/5)
				print("current scale: ",self.getScale())
			
				
	
		cid = fig.canvas.mpl_connect('button_press_event', onclick)

		imgplot = plt.imshow(img)
		plt.show()
		
		self.percent=0
	
	def renderGIF(self):
		"""start rendering of multiple images for gif animation
		
		this rendering takes a lot time
				
		
		Parameters:
		-----------
		None
			
		
		Returns:
		-------
		None
	    
		"""
		frin_ren=self.frin
		nitr_ren=self.nitr
		p_lin_ren=self.p_lin
		q_lin_ren=self.q_lin
		arr=np.linspace(30,360.0, self.numOfFrames)
		self.frameNo=0
		self.percent=0
		for i in arr:
			self.frin=np.float64(10.*(((i)*0.05)**(-(i)*0.05)+exp(-(i)**0.6)))
			
			#if self.frin<9.e-15:
		#		self.frin=np.longdouble(10.*(((i)*0.05)**(-(i)*0.05)+exp(-(i)**0.6)))
			
			self.nitr=np.float64(-115.592077184*log(self.frin)+207.219973)
			#if self.frin<9.e-15:
		#		self.nitr=np.longdouble(-115.592077184*log(self.frin)+207.219973)
				
			
	
			self.p_lin=np.linspace(self.xcent-self.axx/self.ayy*self.frin,self.axx/self.ayy*self.frin+self.xcent, self.axx)
			self.q_lin=np.linspace(self.ycent-self.frin,self.ycent+self.frin, self.ayy)
			
			self.image = Image.new('RGBA', (self.axx, self.ayy), (255,0,0))  
			self.draw = ImageDraw.Draw(self.image) 
			
			list(map(self._drawcycle,list(product(self.p_lin, self.q_lin))))
			self.percent=0
			
			if self.frameNo==0:
				lis=[self.image,self.image]
				gif = self.image
			else:
				lis.append(self.image)
				
			
			self.frameNo=self.frameNo+1
			
		gif.save(self.gifName+'.gif', save_all=True, append_images=lis,loop=0, duration=100, optimize=False)
		

     
		self.frameNo=0
		
		
		self.frin=frin_ren
		self.nitr=nitr_ren
		self.p_lin=p_lin_ren
		self.q_lin=q_lin_ren
		
		

	 
	
@jit
def drawcycle(x,nitr,axx,ayy,frin,xcent,ycent,gamma):
	p=x[0]
	q=x[1]
	c1=(p+axx/ayy*frin-xcent)*(axx/2.*ayy/axx/frin)
	c2=(q+frin-ycent)*(ayy/2./frin)
	axey=cycle(p,q,nitr)
	rectangle=(0,0, 0, 0, 0)
	if (axey[1]>=10.):
		'''while (dim[n]<itr):
			n=n+1'''
		it=(axey[0]+8.5*exp(-sqrt(axey[1]))+8.5*exp(-((axey[1])**0.09)))/nitr
		#it = 0.01*(axey[0] - np.log2(np.log2(axey[1])))
		if gamma=="exper":
			color = (1.0 - 0.01*(axey[0] - np.log2(np.log2(axey[1]))))
			rectangle=(c1,ayy-c2, int(color*255),int(color*255),int(color*255))
		if gamma=="sunset":
			rectangle=(c1,ayy-c2,int((7*gaussian(it,0.76,0.035)+24*gaussian(it,0.3,0.09)+2*gaussian(it,0.915,0.015))*255./40.),int((12*gaussian(it,0.3,0.12)+5*gaussian(it,0.77,0.05)+2*gaussian(it,0.92,0.02))*255./40.),int((2*gaussian(it,0.93,0.02)+12*gaussian(it,0.4,0.12)+5*gaussian(it,0.8,0.05)+gaussian(it,1,0.01))*255./40.))
		if gamma=="acid":
			#self.draw.rectangle(((c1,c2),(c1+10,c2+10)), fill=(int((gaussian(it,0.8,0.05)+24*gaussian(it,0.3,0.09))*255./40.),int((12*gaussian(it,0.3,0.12))*255./40.),int((2*gaussian(it,0.93,0.02)+12*gaussian(it,0.4,0.12)+5*gaussian(it,0.8,0.05)+gaussian(it,1,0.01))*255./40.)))
			rectangle=(c1,ayy-c2, int((7*gaussian(it,0.5,0.12)+2*gaussian(it,0.8,0.05))*255./40.),int((12*gaussian(it,0.3,0.12))*255./40.),int((2*gaussian(it,0.93,0.02)+12*gaussian(it,0.4,0.12)+5*gaussian(it,0.8,0.05)+gaussian(it,1,0.01))*255./40.))
		if gamma=="azure":
			rectangle=(c1,ayy-c2, int((2*gaussian(it,0.8,0.05))*255./40.),int((12*gaussian(it,0.3,0.12))*255./40.),int((2*gaussian(it,0.93,0.02)+12*gaussian(it,0.4,0.12)+5*gaussian(it,0.8,0.05)+gaussian(it,1,0.01))*255./40.))
		if gamma=="bloody grass":
			#green red: 
			rectangle=(c1,ayy-c2, int((2*gaussian(it,0.93,0.02)+12*gaussian(it,0.4,0.12)+5*gaussian(it,0.8,0.05)+gaussian(it,1,0.01))*255./40.),int((12*gaussian(it,0.3,0.12))*255./40.),int((2*gaussian(it,0.7,0.05))*255./40.))
		if gamma=="bw":
			#bw: 
			#self.draw.rectangle(((c1,c2),(c1+10,c2+10)), fill=(int(it*255*(1+10*exp(-it))),int(it*255*(1+10*exp(-it))),int(it*255*(1+10*exp(-it)))))
		#bw rec: 
			rectangle=(c1,ayy-c2, int((2*gaussian(it,0.93,0.02)+12*gaussian(it,0.4,0.12)+5*gaussian(it,0.8,0.05)+gaussian(it,1,0.01))*255./40.),int((2*gaussian(it,0.93,0.02)+12*gaussian(it,0.4,0.12)+5*gaussian(it,0.8,0.05)+gaussian(it,1,0.01))*255./40.),int((2*gaussian(it,0.93,0.02)+12*gaussian(it,0.4,0.12)+5*gaussian(it,0.8,0.05)+gaussian(it,1,0.01))*255./40.))
	if (axey[0]>nitr-2):
		rectangle=(c1,ayy-c2, 0, 0, 0)
	
	
	return rectangle		
			#print(int(c1/self.axx*100))
			
			

		
	
@jit
def gaussian(x, mu, sigma):
	return np.exp(-0.5*((x-mu)/sigma)*((x-mu)/sigma)) / sigma / np.sqrt(2*np.pi)
	
@jit
def cycle(p,q, nitr):
	itr=0.
	xn0=0.
	yn0=0.
	xn=0.
	yn=0.
	while (itr<nitr):
		xn=xn0*xn0-yn0*yn0+p
		yn=2*xn0*yn0+q
		xn0=xn
		yn0=yn
		if (xn0*xn0+yn0*yn0>=10.):
			return (itr, xn0*xn0+yn0*yn0)
			break
		itr=itr+1
	#	print(c1/axx*100)
	return (itr, xn0*xn0+yn0*yn0)
