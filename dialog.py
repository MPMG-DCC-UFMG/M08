# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:47:24 2020

@author: milal
"""

from tkinter import filedialog, Tk, Canvas, NW, mainloop
from PIL import ImageTk,Image 
import os

def _open_dialog_file():
	file_types = [('Pasta de Arquivos', '*.')]
	file_name = None
	title = 'Defina o Diretório Raiz:'

	root = Tk()
	root.withdraw()
	root.filename = filedialog.askdirectory(title=title, initialdir=".")
	file_name = root.filename
	root.destroy()

	return file_name


def _open_dialog_analysis():
	file_types = [('Arquivos numpy', '*.npz')]
	file_name = None
	title = 'Selecione a análise:'

	root = Tk()
	root.withdraw()
	root.filename = filedialog.askopenfilename(title=title, initialdir="./M08/log/",
                                              filetypes=file_types)
	file_name = root.filename
	root.destroy()

	return file_name

def show_local_image(filepath):
    
    root = Tk() 
    pil_image = Image.open(filepath)
    
    canvas = Canvas(root, width = int(pil_image.size[0]), height = int(pil_image.size[1]))      
    canvas.pack()       
    
    img = ImageTk.PhotoImage(pil_image)  

    canvas.create_image(10, 10, anchor=NW, image=img)    
    mainloop()   