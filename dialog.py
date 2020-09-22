# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:47:24 2020

@author: milal
"""

from tkinter import filedialog, Tk
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
