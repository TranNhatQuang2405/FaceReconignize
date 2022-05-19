import tkinter
from tkinter import LEFT, RIGHT, Button, Frame, Tk, Label, PhotoImage, font
from tkinter.filedialog import Open, SaveAs
from PIL import Image, ImageTk
from Predict import Predict


class Main(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()

    def resize_img(self):
        if(self.image.size[0] > self.image.size[1]):
            wpercent = (550/float(self.image.size[0]))
            hsize = int((float(self.image.size[1])*float(wpercent)))
            self.image = self.image.resize((550, hsize), Image.ANTIALIAS)
        else:
            hpercent = (550/float(self.image.size[1]))
            wsize = int((float(self.image.size[0])*float(hpercent)))
            self.image = self.image.resize((wsize, 550), Image.ANTIALIAS)

    def initUI(self):
        self.font = font.Font(size=15, weight="bold")
        self.image = Image.open("./image/system/bg.png")
        self.resize_img()
        self.labelImage = ImageTk.PhotoImage(self.image)
        self.label = Label(root, image=self.labelImage, width=580, height=550)
        self.buttonSelect = Button(
            root, text="Select Image",
            command=self.onOpen,
            activeforeground="Black",
            activebackground="Orange",
            bd=10, bg='#0052cc', fg='#ffffff')
        self.buttonReconignize = Button(
            root, text="Reconignize",
            command=self.onRecognition,
            activeforeground="Black",
            activebackground="Orange",
            bd=10, bg='#0052cc', fg='#ffffff')
        self.buttonSelect['font'] = self.font
        self.buttonReconignize['font'] = self.font
        self.reRender()

    def onOpen(self):
        global ftypes
        ftypes = [('Images', '*.jpg *.png')]
        dlg = Open(self, filetypes=ftypes)
        fl = dlg.show()
        if fl != '':
            self.image = Image.open(fl)
            self.resize_img()
            self.labelImage = ImageTk.PhotoImage(self.image)
            self.label.configure(image=self.labelImage)
        self.reRender()

    def reRender(self):
        self.label.pack(side="top")
        self.buttonSelect.pack(side=LEFT)
        self.buttonReconignize.pack(side=RIGHT)

    def onRecognition(self):
        image = Predict(self.image)
        self.labelImage = ImageTk.PhotoImage(image)
        self.label.configure(image=self.labelImage)
        self.reRender()


root = Tk()
Main(root)
root.geometry("600x630")
root.mainloop()
