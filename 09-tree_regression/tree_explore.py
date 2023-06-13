from numpy import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
import reg_trees


def re_draw(tol_s, tol_n):
    re_draw.f.clf()
    re_draw.a = re_draw.f.add_subplot(111)

    if chk_btn_var.get():
        if tol_n < 2:
            tol_n = 2
        my_tree = reg_trees.create_tree(re_draw.rawDat, reg_trees.model_leaf, reg_trees.model_err, (tol_s, tol_n))
        y_hat = reg_trees.create_fore_cast(my_tree, re_draw.testDat, reg_trees.model_tree_eval)
    else:
        my_tree = reg_trees.create_tree(re_draw.rawDat, ops=(tol_s, tol_n))
        y_hat = reg_trees.create_fore_cast(my_tree, re_draw.testDat)

    re_draw.a.scatter(re_draw.rawDat[:, 0].tolist(), re_draw.rawDat[:, 1].tolist(), s=5)
    re_draw.a.plot(re_draw.testDat, y_hat, linewidth=2.0)
    re_draw.canvas.draw()


def get_inputs():
    try:
        tol_n = int(tol_n_entry.get())
    except:
        tol_n = 10
        print("enter integer for tol_n")
        tol_n_entry.delete(0, END)
        tol_n_entry.insert(0, '10')

    try:
        tol_s = int(tol_s_entry.get())
    except:
        tol_s = 1.0
        print("enter float for tol_s")
        tol_s_entry.delete(0, END)
        tol_s_entry.insert(0, '1.0')
    return tol_n, tol_s


def draw_new_tree():
    tol_n, tol_s = get_inputs()
    re_draw(tol_s, tol_n)


root = Tk()

re_draw.f = Figure(figsize=(5, 4), dpi=100)
re_draw.canvas = FigureCanvasTkAgg(re_draw.f, master=root)
re_draw.canvas.draw()
re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text="tolN").grid(row=1, column=0)
tol_n_entry = Entry(root)
tol_n_entry.grid(row=1, column=1)
tol_n_entry.insert(0, "10")

Label(root, text="tolS").grid(row=2, column=0)
tol_s_entry = Entry(root)
tol_s_entry.grid(row=2, column=1)
tol_s_entry.insert(0, "1.0")

Button(root, text="reDraw", command=draw_new_tree).grid(row=1, column=2, rowspan=3)

chk_btn_var = IntVar()
chk_btn = Checkbutton(root, text="Model Tree", variable=chk_btn_var)
chk_btn.grid(row=3, column=0, columnspan=2)

re_draw.rawDat = mat(reg_trees.load_dataset('resource/sine.txt'))
re_draw.testDat = arange(min(re_draw.rawDat[:, 0]), max(re_draw.rawDat[:, 0]), 0.01)
re_draw(1.0, 10)

root.mainloop()


if __name__ == "__main__":
    pass