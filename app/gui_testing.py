import tkinter as tk


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.exit_button3 = tk.Button(self.main_window, text="1", command=self.change_color)
        self.exit_button3.pack()

        self.frame = tk.Frame(self.main_window)
        self.frame.pack()
        for i in range(3):
            for j in range(3):
                def f_creator(i, j):
                    def print_ind():
                        print("I am {}_{}".format(i, j))
                    return print_ind
                self.exit_button = tk.Button(self.frame, text="{}_{}".format(i, j), command=f_creator(i, j))
                self.exit_button.grid(row=i, column=j, sticky="nsew")

    def change_color(self):
        self.exit_button.configure(bg="red")
        print('a')

a = App()
a.main_window.mainloop()
# Code to add widgets will go here...