import PySimpleGUI as sg

event, (filename,) = sg.Window('Get filename example').\
    Layout([
        [sg.Text('Filename')],
        [sg.Input(), sg.FileBrowse()],
        [sg.OK(), sg.Cancel()]
    ]).Read()
