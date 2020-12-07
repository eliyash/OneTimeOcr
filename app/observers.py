# TODO: make self._observers as set again!
class Subject:
    def __init__(self, init_val=None):
        self._observers = list()
        self._subject_state = init_val

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def _notify(self):
        for observer in self._observers:
            observer(self._subject_state)

    @property
    def data(self):
        return self._subject_state

    @data.setter
    def data(self, arg):
        self._subject_state = arg
        self._notify()
