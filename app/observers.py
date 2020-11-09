class Subject:
    def __init__(self, init_val=None):
        self._observers = set()
        self._subject_state = init_val

    def attach(self, observer):
        self._observers.add(observer)

    def detach(self, observer):
        self._observers.discard(observer)

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
