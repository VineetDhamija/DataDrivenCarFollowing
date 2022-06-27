from interface import Interface


class Moveable(Interface):

    def time_to_change(self, dt):
        pass

    def translate(self, dt):
        pass

    def accelerate(self, dt, fwd=None):
        pass

    def acceleration(self, fwd=None):
        pass

    def distance_to(self, fwd):
        '''
        return fwd.pos - self.pos - self.length
        '''
        pass

    def change(self, f_old, b_old, f_new, b_new):
        pass
