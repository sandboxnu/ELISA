from kivy.app import App
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, Screen
import cv2
import datetime


sm = ScreenManager()


class TestApp(App):

    def build(self):
        return sm


if __name__ == '__main__':
    TestApp().run()


def save_img(texture):
    texture.save(f'/data/{datetime.datetime.now().isoformat()}.png')
    return 0


class KivyCamera(Image):
    def __init__(self, capture=cv2.VideoCapture(-1), fps=30, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        del dt
        ret, frame = self.capture.read()
        if ret:
            # convert it to texture
            raw_buffer = cv2.flip(frame, 0)
            buffer_as_str = raw_buffer.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(
                buffer_as_str, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture

    def save_img(self):
        save_img(self.texture)


class SelectScreen(Screen):
    pass


class CameraScreen(Screen):
    pass


class ReviewScreen(Screen):
    pass


class ScreenManagement(ScreenManager):
    pass


presentation = Builder.load_file("main.kv")


class MainApp(App):

    def build(self):
        return presentation


if __name__ == "__main__":
    MainApp().run()
