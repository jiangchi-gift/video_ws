from image.image_manager import ImageManager
from tools.rectangle_tool import RectangleTool

image = ImageManager.read_image("test.jpg")

result = RectangleTool.draw_rectangle(
    image,
    100,
    100,
    300,
    300
)

ImageManager.show_image("Test", result)