from image_manager import ImageManager
from rectangle_tool import RectangleTool
from config_manager import ConfigManager


def main():
    config = ConfigManager()

    config.set_config("image_path", "test.jpg")
    config.set_config("save_path", "result.jpg")

    image_path = config.get_config("image_path")
    save_path = config.get_config("save_path")

    image = ImageManager.read_image(image_path)

    result = RectangleTool.draw_rectangle(
        image,
        x1=100,
        y1=100,
        x2=300,
        y2=300
    )

    result = RectangleTool.draw_text(
        result,
        text="Target",
        x=100,
        y=90
    )

    ImageManager.show_image("Result", result)
    ImageManager.save_image(save_path, result)


if __name__ == "__main__":
    main()