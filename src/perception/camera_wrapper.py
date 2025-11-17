import numpy as np
import matplotlib.pyplot as plt


class CameraWrapper:
    """
    Wraps the Drake HardwareStation camera ports to provide easy access
    to numpy images and point clouds.
    """

    def __init__(self, station, context, camera_name="wrist_camera_sensor"):
        self.station = station
        self.context = context
        self.camera_name = camera_name

        # Connect to the specific output ports created by MakeHardwareStation
        # Naming convention is usually: {camera_name}.{image_type}
        self.port_rgb = self.station.GetOutputPort(f"{camera_name}.rgb_image")
        self.port_depth = self.station.GetOutputPort(f"{camera_name}.depth_image")

    def get_rgb_image(self) -> np.ndarray:
        """
        Returns:
            img (np.ndarray): (H, W, 4) array of uint8. Channels are RGBA.
        """
        # Evaluate the port to get the AbstractValue
        abstract_img = self.port_rgb.Eval(self.context)

        # Access the data (pydrake.systems.sensors.ImageRgba8U)
        # data returns a copy as a numpy array, but shape is (4, W, H) flattened or similar
        # The .data attribute of Drake images is a property that returns a numpy view
        img_data = abstract_img.data

        # Drake image data is (H, W, 4) if accessed correctly via data property on the python side
        # But sometimes we need to reshape depending on the version.
        # Let's trust the .data property for modern pydrake.
        return np.copy(img_data)

    def get_depth_image(
        self, inject_noise: bool = False, sigma: float = 0.005
    ) -> np.ndarray:
        """
        Returns:
            img (np.ndarray): (H, W, 1) array of float32. Values are depth in meters.
        """
        abstract_img = self.port_depth.Eval(self.context)

        img_data = abstract_img.data

        depth_img = np.copy(img_data)

        if inject_noise:
            # Create noise with the same shape as the image
            depth_img = self._add_noise_function(depth_img, sigma=sigma)

            # Clip negative values (cannot have negative distance)
            depth_img[depth_img < 0] = 0.0

        depth_img = np.array(depth_img, copy=False).reshape(abstract_img.height(), abstract_img.width(), -1)
        depth_img = np.ma.masked_invalid(depth_img)

        return depth_img

    def _add_noise_function(self, depth_image: np.ndarray, sigma: float = 0.005) -> np.ndarray:
        """
        Adds noise to the depth image.
        """

        def _noise_per_pixel(pixel_value):
            return np.random.normal(pixel_value, sigma * pixel_value)

        return np.vectorize(_noise_per_pixel)(depth_image)

    def plot_frames(self) -> None:
        """Helper to visualize what the robot sees"""
        rgb = self.get_rgb_image()
        depth = self.get_depth_image(inject_noise=False)
        depth_noisy = self.get_depth_image(inject_noise=True, sigma=0.01)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(rgb)
        ax[0].set_title("RGB Image")

        ax[1].imshow(depth.squeeze(), cmap="plasma")
        ax[1].set_title("Ideal Depth (Meters)")

        ax[2].imshow(depth_noisy.squeeze(), cmap="plasma")
        ax[2].set_title("Noisy Depth (sigma=0.01)")

        plt.show()
