# SEM Nanoparticle Counter

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![scikit-image](https://img.shields.io/badge/scikit--image-0.18%2B-orange)
![tkinter](https://img.shields.io/badge/tkinter-GUI-yellow)

The **SEM Nanoparticle Counter** is a Python-based tool designed to analyze Scanning Electron Microscope (SEM) images and measure the diameters of nanoparticles. It provides a user-friendly graphical interface (GUI) for loading images, setting scale bar parameters, and visualizing the results, including histograms, Gaussian fits, and particle diameter distributions.

---

## Features

- **Image Preprocessing**: Applies Gaussian blur and adaptive/Otsu's thresholding to prepare the image for analysis.
- **Watershed Segmentation**: Uses the Watershed algorithm to segment nanoparticles in the image.
- **Diameter Measurement**: Measures particle diameters in nanometers using scale bar calibration.
- **Interactive GUI**: Built with `tkinter`, the GUI allows users to:
  - Load SEM images.
  - Set scale bar length and unit (nm or μm).
  - Visualize intermediate steps (e.g., binary image, distance transform).
  - Display histograms and Gaussian fits for diameter distributions.
  - Save results as CSV or PNG files.
- **Verification Tool**: Overlay intermediate processing steps on the original image for verification.

---

## Installation

### Prerequisites

- Python 3.8 or higher.
- Required Python libraries: `opencv-python`, `scikit-image`, `scipy`, `numpy`, `pandas`, `matplotlib`, `tkinter`.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sem-nanoparticle-counter.git
   cd sem-nanoparticle-counter
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python sem_nanoparticle_counter.py
   ```

---

## Usage

1. **Load an Image**:
   - Click the "Browse Image" button to select an SEM image (supported formats: `.tif`, `.jpg`, `.png`).

2. **Set Scale Bar Parameters**:
   - Enter the scale bar length and select the unit (nm or μm).

3. **Analyze the Image**:
   - Click the "Analyze Image" button to process the image and measure nanoparticle diameters.

4. **View Results**:
   - The results include:
     - A histogram of particle diameters (1–100 nm).
     - A Gaussian fit curve (optional).
     - A table of diameter ranges and counts (optional).
     - Mean diameter and standard deviation.

5. **Save Results**:
   - Save the diameter distribution data as a CSV file or the histogram plot as a PNG file.

6. **Verification**:
   - Use the "Verification" button to overlay intermediate processing steps (e.g., binary image, distance transform) on the original image for validation.


---

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with Python and popular libraries like OpenCV, scikit-image, and matplotlib.
- Inspired by the need for simple and efficient nanoparticle analysis tools.

---

## Contact

For questions or feedback, please open an issue on GitHub or contact the author directly.

```
