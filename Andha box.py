import os
import cv2
import numpy as np
import pandas as pd
from skimage import measure, morphology
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from scipy.stats import norm
from tkinter import Tk, Button, Label, filedialog, messagebox, Entry, StringVar, OptionMenu, Checkbutton, IntVar, Frame, Text, Scrollbar, Canvas, Scale, HORIZONTAL, Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import datetime

def load_image(image_path):
    """Load an image from a specified path."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to load the image. Please check the file path.")
    return image

def convert_scale_bar_length(scale_bar_length, scale_bar_unit):
    """Convert scale bar length to micrometers."""
    if scale_bar_unit == "nm":
        return scale_bar_length / 1000
    elif scale_bar_unit == "μm":
        return scale_bar_length
    else:
        raise ValueError("Invalid unit. Use 'nm' or 'μm'.")

def detect_scale_bar(image):
    """
    Detect the scale bar region in the SEM image.
    Scale bars are typically at the bottom and have high contrast.
    """
    # Use bottom 20% of the image where scale bars are typically located
    height, width = image.shape
    roi_height = int(height * 0.2)
    roi = image[height - roi_height:, :]
    
    # Apply adaptive thresholding to find dark regions
    binary = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on aspect ratio and size
    scale_bar_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        
        # Scale bars are typically wide and short
        if w > width * 0.1 and aspect_ratio > 5 and h < roi_height * 0.5:
            scale_bar_contour = contour
            break
    
    # If no contour found, use the bottom center as fallback
    if scale_bar_contour is None:
        bar_width = width // 4
        bar_height = height // 50
        x = (width - bar_width) // 2
        y = height - bar_height - 10
        return (x, y, bar_width, bar_height)
    
    # Adjust coordinates to full image
    x, y, w, h = cv2.boundingRect(scale_bar_contour)
    return (x, y + (height - roi_height), w, h)

def preprocess_image(image):
    """Preprocess the image using both adaptive and Otsu's thresholding."""
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Adaptive thresholding
    binary_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Otsu's thresholding
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Evaluate which binary image is better
    def evaluate_binary(binary_image):
        # Example metric: Number of connected components (particles)
        num_labels, _ = cv2.connectedComponents(binary_image)
        return num_labels
    
    num_particles_adaptive = evaluate_binary(binary_adaptive)
    num_particles_otsu = evaluate_binary(binary_otsu)
    
    # Choose the better binary image
    if num_particles_adaptive > num_particles_otsu:
        thresholding_type = "Adaptive"
        binary_image = binary_adaptive
    else:
        thresholding_type = "Otsu"
        binary_image = binary_otsu
    
    # Remove small objects (noise)
    cleaned_image = morphology.remove_small_objects(binary_image.astype(bool), min_size=50).astype(np.uint8) * 255
    
    return blurred, binary_image, cleaned_image, thresholding_type

def segment_image(image):
    """Segment the image using Watershed algorithm."""
    distance = ndi.distance_transform_edt(image)
    sure_fg = np.uint8(distance > 0.5 * distance.max())
    sure_bg = cv2.dilate(image, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1  # Add 1 to all labels to ensure background is 1
    markers[unknown == 255] = 0  # Mark unknown region as 0
    labels = watershed(-distance, markers, mask=image)
    return distance, sure_fg, sure_bg, markers, labels

def measure_diameters(labels, image_shape, scale_bar_length_um, scale_bar_pixels):
    """Measure the diameters of segmented regions."""
    properties = measure.regionprops(labels)
    diameters_pixels = [prop.equivalent_diameter for prop in properties]
    pixel_size_um = scale_bar_length_um / scale_bar_pixels
    diameters_um = [d * pixel_size_um for d in diameters_pixels]
    diameters_nm = [d * 1000 for d in diameters_um]
    return diameters_nm

def analyze_sem_image(image_path, scale_bar_length, scale_bar_unit):
    """Analyze a single SEM image and return the diameters and labels."""
    image = load_image(image_path)
    scale_bar_length_um = convert_scale_bar_length(scale_bar_length, scale_bar_unit)
    
    # Detect scale bar region
    scale_bar_rect = detect_scale_bar(image)
    x, y, w, h = scale_bar_rect
    scale_bar_pixels = w  # Actual pixel length of the scale bar
    
    blurred, binary_image, cleaned_image, thresholding_type = preprocess_image(image)
    distance, sure_fg, sure_bg, markers, labels = segment_image(cleaned_image)
    diameters_nm = measure_diameters(labels, image.shape, scale_bar_length_um, scale_bar_pixels)
    
    # Draw red rectangle around scale bar
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return (diameters_nm, labels, blurred, binary_image, cleaned_image, 
            distance, sure_fg, sure_bg, markers, image, thresholding_type, 
            image_rgb, scale_bar_rect)

# GUI Functions
def browse_image():
    global file_path
    file_path = filedialog.askopenfilename(
        title="Select SEM Image",
        filetypes=[("Image files", "*.tif *.jpg *.png"), ("All files", "*.*")]
    )
    if file_path:
        file_label.config(text=f"Selected: {os.path.basename(file_path)}")
        scale_bar_length_entry.config(state="normal")
        scale_bar_unit_menu.config(state="normal")
        analyze_button.config(state="normal")

def analyze_image():
    global main_image, blurred, binary_image, cleaned_image, distance, sure_fg, sure_bg
    global markers, labels, thresholding_type, scale_bar_rect, image_with_scale_bar
    
    try:
        scale_bar_length = float(scale_bar_length_entry.get())
        if scale_bar_length <= 0:
            raise ValueError("Scale bar length must be positive.")
        scale_bar_unit = scale_bar_unit_var.get()
    except ValueError as e:
        messagebox.showerror("Invalid Input", str(e))
        return

    if not file_path:
        messagebox.showerror("Error", "No image selected.")
        return

    try:
        status_label.config(text="Status: Analyzing...")
        window.update_idletasks()
        (diameters_nm, labels, blurred, binary_image, cleaned_image, distance, 
         sure_fg, sure_bg, markers, main_image, thresholding_type, 
         image_with_scale_bar, scale_bar_rect) = analyze_sem_image(file_path, scale_bar_length, scale_bar_unit)
        diameters_filtered = [d for d in diameters_nm if 1 <= d <= 100]

        if len(diameters_filtered) == 0:
            messagebox.showinfo("No Particles", "No particles found in the range 1–100 nm.")
            status_label.config(text="Status: Idle")
            return

        mean_diameter = np.mean(diameters_filtered)
        std_diameter = np.std(diameters_filtered)
        
        # Update the status label
        status_label.config(text="Status: Analysis Complete")
        
        # Clear previous results from the visualization frame
        for widget in visualization_frame.winfo_children():
            widget.destroy()

        # Add thresholding and mean output above the diagram and table
        results_frame = Frame(visualization_frame)
        results_frame.pack(fill="x", pady=5)

        thresholding_label = Label(results_frame, text="Thresholding: ")
        thresholding_label.pack(side="left", padx=5)
        thresholding_value = Label(results_frame, text=f"{thresholding_type}", fg="red", font=("Arial", 10, "bold"))
        thresholding_value.pack(side="left", padx=5)

        result_label = Label(results_frame, text="Mean = ")
        result_label.pack(side="left", padx=5)
        result_value = Label(results_frame, text=f"{mean_diameter:.2f} ± {std_diameter:.2f} nm", fg="red", font=("Arial", 10, "bold"))
        result_value.pack(side="left", padx=5)

        # Add scale bar info
        x, y, w, h = scale_bar_rect
        scale_bar_info = Label(results_frame, text=f" | Scale Bar: {w}px → {scale_bar_length}{scale_bar_unit}")
        scale_bar_info.pack(side="left", padx=5)

        if show_table.get():
            hist, bin_edges = np.histogram(diameters_filtered, bins=20, range=(1, 100))
            table_data = pd.DataFrame({
                "Diameter Range (nm)": [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}" for i in range(len(bin_edges) - 1)],
                "Count": hist
            })
            table_frame = Frame(visualization_frame)
            table_frame.pack(fill="both", expand=True)
            table_text = Text(table_frame, wrap="none", width=40, height=10)
            table_text.insert("end", table_data.to_string(index=False))
            table_text.config(state="disabled")
            table_text.pack(side="left", fill="both", expand=True)
            scrollbar = Scrollbar(table_frame, orient="vertical", command=table_text.yview)
            scrollbar.pack(side="right", fill="y")
            table_text.config(yscrollcommand=scrollbar.set)

        if show_histogram.get() or show_gaussian.get():
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)

            if show_histogram.get():
                counts, bins, _ = ax.hist(diameters_filtered, bins=20, edgecolor='black', alpha=0.7, color=histogram_color.get(), label="Histogram")
            if show_gaussian.get():
                x = np.linspace(min(diameters_filtered), max(diameters_filtered), 1000)
                y = norm.pdf(x, mean_diameter, std_diameter) * (len(diameters_filtered) if show_histogram.get() else 1)
                ax.plot(x, y, color=gaussian_color.get(), label="Gaussian Fit")

            ax.set_xlabel('Diameter (nm)')
            ax.set_ylabel('Count')
            ax.set_title(f'Diameter Distribution (1–100 nm) - {os.path.basename(file_path)}')

            if show_legend.get():
                ax.legend()
            if show_mean_sd.get():
                 # Add mean ± standard deviation in a text box
                text_box = f"Mean = {mean_diameter:.2f} ± {std_diameter:.2f} nm"
                ax.text(0.97, 0.95, text_box, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))

            canvas_plot = FigureCanvasTkAgg(fig, master=visualization_frame)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().pack(fill="both", expand=True)
            toolbar_frame = Frame(visualization_frame)
            toolbar_frame.pack(fill="x")
            toolbar = NavigationToolbar2Tk(canvas_plot, toolbar_frame)
            toolbar.update()

      
        # Add Save Results and Verification buttons
        button_frame = Frame(visualization_frame)
        button_frame.pack(fill="x", pady=5)

        save_button = Button(button_frame, text="Save Results", command=lambda: save_results(diameters_filtered, fig))
        save_button.pack(side="left", padx=5)

        verify_button = Button(button_frame, text="Verification", command=open_verification_window)
        verify_button.pack(side="left", padx=5)
        
        canvas_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        status_label.config(text="Status: Idle")

def open_verification_window():
    """Open a new window for overlay verification."""
    verification_window = Toplevel(window)
    verification_window.title("Verification - Overlay Intermediate Steps")
    verification_window.geometry("600x400")

    # Dropdown to select intermediate image
    step_label = Label(verification_window, text="Select Step Image:")
    step_label.pack(pady=5)

    step_var = StringVar(verification_window)
    step_var.set("Original with Scale Bar")
    step_menu = OptionMenu(verification_window, step_var, 
                           "Original with Scale Bar", "Blurred Image", "Binary Image", "Cleaned Image", 
                           "Distance Transform", "Markers", "Watershed Segmentation")
    step_menu.pack(pady=5)

    # Slider to control opacity
    opacity_label = Label(verification_window, text="Opacity:")
    opacity_label.pack(pady=5)

    opacity_scale = Scale(verification_window, from_=0, to=100, orient=HORIZONTAL)
    opacity_scale.set(100)
    opacity_scale.pack(pady=5)

    # Canvas to display overlay
    fig_overlay = Figure(figsize=(6, 6), dpi=100)
    ax_overlay = fig_overlay.add_subplot(111)
    canvas_overlay = FigureCanvasTkAgg(fig_overlay, master=verification_window)
    canvas_overlay.draw()
    canvas_overlay.get_tk_widget().pack(fill="both", expand=True)

    def update_overlay():
        """Update the overlay based on the selected step and opacity."""
        step = step_var.get()
        opacity = opacity_scale.get() / 100.0

        # Special case for original image with scale bar
        if step == "Original with Scale Bar":
            ax_overlay.clear()
            ax_overlay.imshow(image_with_scale_bar)
            ax_overlay.set_title("Original Image with Scale Bar Detection")
            canvas_overlay.draw()
            return

        step_images = {
            "Blurred Image": blurred,
            "Binary Image": binary_image,
            "Cleaned Image": cleaned_image,
            "Distance Transform": distance,
            "Markers": markers,
            "Watershed Segmentation": labels
        }
        step_image = step_images.get(step, blurred)

        # Normalize step image to match the main image's intensity range
        step_image_normalized = cv2.normalize(step_image, None, 0, 255, cv2.NORM_MINMAX)

        # Create RGB version of main image with scale bar
        if len(image_with_scale_bar.shape) == 3:  # Already RGB
            base_image = image_with_scale_bar
        else:
            base_image = cv2.cvtColor(image_with_scale_bar, cv2.COLOR_GRAY2BGR)
        
        # Convert step image to RGB if needed
        if len(step_image_normalized.shape) == 2:
            step_image_rgb = cv2.cvtColor(step_image_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            step_image_rgb = step_image_normalized.astype(np.uint8)

        # Overlay the step image on the main image
        overlay_image = cv2.addWeighted(base_image.astype(np.float32), 1 - opacity, 
                                       step_image_rgb.astype(np.float32), opacity, 0)

        # Update the overlay plot
        ax_overlay.clear()
        ax_overlay.imshow(overlay_image.astype(np.uint8))
        ax_overlay.set_title(f"Overlay - {step} (Opacity: {opacity * 100:.0f}%)")
        canvas_overlay.draw()

    # Bind the update function to the dropdown and slider
    step_var.trace("w", lambda *args: update_overlay())
    opacity_scale.config(command=lambda *args: update_overlay())

    # Initial overlay
    update_overlay()

def on_mouse_wheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

def save_results(diameters_filtered, fig=None):
    save_path = filedialog.asksaveasfilename(
        title="Save Results",
        filetypes=[("CSV files", "*.csv"), ("PNG files", "*.png"), ("All files", "*.*")]
    )

    if save_path:
        if save_path.endswith(".csv"):
            hist, bin_edges = np.histogram(diameters_filtered, bins=20, range=(1, 100))
            table_data = pd.DataFrame({
                "Diameter Range (nm)": [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}" for i in range(len(bin_edges) - 1)],
                "Count": hist
            })
            table_data.to_csv(save_path, index=False)
        elif save_path.endswith(".png") and fig is not None:
            fig.savefig(save_path)
        messagebox.showinfo("Success", "Results saved successfully.")

# Create the main window
window = Tk()
window.title("SEM Nanoparticle Counter by ChemGarage")

canvas = Canvas(window)
canvas.pack(side="left", fill="both", expand=True)
scrollbar = Scrollbar(window, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.bind_all("<MouseWheel>", on_mouse_wheel)

canvas_frame = Frame(canvas)
canvas.create_window((0, 0), window=canvas_frame, anchor="nw")

# Input section
browse_button = Button(canvas_frame, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

file_label = Label(canvas_frame, text="No file selected")
file_label.pack(pady=5)

scale_bar_frame = Frame(canvas_frame)
scale_bar_frame.pack(fill="x", pady=5)

scale_bar_length_label = Label(scale_bar_frame, text="Scale Bar Length:")
scale_bar_length_label.pack(side="left", padx=5)
scale_bar_length_entry = Entry(scale_bar_frame, state="disabled")
scale_bar_length_entry.pack(side="left", padx=5)

scale_bar_unit_label = Label(scale_bar_frame, text="Scale Bar Unit:")
scale_bar_unit_label.pack(side="left", padx=5)
scale_bar_unit_var = StringVar(window)
scale_bar_unit_var.set("μm")
scale_bar_unit_menu = OptionMenu(scale_bar_frame, scale_bar_unit_var, "μm", "nm")
scale_bar_unit_menu.pack(side="left", padx=5)
scale_bar_unit_menu.config(state="disabled")

histogram_frame = Frame(canvas_frame)
histogram_frame.pack(fill="x", pady=5)
show_histogram = IntVar(value=1)
histogram_check = Checkbutton(histogram_frame, text="Show Histogram", variable=show_histogram)
histogram_check.pack(side="left", padx=5)
histogram_color_label = Label(histogram_frame, text="Histogram Color:")
histogram_color_label.pack(side="left", padx=5)
histogram_color = StringVar(window)
histogram_color.set("blue")
histogram_color_menu = OptionMenu(histogram_frame, histogram_color, "blue", "green", "red", "orange", "purple")
histogram_color_menu.pack(side="left", padx=5)

gaussian_frame = Frame(canvas_frame)
gaussian_frame.pack(fill="x", pady=5)
show_gaussian = IntVar(value=1)
gaussian_check = Checkbutton(gaussian_frame, text="Show Gaussian Curve", variable=show_gaussian)
gaussian_check.pack(side="left", padx=5)
gaussian_color_label = Label(gaussian_frame, text="Gaussian Curve Color:")
gaussian_color_label.pack(side="left", padx=5)
gaussian_color = StringVar(window)
gaussian_color.set("red")
gaussian_color_menu = OptionMenu(gaussian_frame, gaussian_color, "red", "blue", "green", "orange", "purple")
gaussian_color_menu.pack(side="left", padx=5)

show_table = IntVar(value=0)
table_check = Checkbutton(canvas_frame, text="Show Diameter and Count Table", variable=show_table)
table_check.pack(pady=5)

legend_frame = Frame(canvas_frame)
legend_frame.pack(fill="x", pady=5)
show_legend = IntVar(value=1)
legend_checkbox = Checkbutton(legend_frame, text="Show Legend", variable=show_legend)
legend_checkbox.pack(side="left", padx=5)

show_mean_sd = IntVar(value=0)
mean_sd_checkbox = Checkbutton(legend_frame, text="Show Mean ± SD", variable=show_mean_sd)
mean_sd_checkbox.pack(side="left", padx=5)

analyze_button = Button(canvas_frame, text="Analyze Image", command=analyze_image, state="disabled")
analyze_button.pack(pady=10)

# Status section
status_label = Label(canvas_frame, text="Status: Idle")
status_label.pack(pady=5)

# Visualization frame - will be after the status
visualization_frame = Frame(canvas_frame)
visualization_frame.pack(fill="both", expand=True, pady=5)


# Add timestamp at the bottom
timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
timestamp_label = Label(canvas_frame, text=f"Calculated at: {timestamp} UTC ", font=("Arial", 8))
timestamp_label.pack(side="bottom", pady=5)

window.mainloop()