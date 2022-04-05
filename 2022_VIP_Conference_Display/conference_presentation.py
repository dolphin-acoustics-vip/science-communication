import tkinter as tk
from tkinter import PhotoImage
from tkinter import Tk
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import tensorflow as tf
import librosa.display
import os
from PIL import Image
import sounddevice as sd
from scipy.io.wavfile import write


# TODO: Dynamically create absolute paths so that program can be run as a compiled 
# executable version on any computer

def set_path(relative_path):
    """" This function sets the absolute path (to avoid issues with conversion to exe file)
    """
    # This function is currently unused
    base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def save_spectrogram_image(
        input_path,
        output_path,
        image_name,
        sampling_rate=48000,
        n_fft=512,
        dpi=96,  # this should be dpi of your own screen
        max_freq=22000,  # for cropping
        min_freq=3000,  # for cropping
        img_size=(413, 202)):
    """
    This function takes in the above parameters and
    generates a spectrogram from a given sample recording and
    saves the spectrogram image
    """

    f_step = sampling_rate / n_fft
    min_bin = int(min_freq / f_step)
    max_bin = int(max_freq / f_step)

    # Generate image
    x, sr = librosa.load(input_path, sr=sampling_rate)

    X = librosa.stft(x, n_fft=n_fft)  # Apply fourier transform
    # Crop image vertically (frequency axis) from min_bin to max_bin
    X = X[min_bin:max_bin, :]

    # TODO change refs
    Xdb = librosa.amplitude_to_db(
        abs(X), ref=np.median
    )  # Convert amplitude spectrogram to dB-scaled spec
    fig = plt.figure(
        frameon=False, figsize=(img_size[0] / dpi, img_size[1] / dpi), dpi=dpi
    )  # Reduce image

    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(Xdb, cmap="gray_r", sr=sr,
                             x_axis="time", y_axis="hz")
    plt.show()

    # Save image
    fig.savefig(os.path.join(output_path, str(image_name) + ".png"))
    plt.close(fig)


def image_to_array(image_file):
    """This function coverts a PIL image to a numpy array.
    """
    # load image
    image = Image.open(image_file).convert("RGB")

    # convert image to numpy array
    return asarray(image)


def get_prediction(image_file, model):
    """ This function returns a string of the predicted dolphin species
    """

    species_classes = ["bottlenose", "common", "melon-headed"]
    image_array = image_to_array(image_file)
    image_array = np.array([image_array.tolist()])
    predicted_class = np.argmax(model.predict(image_array))

    return species_classes[predicted_class]


def update_label(label, image_file, model):
    """ This function updates the species label being displayed in the tkinter gui by 
    obtaining the machine learning model's most recent precition on the audio input
    """

    new_label = get_prediction(image_file, model)
    label.config(text=new_label)


def record_audio(fs=48000, seconds=1):
    """ This function records audio and writes it to a wav file
    """
    # fs: sample rate, seconds: duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('./tmp/output.wav', fs, myrecording)  # Save as WAV file


def record_and_predict(prediction_label, model):
    """
    This function records input audio and updates the species prediction label accordingly.
    """
    record_audio()
    save_spectrogram_image("./tmp/output.wav", "tmp", "saved_spectrogram")
    update_label(prediction_label, "./tmp/saved_spectrogram.png", model)


def run_spectrogram_loop(root, canvas, screen_centre, species_label, model,
                     spectrogram_image=None):
    """This function continuously updates the spectrogram
    display and its corresponding dolphin species classification.
    """
    # create list of arguments to recursively pass to root.after()
    args = [root, canvas, screen_centre, species_label, model]
    
    # record and audio input and predict new species
    record_and_predict(species_label, model)

    # remove current spectrogram from canvas
    canvas.delete(spectrogram_image)

    # add new spectogram to canvas
    canvas.img = PhotoImage(file='./tmp/saved_spectrogram.png')
    spectrogram_image = canvas.create_image(
        screen_centre, anchor="center", image=canvas.img)
    canvas.tag_raise(spectrogram_image)
    
    # run again after 1 second
    root.after(1000, run_spectrogram_loop, *args)

def main():
    """This function is responsible for executing the display.
    """

    # initailize machine learning model
    directory_name = "./"
    model = tf.keras.models.load_model(
        directory_name + "2022_02_18_transfer_learning_classifier_fine_tuned_improved.h5")

    # create root of tkinter GUI
    root = Tk()

    # set the display widnow to be the size of the screen
    screen_height = root.winfo_screenheight()
    screen_width = root.winfo_screenwidth()
    screen_centre = ((screen_width/2),
                     (screen_height/2))
    root.geometry(f"{screen_width}x{screen_height}")

    # resize background image to fit size of screen
    background_image = Image.open("./images/background.png")
    resized_background = background_image.resize((screen_width, screen_height))
    resized_background.save("./images/resized_background.png")

    # create tkinter object with resized image
    tk_background_image = tk.PhotoImage(file="./images/resized_background.png")

    # create canvas for GUI
    canvas = tk.Canvas(root, width=500, height=500)
    canvas.pack(fill="both", expand=True)  # display image
    canvas_background_image = canvas.create_image(
        0, 0, image=tk_background_image, anchor="nw")
    # send background image to background
    canvas.tag_lower(canvas_background_image)

    # create title label for GUI
    title_label = tk.Label(root, text="So, which Dolphin Species are you?", font=("Candara", 50),
                           bg="#2ababf")
    title_label.place(relx=0.5, rely=0.3, anchor="center")

    # create text label for species predictions
    text_label = tk.Label(root, fg='#fff', bg="#000")
    text_label.place(relx=0.5, rely=0.5, anchor='center')

    # start display
    run_spectrogram_loop(root, canvas, screen_centre, text_label, model)
    tk.mainloop()


if __name__ == "__main__":
    main()
