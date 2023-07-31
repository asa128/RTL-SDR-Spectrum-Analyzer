from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QGridLayout, QDoubleSpinBox, QCheckBox,
    QSpinBox, QComboBox, QRadioButton, QProgressBar
)
from PyQt6.QtCore import Qt

from pylab import *
from rtlsdr import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class Window(QWidget):  # user interface
    def __init__(self):
        self.cancel_scan_clicked = False
        super().__init__()
        layout = QGridLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        self.setWindowTitle("SpecScan")
        self.setWindowIcon(QIcon("icon.png"))
        self.setLayout(layout)

        # title
        self.title_image = QPixmap("title.png")
        self.title_label = QLabel()
        self.title_label.setPixmap(self.title_image)
        layout.addWidget(self.title_label, 0, 0, 1, 3, Qt.AlignmentFlag.AlignCenter)

        # load preset label
        load_preset_label = QLabel("Load Preset: ")
        layout.addWidget(load_preset_label, 1, 0)

        # load preset box
        self.preset_in = QComboBox()
        self.preset_in.addItems(['', 'Amateur 6m Band', 'TV Channels 2-6 (Low-VHF Band)', 'FM Radio',
                                 'Aircraft Band', 'Amateur 2m Band', 'NOAA Weather Radio',
                                 'TV Channels 7-13 (High-VHF Band)', 'Amateur 1.25m Band', 'Amateur 70cm Band',
                                 'TV Channels 14-36 (UHF Band)', 'LTE/5G Band 71 Downlink', 'LTE/5G Band 71 Uplink',
                                 'LTE/5G Band 12 Downlink', 'LTE/5G Band 12 Uplink', 'LTE/5G Band 13 Downlink',
                                 'LTE/5G Band 13 Uplink', 'LTE/5G Band 14 Downlink', 'LTE/5G Band 14 Uplink',
                                 'LTE/5G Band 5 Downlink', 'LTE/5G Band 5 Uplink', 'Amateur 33cm Band',
                                 'Amateur 23cm Band'])
        self.preset_in.setCurrentIndex(10)
        layout.addWidget(self.preset_in, 1, 1, 1, 2)
        self.preset_in.currentIndexChanged.connect(self.load_preset)

        # start frequency label
        start_label = QLabel("Start Frequency (MHz): ")
        layout.addWidget(start_label, 2, 0)

        # start frequency box
        self.start_freq_in = QDoubleSpinBox()
        self.start_freq_in.setDecimals(1)
        self.start_freq_in.setSingleStep(0.1)
        self.start_freq_in.setRange(30.0, 1700.0)
        self.start_freq_in.setValue(470.0)
        layout.addWidget(self.start_freq_in, 2, 1, 1, 2)
        self.start_freq_in.valueChanged.connect(self.update_filter_length)
        self.start_freq_in.editingFinished.connect(self.verify_start_freq)
        self.start_freq_in.setToolTip("Valid values are 30.0 - 1700.0")

        # stop frequency label
        stop_label = QLabel("Stop Frequency (MHz): ")
        layout.addWidget(stop_label, 3, 0)

        # stop frequency box
        self.stop_freq_in = QDoubleSpinBox()
        self.stop_freq_in.setDecimals(1)
        self.stop_freq_in.setSingleStep(0.1)
        self.stop_freq_in.setRange(30.0, 1700.0)
        self.stop_freq_in.setValue(608.0)
        layout.addWidget(self.stop_freq_in, 3, 1, 1, 2)
        self.stop_freq_in.valueChanged.connect(self.update_filter_length)
        self.stop_freq_in.editingFinished.connect(self.verify_stop_freq)
        self.stop_freq_in.setToolTip("Valid values are 30.0 - 1700.0")

        # resolution label
        resolution_label = QLabel("Resolution (FFT Length): ")
        layout.addWidget(resolution_label, 4, 0)

        # resolution box
        self.resolution_in = QComboBox()
        self.resolution_in.addItems(['37.5 kHz (64)', '18.75 kHz (128)', '9.375 kHz (256)', '4.6875 kHz (512)',
                                     '2.34375 kHz (1024)'])
        self.resolution_in.setCurrentIndex(4)
        layout.addWidget(self.resolution_in, 4, 1, 1, 2)
        self.resolution_in.currentIndexChanged.connect(self.update_filter_length)

        # gain label
        gain_label = QLabel("RF Gain (dB): ")
        layout.addWidget(gain_label, 5, 0)

        # gain box
        self.gain_in = QSpinBox()
        self.gain_in.setRange(0, 30)
        self.gain_in.setValue(15)
        layout.addWidget(self.gain_in, 5, 1, 1, 2)
        self.gain_in.setToolTip("Valid values are 0 - 30")

        # smoothing label
        smoothing_label = QLabel("Smoothing: ")
        layout.addWidget(smoothing_label, 6, 0)
        smoothing_label.setToolTip("1st-Order Savitzky-Golay Filter on PSD Plot")

        # smoothing automatic radio button
        self.smooth_auto_radio = QRadioButton("Automatic")
        self.smooth_auto_radio.setChecked(True)
        layout.addWidget(self.smooth_auto_radio, 6, 1)
        self.smooth_auto_radio.clicked.connect(self.update_filter_length)

        # smoothing manual radio button
        self.smooth_manual_radio = QRadioButton("Manual")
        layout.addWidget(self.smooth_manual_radio, 7, 1)
        self.smooth_manual_radio.clicked.connect(self.filter_switched_to_manual)

        # smoothing off radio button
        self.smooth_off_radio = QRadioButton("Off")
        layout.addWidget(self.smooth_off_radio, 8, 1)
        self.smooth_off_radio.clicked.connect(self.filter_switched_to_off)

        # smoothing window length label
        self.smoothing_window_label = QLabel("Smoothing Filter Length: ")
        layout.addWidget(self.smoothing_window_label, 9, 0)
        not_resize = self.smoothing_window_label.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.smoothing_window_label.setSizePolicy(not_resize)

        # smoothing window length box
        self.smoothing_window_length = QSpinBox()
        self.smoothing_window_length.setRange(2, 1000)
        self.smoothing_window_length.setValue(138)
        self.smoothing_window_length.setDisabled(True)
        layout.addWidget(self.smoothing_window_length, 9, 1, 1, 2)
        not_resize = self.smoothing_window_length.sizePolicy()
        not_resize.setRetainSizeWhenHidden(True)
        self.smoothing_window_length.setSizePolicy(not_resize)
        self.smoothing_window_length.setToolTip("Valid values are 2 - 1000")

        # plot in decibels checkbox
        self.db_option = QCheckBox("Plot in dB")
        self.db_option.setChecked(True)
        layout.addWidget(self.db_option, 10, 1)

        # scan button
        self.scan_button = QPushButton("Scan")
        self.scan_button.clicked.connect(self.start_scan)
        layout.addWidget(self.scan_button, 11, 2)

        # progress bar
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar, 11, 1)

        # cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_scan)
        layout.addWidget(self.cancel_button, 12, 2)
        self.cancel_button.setDisabled(True)

    def load_preset(self):  # frequencies to load for each preset
        if self.preset_in.currentText() == "Amateur 6m Band":
            self.start_freq_in.setValue(50.0)
            self.stop_freq_in.setValue(54.0)
        if self.preset_in.currentText() == "TV Channels 2-6 (Low-VHF Band)":
            self.start_freq_in.setValue(54.0)
            self.stop_freq_in.setValue(88.0)
        if self.preset_in.currentText() == "FM Radio":
            self.start_freq_in.setValue(87.7)
            self.stop_freq_in.setValue(108.1)
        if self.preset_in.currentText() == "Aircraft Band":
            self.start_freq_in.setValue(108.0)
            self.stop_freq_in.setValue(137.0)
        if self.preset_in.currentText() == "Amateur 2m Band":
            self.start_freq_in.setValue(144.0)
            self.stop_freq_in.setValue(148.0)
        if self.preset_in.currentText() == "NOAA Weather Radio":
            self.start_freq_in.setValue(162.3)
            self.stop_freq_in.setValue(162.7)
        if self.preset_in.currentText() == "TV Channels 7-13 (High-VHF Band)":
            self.start_freq_in.setValue(174.0)
            self.stop_freq_in.setValue(216.0)
        if self.preset_in.currentText() == "Amateur 1.25m Band":
            self.start_freq_in.setValue(219.0)
            self.stop_freq_in.setValue(225.0)
        if self.preset_in.currentText() == "Amateur 70cm Band":
            self.start_freq_in.setValue(420.0)
            self.stop_freq_in.setValue(450.0)
        if self.preset_in.currentText() == "TV Channels 14-36 (UHF Band)":
            self.start_freq_in.setValue(470.0)
            self.stop_freq_in.setValue(608.0)
        if self.preset_in.currentText() == "LTE/5G Band 71 Downlink":
            self.start_freq_in.setValue(617.0)
            self.stop_freq_in.setValue(652.0)
        if self.preset_in.currentText() == "LTE/5G Band 71 Uplink":
            self.start_freq_in.setValue(663.0)
            self.stop_freq_in.setValue(698.0)
        if self.preset_in.currentText() == "LTE/5G Band 12 Downlink":
            self.start_freq_in.setValue(728.0)
            self.stop_freq_in.setValue(746.0)
        if self.preset_in.currentText() == "LTE/5G Band 12 Uplink":
            self.start_freq_in.setValue(698.0)
            self.stop_freq_in.setValue(716.0)
        if self.preset_in.currentText() == "LTE/5G Band 13 Downlink":
            self.start_freq_in.setValue(746.0)
            self.stop_freq_in.setValue(756.0)
        if self.preset_in.currentText() == "LTE/5G Band 13 Uplink":
            self.start_freq_in.setValue(777.0)
            self.stop_freq_in.setValue(787.0)
        if self.preset_in.currentText() == "LTE/5G Band 14 Downlink":
            self.start_freq_in.setValue(758.0)
            self.stop_freq_in.setValue(768.0)
        if self.preset_in.currentText() == "LTE/5G Band 13 Uplink":
            self.start_freq_in.setValue(788.0)
            self.stop_freq_in.setValue(798.0)
        if self.preset_in.currentText() == "LTE/5G Band 5 Downlink":
            self.start_freq_in.setValue(869.0)
            self.stop_freq_in.setValue(894.0)
        if self.preset_in.currentText() == "LTE/5G Band 5 Uplink":
            self.start_freq_in.setValue(824.0)
            self.stop_freq_in.setValue(849.0)
        if self.preset_in.currentText() == "Amateur 33cm Band":
            self.start_freq_in.setValue(902.0)
            self.stop_freq_in.setValue(928.0)
        if self.preset_in.currentText() == "Amateur 23cm Band":
            self.start_freq_in.setValue(1240.0)
            self.stop_freq_in.setValue(1300.0)

    def verify_start_freq(self):  # force stop frequency to be greater than start frequency
        self.preset_in.setCurrentIndex(0)
        if self.start_freq_in.value() >= 1700.0:
            self.start_freq_in.setValue(1699.9)
        if self.start_freq_in.value() >= self.stop_freq_in.value():
            self.stop_freq_in.setValue(self.start_freq_in.value() + 0.1)

    def verify_stop_freq(self):  # force start frequency to be less than stop frequency
        self.preset_in.setCurrentIndex(0)
        if self.stop_freq_in.value() <= 30.0:
            self.stop_freq_in.setValue(30.1)
        if self.stop_freq_in.value() <= self.start_freq_in.value():
            self.start_freq_in.setValue(self.stop_freq_in.value() - 0.1)

    def update_filter_length(self):  # automatically calculate filter length based on inputs
        if self.smooth_auto_radio.isChecked():
            self.smoothing_window_length.show()
            self.smoothing_window_label.show()
            start_freq = self.start_freq_in.value() * 1e6
            stop_freq = self.stop_freq_in.value() * 1e6
            fft_length = int(2 ** (self.resolution_in.currentIndex() + 6))

            self.smoothing_window_length.setValue(int((stop_freq-start_freq)*1e-6*fft_length/1024))
            self.smoothing_window_length.setDisabled(True)

    def filter_switched_to_manual(self):  # enable entry in filter length box
        self.smoothing_window_length.show()
        self.smoothing_window_label.show()
        self.smoothing_window_length.setEnabled(True)

    def filter_switched_to_off(self):  # hide filter length box
        self.smoothing_window_length.setDisabled(True)
        self.smoothing_window_length.hide()
        self.smoothing_window_label.hide()

    def cancel_scan(self):  # cancel in progress scan
        self.cancel_scan_clicked = True

    def start_scan(self):  # start scan

        self.cancel_scan_clicked = False

        # disable scan button and enable cancel button
        self.scan_button.setDisabled(True)
        self.cancel_button.setEnabled(True)

        sdr = RtlSdr()

        # read in scan settings from the GUI
        start_freq = self.start_freq_in.value()*1e6
        stop_freq = self.stop_freq_in.value()*1e6
        fft_length = int(2 ** (self.resolution_in.currentIndex() + 6))
        gain = self.gain_in.value()
        smoothing_filter = self.smooth_manual_radio.isChecked() or self.smooth_auto_radio.isChecked()
        smoothing_filter_length = self.smoothing_window_length.value()
        plot_in_db = self.db_option.isChecked()

        # define other variables and set values on the RTL-SDR
        resolution = 2.4e6
        t_s = 1 / resolution
        sdr.sample_rate = resolution
        sdr.gain = gain

        spectrum_array = []
        freq_array = []

        freq = start_freq

        # while the cancel button is not clicked
        while not self.cancel_scan_clicked:

            # set tuning frequency on sdr
            sdr.center_freq = freq

            # read samples from sdr
            samples_raw = sdr.read_samples(4096)

            # discard samples at the beginning
            samples = samples_raw[int(4096 - fft_length):int(4096)]

            # calculate spectrum from samples
            spectrum = fftshift(fft(samples))

            # truncate spectrum to remove band edges
            spectrum_truncated = spectrum[int(fft_length * 1 / 4):int(fft_length * 3 / 4 - 1)]

            # create array of frequencies corresponding to this truncated spectrum
            spectrum_truncated_freq = np.linspace((freq - resolution / 4),
                                                  (freq + resolution / 4 - (1 / fft_length) * resolution),
                                                  num=int(fft_length / 2 - 1))

            # append truncated spectrum and frequencies for the current band to the full arrays
            spectrum_array = np.append(spectrum_array, spectrum_truncated)
            freq_array = np.append(freq_array, spectrum_truncated_freq)

            # set progress bar value
            self.progress_bar.setValue(int(100*(freq-start_freq)/(stop_freq-start_freq)))

            # refresh GUI
            QApplication.processEvents()

            # break out of loop if current frequency is greater than or equal to the stop frequency
            if freq >= stop_freq:
                break

            # increment frequency to scan at
            freq = freq + resolution / 2

        sdr.close()

        # calculate the magnitude spectrum and scale appropriately
        spectrum_mag = [abs(i)*5*t_s for i in spectrum_array]

        # calculate power at each frequency
        psd_array = [i ** 2 * (1000 / fft_length) for i in spectrum_mag]

        # convert frequencies to MHz
        freq_array = [i * 1e-6 for i in freq_array]

        # convert power to dB units if option is selected
        if plot_in_db:
            psd_array = 10 * log10(psd_array)
            y_label_string = "Power (dBmW/Hz)"
        else:
            y_label_string = "Power (mW/Hz)"

        # smooth the data if option is selected
        if smoothing_filter:

            # force the smoothing filter length to be length than the length of the PSD array
            if len(psd_array) < smoothing_filter_length:
                smoothing_filter_length = len(psd_array)

            psd_array = savgol_filter(psd_array, smoothing_filter_length, 1)

        # close previously created Matplotlib plot
        close()

        # update progress bar to 100
        self.progress_bar.setValue(100)

        # plot the PSD with Matplotlib
        plt.figure("SpecScan Spectrum Viewer")
        plt.title("Power Spectral Density")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel(y_label_string)
        plt.plot(freq_array, psd_array, color="red")
        plt.xlim([start_freq*1e-6, stop_freq*1e-6])
        plt.grid()
        show()

        # Disable cancel button and re-enable scan button
        self.cancel_button.setDisabled(True)
        self.scan_button.setEnabled(True)


app = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(app.exec())
