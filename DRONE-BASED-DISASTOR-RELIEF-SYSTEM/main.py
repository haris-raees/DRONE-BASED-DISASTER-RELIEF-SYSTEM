import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QFileDialog,
                             QVBoxLayout, QWidget, QMessageBox)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from functions import create_directories, stitch_images, extract_coordinates, generate_drone_images

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Drone Image Processing')
        self.setGeometry(100, 100, 800, 600)

        self.upload_folder = 'images/uploads/'
        self.processed_folder = 'images/processed/'
        create_directories()

        self.initUI()

    def initUI(self):
        """Initializes the main UI components."""
        main_layout = QVBoxLayout()

        self.upload_button = QPushButton('Upload Images')
        self.upload_button.clicked.connect(self.upload_images)
        main_layout.addWidget(self.upload_button)

        self.stitch_button = QPushButton('Stitch Images')
        self.stitch_button.clicked.connect(self.stitch_images)
        main_layout.addWidget(self.stitch_button)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)

        self.next_button = QPushButton('Next')
        self.next_button.clicked.connect(self.go_to_process_window)
        main_layout.addWidget(self.next_button)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def upload_images(self):
        """Handles image upload functionality."""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images Files (*.png *.jpg *.bmp)")
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(self.upload_folder, file_name)
            with open(file_path, 'rb') as f:
                with open(dest_path, 'wb') as dest:
                    dest.write(f.read())
        self.show_message("Images uploaded successfully!")

    def stitch_images(self):
        """Handles image stitching functionality."""
        try:
            stitched_image_path = stitch_images(self.upload_folder)
            self.show_image(stitched_image_path)
        except Exception as e:
            self.show_message(f"Error stitching images: {str(e)}")

    def show_image(self, image_path):
        """Displays the image on the label."""
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def show_message(self, message):
        """Displays a message box."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Info")
        msg.exec_()

    def go_to_process_window(self):
        """Navigates to the process window."""
        self.process_window = ProcessWindow(self)
        self.process_window.show()
        self.hide()

class ProcessWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Process Coordinates')
        self.setGeometry(100, 100, 800, 600)
        self.stitched_image_path = os.path.join('images/uploads', 'stitched_image.jpg')
        self.start_coords = None
        self.initUI()

    def initUI(self):
        """Initializes the process window UI components."""
        main_layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(self.stitched_image_path)
        self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        main_layout.addWidget(self.image_label)

        self.fetch_button = QPushButton('Fetch Coordinates')
        self.fetch_button.clicked.connect(self.fetch_coordinates)
        main_layout.addWidget(self.fetch_button)

        self.coords_label = QLineEdit()
        self.coords_label.setReadOnly(True)
        main_layout.addWidget(self.coords_label)

        self.next_button = QPushButton('Next')
        self.next_button.clicked.connect(self.go_to_generate_window)
        main_layout.addWidget(self.next_button)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def fetch_coordinates(self):
        """Fetches coordinates from the stitched image."""
        try:
            start_coords = extract_coordinates(self.stitched_image_path)
            self.start_coords = start_coords
            self.coords_label.setText(f"Start: {', '.join(map(str, start_coords))}")
        except Exception as e:
            self.show_message(f"Error fetching coordinates: {str(e)}")

    def go_to_generate_window(self):
        """Navigates to the generate window if coordinates are set."""
        if self.start_coords:
            self.generate_window = GenerateWindow(self.start_coords)
            self.generate_window.show()
            self.hide()
        else:
            self.show_message("Please fetch coordinates first!")

    def show_message(self, message):
        """Displays a message box."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Info")
        msg.exec_()

class GenerateWindow(QMainWindow):
    def __init__(self, start_coords, parent=None):
        super().__init__(parent)
        self.start_coords = start_coords
        self.initUI()

    def initUI(self):
        """Initializes the generate window UI components."""
        self.setWindowTitle('Generate Drone Movements')
        self.setGeometry(100, 100, 800, 600)

        main_layout = QVBoxLayout()
        # main_layout.setSpacing(0)  # Set spacing between widgets
        # main_layout.setContentsMargins(10, 10, 10, 10)

        self.coord_input = QLineEdit(self)
        self.coord_input.setPlaceholderText('Enter Starting Coordinates')
        self.coord_input.setText(', '.join(map(str, self.start_coords)))
        main_layout.addWidget(self.coord_input)

        self.generate_btn = QPushButton('Generate Images', self)
        self.generate_btn.clicked.connect(self.generate_images)
        main_layout.addWidget(self.generate_btn)

        self.first_picture_label = QLabel('First Picture:', self)
        self.first_picture = QLabel(self)
        self.first_picture.setFixedSize(500, 200)
        self.first_picture.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.first_picture_label)
        main_layout.addWidget(self.first_picture)

        self.reach_dest_label = QLabel('Picture After Reaching Destination:', self)
        self.reach_dest_picture = QLabel(self)
        self.reach_dest_picture.setFixedSize(500, 200)
        self.reach_dest_picture.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.reach_dest_label)
        main_layout.addWidget(self.reach_dest_picture)

        self.return_start_label = QLabel('Picture After Returning to Starting Point:', self)
        self.return_start_picture = QLabel(self)
        self.return_start_picture.setFixedSize(500, 200)
        self.return_start_picture.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.return_start_label)
        main_layout.addWidget(self.return_start_picture)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def generate_images(self):
        """Generates drone images based on the provided coordinates."""
        processed_folder = 'images/processed/'
        first_img_path = os.path.join(processed_folder, 'output_image.jpg')
        try:
            start_coords = tuple(map(int, self.coord_input.text().split(',')))
            dest_img_path, return_img_path = generate_drone_images(start_coords)
            
            self.first_picture.setPixmap(QPixmap(first_img_path).scaled(self.first_picture.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.reach_dest_picture.setPixmap(QPixmap(dest_img_path).scaled(self.reach_dest_picture.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.return_start_picture.setPixmap(QPixmap(return_img_path).scaled(self.return_start_picture.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.show_message(f"Error generating images: {str(e)}")

    def show_message(self, message):
        """Displays a message box."""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setWindowTitle("Info")
        msg.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
