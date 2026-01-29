import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import(
    QApplication, QMainWindow, QWidget, QFileDialog, 
    QLabel, QAction
)
from PyQt5.QtGui import QPainter, QImage, QPixmap, QPen, QCursor
from PyQt5.QtCore import Qt, QPoint


class LineMaskTool(QMainWindow):
    # Definiranje općih varijabli i glavnog prozora
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Line Mask Labeler")
        self.resize(1366, 768)       

        self.image = None
        self.qimage = None
        self.mask = None

        self.SAVE_PATH = "/home/tomo/Faks/ProjektE/Code_and_Data/wire_detection_algorithm/GUI_labeling/masks"
        os.makedirs(self.SAVE_PATH, exist_ok=True)

        self.lines = []
        self.start_point = None

        self.images_path = [] 
        self.current_index = 0
        self.image_path = ""

        # Postavke za ToolBar
        self._createToolBars()
        self.setContextMenuPolicy(Qt.NoContextMenu)

        # Custom cursor za preciznije označavanje (crosshair.png)
        crosshair = QPixmap("crosshair.png")
        cursor = QCursor(crosshair)
        self.setCursor(cursor)

        self.load_image()

    # Stvaranje ToolBar-a za funkcije: next_image, previous_image, save_image, itd.
    def _createToolBars(self):
        toolbar = self.addToolBar("Tools")

        next_action = QAction("NEXT", self)
        next_action.triggered.connect(self.next_image)

        previous_action = QAction("PREVIOUS", self)
        previous_action.triggered.connect(self.previous_image)

        save_action = QAction("SAVE", self)
        save_action.triggered.connect(self.save_image)

        toolbar.addAction(next_action)
        toolbar.addAction(previous_action)
        toolbar.addAction(save_action)

    # Učitavanje slika (treba doraditi i dodati gumb za slijedno listanje po direktoriju koristeći gumb)
    def load_image(self):

        # 1. Dohvačanje path-a prve slike
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            sys.exit()

        # 2. Dohvaćanje direktorija i ostalih slika
        self.dir_path = os.path.dirname(path)
        images_path = []
        for f in os.listdir(self.dir_path):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                full_path = os.path.join(self.dir_path, f)
                images_path.append(full_path)

        images_path.sort()
        self.images_path = images_path

        self.current_index = self.images_path.index(path)
        self.load_next_image(self.current_index)

    # Učitavanje sljedeće slike za prikazivanje (ovisno o akciji next_image/previous_image)
    def load_next_image(self, index):
        self.image_path = self.images_path[index]

        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        h, w, _ = self.image.shape
        self.mask = np.zeros((h, w), dtype=np.uint8)

        self.lines = []
        self.start_point = None

        self.qimage = QImage(
            self.image.data, w, h, 3 * w, QImage.Format_RGB888
        )

        self.update()

    # Akcija za prijelaz na sljedeću sliku
    def next_image(self):
        self.current_index += 1
        if self.current_index >= len(self.images_path):
            self.current_index = 0
        self.load_next_image(self.current_index)

    # Akcija za prijelaz na prethodnu sliku
    def previous_image(self):
        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = len(self.images_path) - 1
        self.load_next_image(self.current_index)    

    # Definiranje koordinata početne točke (x1, y1) linije i konačne točke (x2, y2)
    def mousePressEvent(self, event):
        pos = event.pos()

        # Računanje offseta pozicije miša za sliku
        x = pos.x() - self.offset_x
        y = pos.y() - self.offset_y

        # Zanemarivanje klikova izvan slike
        if x < 0 or y < 0 or x >= self.qimage.width() or y >= self.qimage.height():
            return

        if event.button() == Qt.LeftButton:
            if self.start_point is None:
                self.start_point = QPoint(x, y)
            else:
                x1, y1 = self.start_point.x(), self.start_point.y()
                x2, y2 = x, y

                self.lines.append((x1, y1, x2, y2))
                cv2.line(self.mask, (x1, y1), (x2, y2), 255, 2)

                self.start_point = None
                self.update()

        elif event.button() == Qt.RightButton:
            if self.lines:
                self.lines.pop()
                self.mask[:] = 0 # Postavljanje svakog piksela maske na 0(crno)
                for l in self.lines:
                    cv2.line(self.mask, (l[0], l[1]), (l[2], l[3]), 255, 2)
                self.update()

    # Crtanje
    def paintEvent(self, event):

        painter = QPainter(self)

        # Računanje offseta prozora i slike
        window_w = self.width() 
        window_h = self.height()
        image_w = self.qimage.width()
        image_h = self.qimage.height()

        self.offset_x = (window_w - image_w)//2
        self.offset_y = (window_h - image_h)//2

        # Prikaz bazne slike
        painter.drawImage(self.offset_x, self.offset_y, self.qimage)

        # Crtanje linije (vizualizacija)
        painter.setPen(QPen(Qt.red, 2)) # Korištenje crvenog markera za vizualizaciju
        for x1, y1, x2, y2 in self.lines:
            painter.drawLine(
                x1 + self.offset_x,
                y1 + self.offset_y,
                x2 + self.offset_x,
                y2 + self.offset_y
                )

        # Crtanje maske preko bazne slike (bolja vizualizacija nego bez)
        overlay = np.zeros((self.mask.shape[0], self.mask.shape[1], 4), dtype=np.uint8)
        overlay[self.mask > 0] = [255, 0, 0, 100]  # Crveni marker

        h, w, _ = overlay.shape
        qoverlay = QImage(
            overlay.data, w, h, 4 * w, QImage.Format_RGBA8888
        )

        painter.drawImage(self.offset_x, self.offset_y, qoverlay)

    # Spremanje maske linija koristeći ToolBar
    def save_image(self):
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        save_path = os.path.join(self.SAVE_PATH, f"{base}.png")
        cv2.imwrite(save_path, self.mask)

    # Spremanje maske linija koristeći event.key()
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            save_path = os.path.join(self.SAVE_PATH, f"{base}.png")
            cv2.imwrite(save_path, self.mask)

    # Spremanje zadnje uređivane slike/indexa prije zatvaranja prozora
    def closeEvent(self, event):
        print(f"{os.path.splitext(os.path.basename(self.images_path[self.current_index]))[0]}")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tool = LineMaskTool()
    tool.show()
    if (sys.exit(app.exec_())):
        print(f"Last image index/name: {os.path.splitext(os.path.basename(self.images_path[index]))[0]}")
