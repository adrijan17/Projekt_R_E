import sys
import cv2
import json
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtGui import QPainter, QImage, QPixmap, QPen
from PyQt5.QtCore import Qt, QPoint


# Treba doraditi GUI korištenjem designera i dodati još akcija (redo/undo, next, itd.)


class LineMaskTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Line Mask Labeler")

        self.image = None
        self.qimage = None
        self.mask = None

        self.lines = []
        self.start_point = None

        self.load_image()

    # Učitavanje slika (treba doraditi i dodati gumb za slijedno listanje po direktoriju koristeći gumb)
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg)"
        )
        if not path:
            sys.exit()

        self.image = cv2.imread(path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        h, w, _ = self.image.shape
        self.mask = np.zeros((h, w), dtype=np.uint8)

        self.qimage = QImage(
            self.image.data, w, h, 3 * w, QImage.Format_RGB888
        )

        self.resize(w, h)

    # Definiranje koordinata početne točke (x1, y1) linije i konačne točke (x2, y2)
    def mousePressEvent(self, event):
        pos = event.pos()

        if event.button() == Qt.LeftButton:
            if self.start_point is None:
                self.start_point = pos
            else:
                x1, y1 = self.start_point.x(), self.start_point.y()
                x2, y2 = pos.x(), pos.y()

                self.lines.append((x1, y1, x2, y2))
                cv2.line(self.mask, (x1, y1), (x2, y2), 255, 2)

                self.start_point = None
                self.update()

        elif event.button() == Qt.RightButton:
            if self.lines:
                self.lines.pop()
                self.mask[:] = 0
                for l in self.lines:
                    cv2.line(self.mask, (l[0], l[1]), (l[2], l[3]), 255, 2)
                self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.qimage)

        # Crtanje linije (vizualizacija)
        painter.setPen(QPen(Qt.red, 2))
        for x1, y1, x2, y2 in self.lines:
            painter.drawLine(x1, y1, x2, y2)

        # Crtanje maske
        overlay = np.zeros((self.mask.shape[0], self.mask.shape[1], 4), dtype=np.uint8)
        overlay[self.mask > 0] = [255, 0, 0, 100]  # Crveni marker

        h, w, _ = overlay.shape
        qoverlay = QImage(
            overlay.data, w, h, 4 * w, QImage.Format_RGBA8888
        )

        painter.drawImage(0, 0, qoverlay)

    # Spremanje maske linija
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            cv2.imwrite("line_mask.png", self.mask)
            print("Mask saved as line_mask.png")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    tool = LineMaskTool()
    tool.show()
    sys.exit(app.exec_())
