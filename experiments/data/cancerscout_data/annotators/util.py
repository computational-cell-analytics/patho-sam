import napari
import numpy as np
from micro_sam.sam_annotator.object_classifier import object_classifier
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class ClassCountWidget(QWidget):
    """Widget that displays instance counts per class in the annotations layer."""

    def __init__(
        self,
        viewer: napari.Viewer,
        labels: dict,
        annotation_layer_name: str = "annotations",
        segmentation_layer_name: str = "segmentation",
    ):
        super().__init__()
        self.viewer = viewer
        self.labels = labels  # e.g. {"Tumor": 1, "Stroma": 2, ...}
        self.annotation_layer_name = annotation_layer_name
        self.segmentation_layer_name = segmentation_layer_name

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- Title ---
        title = QLabel("<b>Class Instance Counts</b>")
        layout.addWidget(title)

        # --- Table: one row per class ---
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setRowCount(len(self.labels))
        self.table.setHorizontalHeaderLabels(["Class", "Count"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Pre-fill class names with placeholder counts
        for row, class_name in enumerate(self.labels.keys()):
            self.table.setItem(row, 0, QTableWidgetItem(class_name))
            self.table.setItem(row, 1, QTableWidgetItem("-"))

        self._fit_table_height()
        layout.addWidget(self.table)

        # --- Refresh button ---
        self.refresh_btn = QPushButton("Refresh Counts")
        self.refresh_btn.clicked.connect(self.refresh_counts)
        layout.addWidget(self.refresh_btn)

    def _fit_table_height(self):
        """Resize table height to fit all rows without scrolling."""
        header_height = self.table.horizontalHeader().height()
        row_height = sum(self.table.rowHeight(row) for row in range(self.table.rowCount()))
        self.table.setFixedHeight(header_height + row_height + 2)

    def refresh_counts(self):
        """
        For each instance in the segmentation layer, look up which pixels
        it covers in the annotation layer, take the majority class label,
        and count instances per class.
        """
        # --- Validate layers exist ---
        for layer_name in [self.annotation_layer_name, self.segmentation_layer_name]:
            if layer_name not in self.viewer.layers:
                print(f"Layer '{layer_name}' not found.")
                self._set_all_counts("-")
                return

        annotation_data = np.asarray(self.viewer.layers[self.annotation_layer_name].data)
        segmentation_data = np.asarray(self.viewer.layers[self.segmentation_layer_name].data)

        if annotation_data.shape != segmentation_data.shape:
            print(f"Shape mismatch: annotations {annotation_data.shape} vs segmentation {segmentation_data.shape}")
            self._set_all_counts("err")
            return

        counts = self._count_instances_per_class(annotation_data, segmentation_data)

        for row, (class_name, label_value) in enumerate(self.labels.items()):
            count = counts.get(label_value, 0)
            self.table.setItem(row, 0, QTableWidgetItem(class_name))
            self.table.setItem(row, 1, QTableWidgetItem(str(count)))

    def _count_instances_per_class(self, annotation_data: np.ndarray, segmentation_data: np.ndarray) -> dict:
        """
        For each class label in the annotation layer, find the pixels
        belonging to that class and count unique instance IDs from the
        segmentation layer at those pixels.

        This is much faster than iterating over every instance ID.
        """
        counts = {}

        for label_value in self.labels.values():
            # Mask of all pixels annotated as this class
            class_mask = annotation_data == label_value

            if not np.any(class_mask):
                counts[label_value] = 0
                continue

            # Grab segmentation instance IDs at those pixels
            instance_ids = segmentation_data[class_mask]

            # Count unique non-background instances
            unique_ids = np.unique(instance_ids)
            unique_ids = unique_ids[unique_ids != 0]

            counts[label_value] = len(unique_ids)

        return counts

    def _set_all_counts(self, value: str):
        for row, class_name in enumerate(self.labels.keys()):
            self.table.setItem(row, 0, QTableWidgetItem(class_name))
            self.table.setItem(row, 1, QTableWidgetItem(value))


def get_grid(img_shape, cell_width=512):
    """Add Grid for better overview in annotated regions"""
    return [
        np.array([[0, cell_width * i], [img_shape[0], cell_width * i]])
        for i in range(0, img_shape[1] // cell_width + 1)
    ] + [
        np.array([[cell_width * i, 0], [cell_width * i, img_shape[1]]])
        for i in range(0, img_shape[0] // cell_width + 1)
    ]


def get_object_classifier_viewer(image, segmentation, embedding_path, **viewer_kwargs):
    tile_shape, halo = (384, 384), (64, 64)
    viewer = object_classifier(
        image=image,
        segmentation=segmentation,
        model_type="vit_b_histopathology",
        embedding_path=embedding_path,
        tile_shape=tile_shape,
        halo=halo,
        ndim=2,
        return_viewer=True,
        **viewer_kwargs,
    )
    return viewer
