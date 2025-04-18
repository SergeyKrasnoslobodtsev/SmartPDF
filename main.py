from dataclasses import dataclass, field
import logging
# from operator import le
from typing import Any, Dict, List, Tuple
from pdf2image import convert_from_path

import cv2 as cv
import numpy as np
from pytesseract import Output
import pytesseract

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Доменные модели (Domain Models)
@dataclass
class Text:
    """Сущность текст"""
    x: int
    y: int
    w: int
    h: int
    level: int
    block_num: int
    par_num: int
    line_num: int
    word_num: int
    conf: int
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinates": (self.x, self.y, self.w, self.h),
            "text": self.text,
            "level": self.level,
            "block_num": self.block_num,
            "par_num": self.par_num,
            "line_num": self.line_num,
            "word_num": self.word_num,
            "conf": self.conf,
        }
@dataclass
class Cell:
    """Сущность ячейки таблицы"""
    x: int
    y: int
    w: int
    h: int
    text: List[Text] = field(default_factory=list)
    column_index: int = -1
    row_index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinates": (self.x, self.y, self.w, self.h),
            "text": self.text,
            "column": self.column_index,
            "row": self.row_index
        }

@dataclass
class Column:
    """Сущность колонки таблицы"""
    x: int
    cells: List[Cell] = field(default_factory=list)
    _width: int = field(init=False, default=0)
    _avg_x: float = field(init=False, default=0.0)

    def add_cell(self, cell: Cell) -> None:
        """Добавление ячейки с обновлением метрик"""
        self.cells.append(cell)
        self._calculate_metrics()

    def __post_init__(self):
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        if self.cells:
            self._width = max(cell.w for cell in self.cells)
            x_coords = [cell.x for cell in self.cells]
            self._avg_x = sum(x_coords) / len(x_coords)

    @property
    def width(self) -> int:
        return self._width

    @property
    def avg_x(self) -> float:
        return self._avg_x

@dataclass
class Row:
    """Сущность строки таблицы"""
    cells: List[Cell] = field(default_factory=list)
    _height: int = field(init=False, default=0)
    _avg_y: float = field(init=False, default=0.0)

    def add_cell(self, cell: Cell) -> None:
        """Добавление ячейки с обновлением метрик"""
        self.cells.append(cell)
        self._calculate_metrics()

    def __post_init__(self):
        self._calculate_metrics()

    def _calculate_metrics(self) -> None:
        if self.cells:
            self._avg_y = min(cell.y for cell in self.cells)
            self._height = max(cell.h for cell in self.cells)
    @property
    def height(self) -> int:
        return self._height

    @property
    def avg_y(self) -> float:
        return self._avg_y

@dataclass
class Table:
    """Сущность таблицы"""
    x: int
    y: int
    w: int
    h: int
    rows: List[Row] = field(default_factory=list)
    columns: List[Column] = field(default_factory=list)

    def add_row(self, cells: List[Cell]) -> None:
        """Группировка ячеек по строкам на основе Y-координаты"""
        sorted_cells = sorted(cells, key=lambda c: c.y)
        grouped_rows: Dict[int, List[Cell]] = {}
        y_tolerance = 5

        for cell in sorted_cells:
            added = False
            for y in grouped_rows.keys():
                if abs(cell.y - y) <= y_tolerance:
                    grouped_rows[y].append(cell)
                    added = True
                    break
            if not added:
                grouped_rows[cell.y] = [cell]

        # Создаем строки из сгруппированных ячеек
        for i, (_, row_cells) in enumerate(sorted(grouped_rows.items())):
            row = Row()
            sorted_row_cells = sorted(row_cells, key=lambda c: c.x)
            for cell in sorted_row_cells:
                cell.row_index = i
                row.add_cell(cell)
            self.rows.append(row)

        self._update_columns()

    def _update_columns(self, x_tolerance: int = 5) -> None:
        """Обновить колонки после добавления ячеек"""
        self.columns.clear()
        all_cells = [cell for row in self.rows for cell in row.cells]
        sorted_cells = sorted(all_cells, key=lambda c: c.x)

        for cell in sorted_cells:
            added = False
            for col in self.columns:
                if abs(cell.x - col.avg_x) <= x_tolerance:
                    col.add_cell(cell)
                    cell.column_index = self.columns.index(col)
                    added = True
                    break
            if not added:
                new_column = Column(cell.x, [cell])
                cell.column_index = len(self.columns)
                self.columns.append(new_column)
    

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.columns)

class TableDetector():
    """Реализация детектора таблиц на основе OpenCV"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def detect(self, image: np.ndarray) -> List[Table]:
        """Реализация обнаружения таблиц через OpenCV"""
        binary_image = self._detect_lines(self._image_processing(image))

        # cv.namedWindow('lines', cv.WINDOW_NORMAL)      
        # cv.imshow('lines', binary_image)
        # cv.resizeWindow('lines', 640, 480)
        rects = self._get_rects(binary_image)
        
        tables: List[Table] = []

        for rect in rects:
            inner_cells = [
                inner for inner in rects 
                if self._is_inside(inner, rect) and inner != rect
            ]
            
            if inner_cells:
                x, y, w, h = rect
                table = Table(x, y, w, h)
                cells = [Cell(cx, cy, cw, ch) for cx, cy, cw, ch in inner_cells]
                table.add_row(cells)
                tables.append(table)

        return tables

    def _is_inside(self, inner, outer):
        """Проверка, вложен ли один прямоугольник в другой"""
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        return ox <= ix and oy <= iy and (ix + iw) <= (ox + ow) and (iy + ih) <= (oy + oh)

    def _image_processing(self, image: np.ndarray) -> np.ndarray:
        img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, img_bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        img_bin = 255-img_bin

        return img_bin
    
    def _get_rects(self, image: np.ndarray) -> Tuple[int, int, int, int]:
         # Поиск контуров
        contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Создание списка координат
        rects = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w > 10 and h > 10:
                rects.append((x, y, w, h))

        return rects
    
    def _detect_lines(self, image:np.ndarray) -> np.ndarray:

        # Обнаружение линий
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, np.array(image).shape[1]//250))
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (np.array(image).shape[1]//200, 1))

        # Обработка вертикальных линий
        eroded_image_ver = cv.erode(image, vertical_kernel, iterations=5)
        vertical_lines = cv.dilate(eroded_image_ver, vertical_kernel, iterations=5)

        # Обработка горизонтальных линий
        eroded_image_hor = cv.erode(image, horizontal_kernel, iterations=5)
        horizontal_lines = cv.dilate(eroded_image_hor, horizontal_kernel, iterations=5)

        # Объединение линий
        combined_lines = cv.bitwise_or(vertical_lines, horizontal_lines)

        return combined_lines

class TextDetector:
    def __init__(self, config: Dict[str, Any]):
       self.config = {
           "lang": config.get("lang", "rus+eng"),  # Язык
           "psm": config.get("psm", 4),  # Единый блок текста
           "oem": config.get("oem", 1)   # LSTM + legacy
       }

    def detect(self, image: np.ndarray) -> List[Text]:
        # Предобработка изображения для улучшения распознавания
        processed = self._preprocess_image(image)
        
        custom_config = f'--oem {self.config["oem"]} --psm {self.config["psm"]}'

        data = pytesseract.image_to_data(
            processed, 
            lang=self.config["lang"],
            config=custom_config,
            output_type=Output.DICT
        )

        results:List[Text] = []
        for i in range(0, len(data['text'])):
            conf = int(data['conf'][i])
            if conf > 50:
                results.append(Text(
                    data['left'][i],
                    data['top'][i],
                    data['width'][i],
                    data['height'][i],
                    data['level'][i],
                    data['block_num'][i],
                    data['par_num'][i],
                    data['line_num'][i],
                    data['word_num'][i],
                    conf,
                    data['text'][i]
                ))
        
        return results

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Конвертация в grayscale если нужно
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image

        # Адаптивный порог для лучшего контраста
        thresh = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv.THRESH_BINARY, 11, 2
        )
        
    
        # Удаление шума
        denoised = cv.fastNlMeansDenoising(thresh)
       
        return gray

# Основной класс OCR
class OCR:
    """Основной класс для оптического распознавания"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.table_detector = TableDetector(config)
        self.text_detector = TextDetector(config)

    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Обработка изображения"""
        try:
            # Обнаружение таблиц
            tables = self.table_detector.detect(image)
            
            # Обработка каждой таблицы
            result = []
            for table in tables:
                # Обработка текста в каждой ячейке
                for row in table.rows:
                    for cell in row.cells:
                        cell_image = image[cell.y:cell.y + cell.h, cell.x:cell.x + cell.w]
                        cell.text = self.text_detector.detect(cell_image)
                result.append(table)

            return {
                "tables": result,
                "table_count": len(tables),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                "tables": [],
                "table_count": 0,
                "status": "error",
                "error": str(e)
            }

def visualize_tables(image: np.ndarray, result: Dict[str, Any]) -> None:
    from typing import cast
    
    if result["status"] != "success":
        logger.debug(f"Error: {result.get('error')}")
        return

    tables = cast(List[Table], result['tables'])
    logger.debug(f"Found {len(tables)} tables")
    
    for idx, table in enumerate(tables):
        logger.debug(f"\nTable {idx}:")
        logger.debug(f"Rows: {table.row_count}")
        logger.debug(f"Columns: {table.column_count}")
        
        # Draw table borders
        cv.rectangle(image, (table.x, table.y), 
                    (table.x + table.w, table.y + table.h), 
                    (255, 0, 0), 2)
        
        # Draw cells with indices
        for i, row in enumerate(table.rows):
            for j, cell in enumerate(row.cells):
                # cv.rectangle(image, (cell.x, cell.y),
                #             (cell.x + cell.w, cell.y + cell.h),
                #             (0, 255, 0), 1)
                          
                text = f"{i}:{j}"
                cv.putText(image, text,
                            (cell.x + 5, cell.y + 20),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
                for text in cell.text:
                     cv.rectangle(image, (cell.x + text.x, cell.y + text.y),
                            (text.x + cell.x + text.w, text.y + cell.y + text.h),
                            (0, 0, 255), 1)
    
    cv.namedWindow('Tables', cv.WINDOW_NORMAL)      
    cv.imshow('Tables', image)
    cv.resizeWindow('Tables', 640, 480)
    cv.waitKey(0)
    cv.destroyAllWindows()

def save_table_to_csv(result: Dict[str, Any], output_file: str) -> None:
    import csv
    """Сохранение таблицы в CSV файл"""        
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Table', 'Row', 'Column', 'Text'])
        
        for table_idx, table in enumerate(result['tables']):
            for i, row in enumerate(table.rows):
                for j, cell in enumerate(row.cells):
                    # text = f"{float(cell.text):.2f}" if cell.text else "0.00"
                    writer.writerow([table_idx, i, j, [cell.text[i].text for i in range(len(cell.text))]])

config = {
   "table_detector": {
       "min_size": 10,
       "kernel_division": 150,
       "iterations": 5,
       "x_tolerance": 5,
       "y_tolerance": 5
   },
   "text_detector": {
       "lang": "rus+eng",  # Язык
       "psm": 6,  # 6 - однородный блок текста
       "oem": 1   # 3 - Использовать LSTM и legacy движок
   }
}

def tesseract_test(image):
    ocr = OCR(config)
    result = ocr.process_image(img)
    visualize_tables(img, result)
    save_table_to_csv(result, 'decode_text.csv')

# TODO: Проверить работу с PyMuPDF 
# https://medium.com/@pymupdf/table-recognition-and-extraction-with-pymupdf-54e54b40b760
# 

def ocr_img2table(image: np.ndarray) -> None:
    from img2table.document import Image
    from img2table.ocr import TesseractOCR
    from img2table.ocr import PaddleOCR

    # convert image To bytes
    
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_byte = cv.imencode('.png', image)[1].tobytes()
    paddle_ocr = PaddleOCR(lang="ru", kw={"use_dilation": True})
    # ocr = TesseractOCR(lang="rus",n_threads=4, psm=3)
    doc = Image(image_byte, detect_rotation=False)
    extracted_tables = doc.extract_tables()

    for table in extracted_tables:
        for row in table.content.values():
            for cell in row:
                cv.rectangle(image, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (255, 0, 0), 1)
                
    cv.imshow("Image with detected tables", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    doc.to_xlsx(
        dest="output.xlsx",
        ocr=paddle_ocr, min_confidence=50)
  

# TODO: Tesseract плохо работает с отдельными ячейками для этого необходимо добавить рамку
# https://groups.google.com/g/tesseract-ocr/c/v26a-RYPSOE/m/2Sppq61GBwAJ




if __name__ == "__main__":
    file_path = "D:\workspace\docs\pdf\Акт сверки взаимных расчетов приложение (ЭДО) № КР00-001215 от 25.04.2024.pdf"
    # https://github.com/tesseract-ocr/tessdoc/blob/main/ImproveQuality.md
    # именно tesseract работает с изображениями, имеющими dpi 300 и выше
    images = convert_from_path(file_path, dpi=400)
    img = np.array(images[0])
    # ocr_img2table(img)
    tesseract_test(img)
    

    # text_detection_test(img)


  
    
    