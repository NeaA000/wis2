"""
GUI 스타일 정의
"""
import darkdetect

def get_theme():
    """시스템 테마 감지"""
    return "dark" if darkdetect.isDark() else "light"

# 색상 팔레트
COLORS = {
    "dark": {
        "bg": "#1e1e1e",
        "bg_secondary": "#2d2d2d",
        "bg_hover": "#3d3d3d",
        "text": "#ffffff",
        "text_secondary": "#b0b0b0",
        "accent": "#0d7377",
        "accent_hover": "#14b8bd",
        "border": "#404040",
        "success": "#4CAF50",
        "error": "#f44336",
        "warning": "#ff9800"
    },
    "light": {
        "bg": "#ffffff",
        "bg_secondary": "#f5f5f5",
        "bg_hover": "#e8e8e8",
        "text": "#1e1e1e",
        "text_secondary": "#666666",
        "accent": "#0d7377",
        "accent_hover": "#14b8bd",
        "border": "#e0e0e0",
        "success": "#4CAF50",
        "error": "#f44336",
        "warning": "#ff9800"
    }
}

def get_stylesheet(theme="auto"):
    """전체 애플리케이션 스타일시트 반환"""
    if theme == "auto":
        theme = get_theme()
    
    colors = COLORS[theme]
    
    return f"""
    /* 메인 윈도우 */
    QMainWindow {{
        background-color: {colors['bg']};
        color: {colors['text']};
    }}
    
    /* 기본 위젯 */
    QWidget {{
        background-color: {colors['bg']};
        color: {colors['text']};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 14px;
    }}
    
    /* 버튼 */
    QPushButton {{
        background-color: {colors['accent']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        min-height: 36px;
    }}
    
    QPushButton:hover {{
        background-color: {colors['accent_hover']};
    }}
    
    QPushButton:pressed {{
        background-color: {colors['accent']};
    }}
    
    QPushButton:disabled {{
        background-color: {colors['bg_secondary']};
        color: {colors['text_secondary']};
    }}
    
    /* 보조 버튼 */
    QPushButton.secondary {{
        background-color: {colors['bg_secondary']};
        color: {colors['text']};
    }}
    
    QPushButton.secondary:hover {{
        background-color: {colors['bg_hover']};
    }}
    
    /* 레이블 */
    QLabel {{
        color: {colors['text']};
    }}
    
    QLabel.secondary {{
        color: {colors['text_secondary']};
    }}
    
    QLabel.heading {{
        font-size: 24px;
        font-weight: 600;
        margin: 10px 0;
    }}
    
    QLabel.subheading {{
        font-size: 18px;
        font-weight: 500;
        margin: 8px 0;
    }}
    
    /* 콤보박스 */
    QComboBox {{
        background-color: {colors['bg_secondary']};
        border: 1px solid {colors['border']};
        border-radius: 6px;
        padding: 8px 12px;
        min-height: 36px;
    }}
    
    QComboBox:hover {{
        border-color: {colors['accent']};
    }}
    
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    
    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid {colors['text']};
        margin-right: 5px;
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {colors['bg_secondary']};
        border: 1px solid {colors['border']};
        selection-background-color: {colors['accent']};
        selection-color: white;
    }}
    
    /* 프로그레스 바 */
    QProgressBar {{
        background-color: {colors['bg_secondary']};
        border: 1px solid {colors['border']};
        border-radius: 6px;
        height: 8px;
        text-align: center;
    }}
    
    QProgressBar::chunk {{
        background-color: {colors['accent']};
        border-radius: 4px;
    }}
    
    /* 리스트 위젯 */
    QListWidget {{
        background-color: {colors['bg_secondary']};
        border: 1px solid {colors['border']};
        border-radius: 6px;
        padding: 8px;
        outline: none;
    }}
    
    QListWidget::item {{
        padding: 8px;
        border-radius: 4px;
    }}
    
    QListWidget::item:hover {{
        background-color: {colors['bg_hover']};
    }}
    
    QListWidget::item:selected {{
        background-color: {colors['accent']};
        color: white;
    }}
    
    /* 체크박스 */
    QCheckBox {{
        spacing: 8px;
    }}
    
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {colors['border']};
        border-radius: 4px;
        background-color: {colors['bg_secondary']};
    }}
    
    QCheckBox::indicator:checked {{
        background-color: {colors['accent']};
        border-color: {colors['accent']};
        image: url(checkmark.png);
    }}
    
    /* 스크롤바 */
    QScrollBar:vertical {{
        background-color: {colors['bg']};
        width: 12px;
        border: none;
    }}
    
    QScrollBar::handle:vertical {{
        background-color: {colors['bg_hover']};
        border-radius: 6px;
        min-height: 20px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background-color: {colors['text_secondary']};
    }}
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        border: none;
        background: none;
    }}
    
    /* 그룹박스 */
    QGroupBox {{
        background-color: {colors['bg_secondary']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 12px;
        font-weight: 500;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 8px;
        background-color: {colors['bg_secondary']};
        color: {colors['text']};
    }}
    
    /* 드롭 영역 스타일 */
    DropZone {{
        background-color: {colors['bg_secondary']};
        border: 2px dashed {colors['border']};
        border-radius: 12px;
        min-height: 200px;
    }}
    
    DropZone:hover {{
        border-color: {colors['accent']};
        background-color: {colors['bg_hover']};
    }}
    
    DropZone.dragging {{
        border-color: {colors['accent']};
        background-color: {colors['bg_hover']};
        border-width: 3px;
    }}
    """