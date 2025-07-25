"""
비디오 파일 드래그&드롭 컴포넌트
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QDragMoveEvent

class DropZone(QWidget):
    """드래그&드롭 영역"""
    
    # 파일이 드롭되었을 때 발생하는 시그널
    filesDropped = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("DropZone")
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        # 아이콘 (텍스트로 대체)
        icon_label = QLabel("📹")
        icon_label.setStyleSheet("font-size: 48px;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 안내 텍스트
        self.info_label = QLabel("비디오 파일을 여기에 드래그하세요")
        self.info_label.setObjectName("info")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("font-size: 16px; font-weight: 500;")
        
        # 지원 포맷
        format_label = QLabel("지원 포맷: MP4, AVI, MOV, MKV")
        format_label.setObjectName("format")
        format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        format_label.setStyleSheet("font-size: 12px; opacity: 0.7;")
        
        # 파일 선택 버튼
        self.browse_btn = QPushButton("파일 선택")
        self.browse_btn.setObjectName("browse")
        self.browse_btn.clicked.connect(self.browse_files)
        self.browse_btn.setStyleSheet("""
            QPushButton {
                min-width: 120px;
                max-width: 200px;
            }
        """)
        
        # 레이아웃에 추가
        layout.addWidget(icon_label)
        layout.addWidget(self.info_label)
        layout.addWidget(format_label)
        layout.addWidget(self.browse_btn)
        
        self.setLayout(layout)
        self.setMinimumHeight(250)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """드래그 진입 이벤트"""
        if event.mimeData().hasUrls():
            # 비디오 파일인지 확인
            urls = event.mimeData().urls()
            valid_files = self.get_valid_video_files(urls)
            
            if valid_files:
                event.acceptProposedAction()
                self.setProperty("dragging", True)
                self.style().polish(self)
        
    def dragMoveEvent(self, event: QDragMoveEvent):
        """드래그 이동 이벤트"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dragLeaveEvent(self, event):
        """드래그 나가기 이벤트"""
        self.setProperty("dragging", False)
        self.style().polish(self)
        
    def dropEvent(self, event: QDropEvent):
        """드롭 이벤트"""
        self.setProperty("dragging", False)
        self.style().polish(self)
        
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            valid_files = self.get_valid_video_files(urls)
            
            if valid_files:
                self.filesDropped.emit(valid_files)
                event.acceptProposedAction()
                
                # UI 업데이트
                if len(valid_files) == 1:
                    self.info_label.setText(f"선택됨: {valid_files[0].split('/')[-1]}")
                else:
                    self.info_label.setText(f"{len(valid_files)}개 파일 선택됨")
                    
    def get_valid_video_files(self, urls):
        """유효한 비디오 파일만 필터링"""
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        valid_files = []
        
        for url in urls:
            file_path = url.toLocalFile()
            if any(file_path.lower().endswith(ext) for ext in valid_extensions):
                valid_files.append(file_path)
                
        return valid_files
        
    def browse_files(self):
        """파일 선택 다이얼로그"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "비디오 파일 선택",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv);;All Files (*.*)"
        )
        
        if files:
            self.filesDropped.emit(files)
            
            # UI 업데이트
            if len(files) == 1:
                self.info_label.setText(f"선택됨: {files[0].split('/')[-1]}")
            else:
                self.info_label.setText(f"{len(files)}개 파일 선택됨")
                
    def reset(self):
        """드롭존 초기화"""
        self.info_label.setText("비디오 파일을 여기에 드래그하세요")
        self.setProperty("dragging", False)
        self.style().polish(self)