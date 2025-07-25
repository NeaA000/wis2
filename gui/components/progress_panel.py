"""
진행 상황 표시 패널
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QProgressBar, QPushButton, QListWidget, QListWidgetItem,
    QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QTextCursor
import time

class ProgressPanel(QWidget):
    """진행 상황 패널"""
    
    # 취소 버튼 클릭 시그널
    cancelRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_time = None
        self.current_file = None
        self.total_files = 0
        self.completed_files = 0
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # 제목
        title = QLabel("처리 진행 상황")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        
        # 현재 파일 정보
        self.current_file_label = QLabel("대기 중...")
        self.current_file_label.setStyleSheet("font-size: 14px; font-weight: 500;")
        
        # 전체 진행률
        overall_layout = QHBoxLayout()
        overall_label = QLabel("전체 진행률:")
        self.overall_progress = QProgressBar()
        self.overall_progress.setTextVisible(True)
        self.overall_percent_label = QLabel("0%")
        
        overall_layout.addWidget(overall_label)
        overall_layout.addWidget(self.overall_progress, 1)
        overall_layout.addWidget(self.overall_percent_label)
        
        # 현재 파일 진행률
        current_layout = QHBoxLayout()
        current_label = QLabel("현재 파일:")
        self.current_progress = QProgressBar()
        self.current_progress.setTextVisible(False)
        self.current_percent_label = QLabel("0%")
        
        current_layout.addWidget(current_label)
        current_layout.addWidget(self.current_progress, 1)
        current_layout.addWidget(self.current_percent_label)
        
        # 상태 메시지
        self.status_label = QLabel("준비 중...")
        self.status_label.setStyleSheet("color: #666; font-size: 13px;")
        
        # 시간 정보
        time_layout = QHBoxLayout()
        self.elapsed_label = QLabel("경과 시간: 00:00")
        self.estimated_label = QLabel("예상 남은 시간: 계산 중...")
        time_layout.addWidget(self.elapsed_label)
        time_layout.addStretch()
        time_layout.addWidget(self.estimated_label)
        
        # 로그 영역
        log_label = QLabel("처리 로그:")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        
        # 처리 완료 파일 목록
        completed_label = QLabel("완료된 파일:")
        self.completed_list = QListWidget()
        self.completed_list.setMaximumHeight(100)
        
        # 취소 버튼
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.on_cancel_clicked)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        button_layout.addWidget(self.cancel_btn)
        
        # 레이아웃에 추가
        layout.addWidget(title)
        layout.addWidget(self.current_file_label)
        layout.addLayout(overall_layout)
        layout.addLayout(current_layout)
        layout.addWidget(self.status_label)
        layout.addLayout(time_layout)
        layout.addWidget(log_label)
        layout.addWidget(self.log_text)
        layout.addWidget(completed_label)
        layout.addWidget(self.completed_list)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # 타이머 설정 (1초마다 시간 업데이트)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        
    def start_processing(self, total_files):
        """처리 시작"""
        self.total_files = total_files
        self.completed_files = 0
        self.start_time = time.time()
        self.timer.start(1000)  # 1초마다
        self.cancel_btn.setEnabled(True)
        self.completed_list.clear()
        self.log_text.clear()
        
    def update_file_progress(self, filename, current_percent, status_message):
        """현재 파일 진행률 업데이트"""
        self.current_file = filename
        self.current_file_label.setText(f"처리 중: {filename}")
        self.current_progress.setValue(int(current_percent))
        self.current_percent_label.setText(f"{int(current_percent)}%")
        self.status_label.setText(status_message)
        
        # 전체 진행률 계산
        overall_percent = (self.completed_files * 100 + current_percent) / self.total_files
        self.overall_progress.setValue(int(overall_percent))
        self.overall_percent_label.setText(f"{int(overall_percent)}%")
        
    def file_completed(self, filename, output_path):
        """파일 처리 완료"""
        self.completed_files += 1
        
        # 완료 목록에 추가
        item = QListWidgetItem(f"✓ {filename}")
        item.setData(Qt.ItemDataRole.UserRole, output_path)
        self.completed_list.addItem(item)
        
        # 로그에 추가
        self.add_log(f"[완료] {filename} → {output_path}")
        
        # 진행률 업데이트
        if self.completed_files < self.total_files:
            self.current_progress.setValue(0)
            self.current_percent_label.setText("0%")
        else:
            # 모든 파일 처리 완료
            self.finish_processing()
            
    def finish_processing(self):
        """처리 완료"""
        self.timer.stop()
        self.cancel_btn.setEnabled(False)
        self.current_file_label.setText("✨ 모든 처리가 완료되었습니다!")
        self.status_label.setText(f"총 {self.completed_files}개 파일 처리 완료")
        self.current_progress.setValue(100)
        self.current_percent_label.setText("100%")
        self.add_log(f"\n[완료] 총 처리 시간: {self.elapsed_label.text()}")
        
    def add_log(self, message):
        """로그 메시지 추가"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # 자동 스크롤
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
        
    def update_time(self):
        """경과 시간 업데이트"""
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            
            if hours > 0:
                self.elapsed_label.setText(f"경과 시간: {hours:02d}:{minutes:02d}:{seconds:02d}")
            else:
                self.elapsed_label.setText(f"경과 시간: {minutes:02d}:{seconds:02d}")
                
            # 예상 남은 시간 계산
            if self.completed_files > 0 or self.current_progress.value() > 0:
                total_progress = self.overall_progress.value()
                if total_progress > 0:
                    estimated_total = elapsed * 100 / total_progress
                    remaining = int(estimated_total - elapsed)
                    
                    if remaining > 0:
                        if remaining > 3600:
                            hours = remaining // 3600
                            minutes = (remaining % 3600) // 60
                            self.estimated_label.setText(f"예상 남은 시간: {hours}시간 {minutes}분")
                        elif remaining > 60:
                            minutes = remaining // 60
                            seconds = remaining % 60
                            self.estimated_label.setText(f"예상 남은 시간: {minutes}분 {seconds}초")
                        else:
                            self.estimated_label.setText(f"예상 남은 시간: {remaining}초")
                    else:
                        self.estimated_label.setText("예상 남은 시간: 곧 완료")
                        
    def on_cancel_clicked(self):
        """취소 버튼 클릭"""
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("취소 중...")
        self.status_label.setText("작업을 취소하는 중입니다...")
        self.cancelRequested.emit()
        
    def reset(self):
        """패널 초기화"""
        self.timer.stop()
        self.start_time = None
        self.current_file = None
        self.total_files = 0
        self.completed_files = 0
        
        self.current_file_label.setText("대기 중...")
        self.overall_progress.setValue(0)
        self.overall_percent_label.setText("0%")
        self.current_progress.setValue(0)
        self.current_percent_label.setText("0%")
        self.status_label.setText("준비 중...")
        self.elapsed_label.setText("경과 시간: 00:00")
        self.estimated_label.setText("예상 남은 시간: 계산 중...")
        self.completed_list.clear()
        self.log_text.clear()
        self.cancel_btn.setText("취소")
        self.cancel_btn.setEnabled(False)