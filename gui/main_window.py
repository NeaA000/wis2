"""
메인 윈도우
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QPushButton, QMessageBox, QLabel
)
from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtGui import QIcon
import os
import sys

from .components import DropZone, ProgressPanel, SettingsPanel
from .worker.worker import SubtitleWorker
from .styles import get_stylesheet

class MainWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.video_queue = []
        self.is_closing = False
        self.init_ui()
        self.load_style()
        
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("Auto Subtitle & Translate")
        self.setMinimumSize(900, 700)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 왼쪽 패널 (드롭존 + 시작 버튼)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setSpacing(20)
        
        # 헤더
        header = QLabel("🎬 Auto Subtitle & Translate")
        header.setStyleSheet("font-size: 28px; font-weight: bold; margin-bottom: 10px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        subtitle = QLabel("Whisper + LLaMA 기반 자동 자막 생성 및 번역")
        subtitle.setStyleSheet("font-size: 14px; color: #666; margin-bottom: 20px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 드롭존
        self.drop_zone = DropZone()
        self.drop_zone.filesDropped.connect(self.on_files_dropped)
        
        # 파일 큐 표시
        self.queue_label = QLabel("대기 중인 파일: 0개")
        self.queue_label.setStyleSheet("font-size: 13px; color: #666;")
        
        # 시작 버튼
        self.start_btn = QPushButton("처리 시작")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                min-height: 50px;
            }
        """)
        
        left_layout.addWidget(header)
        left_layout.addWidget(subtitle)
        left_layout.addWidget(self.drop_zone)
        left_layout.addWidget(self.queue_label)
        left_layout.addWidget(self.start_btn)
        left_layout.addStretch()
        
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)
        
        # 오른쪽 패널 (설정 + 진행상황)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(0)
        
        # 탭 버튼
        tab_layout = QHBoxLayout()
        tab_layout.setSpacing(0)
        
        self.settings_tab_btn = QPushButton("⚙️ 설정")
        self.settings_tab_btn.clicked.connect(lambda: self.switch_tab(0))
        self.settings_tab_btn.setCheckable(True)
        self.settings_tab_btn.setChecked(True)
        self.settings_tab_btn.setStyleSheet("""
            QPushButton {
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                border-radius: 0;
                border-top-left-radius: 8px;
                border-top-right-radius: 0;
            }
            QPushButton:checked {
                background-color: #0d7377;
                color: white;
            }
        """)
        
        self.progress_tab_btn = QPushButton("📊 진행 상황")
        self.progress_tab_btn.clicked.connect(lambda: self.switch_tab(1))
        self.progress_tab_btn.setCheckable(True)
        self.progress_tab_btn.setStyleSheet("""
            QPushButton {
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                border-radius: 0;
                border-top-left-radius: 0;
                border-top-right-radius: 8px;
            }
            QPushButton:checked {
                background-color: #0d7377;
                color: white;
            }
        """)
        
        tab_layout.addWidget(self.settings_tab_btn)
        tab_layout.addWidget(self.progress_tab_btn)
        tab_layout.addStretch()
        
        # 스택 위젯 (설정/진행상황 전환)
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("""
            QStackedWidget {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 8px;
                border-top-left-radius: 0;
            }
        """)
        
        # 설정 패널
        self.settings_panel = SettingsPanel()
        self.settings_panel.settingsChanged.connect(self.on_settings_changed)
        
        # 진행 상황 패널
        self.progress_panel = ProgressPanel()
        self.progress_panel.cancelRequested.connect(self.cancel_processing)
        
        self.stack.addWidget(self.settings_panel)
        self.stack.addWidget(self.progress_panel)
        
        right_layout.addLayout(tab_layout)
        right_layout.addWidget(self.stack)
        
        right_panel.setLayout(right_layout)
        
        # 메인 레이아웃에 추가
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)
        
        central_widget.setLayout(main_layout)
        
    def load_style(self):
        """스타일시트 적용"""
        self.setStyleSheet(get_stylesheet())
        
    def switch_tab(self, index):
        """탭 전환"""
        self.stack.setCurrentIndex(index)
        self.settings_tab_btn.setChecked(index == 0)
        self.progress_tab_btn.setChecked(index == 1)
        
    def on_files_dropped(self, files):
        """파일 드롭 이벤트"""
        # 중복 제거
        for file in files:
            if file not in self.video_queue:
                self.video_queue.append(file)
                
        self.update_queue_label()
        self.start_btn.setEnabled(len(self.video_queue) > 0)
        
    def update_queue_label(self):
        """큐 레이블 업데이트"""
        count = len(self.video_queue)
        if count == 0:
            self.queue_label.setText("대기 중인 파일: 0개")
        elif count == 1:
            filename = os.path.basename(self.video_queue[0])
            self.queue_label.setText(f"선택된 파일: {filename}")
        else:
            self.queue_label.setText(f"대기 중인 파일: {count}개")
            
    def start_processing(self):
        """처리 시작"""
        if not self.video_queue:
            return
            
        # 설정 가져오기
        settings = self.settings_panel.get_settings()
        
        # 출력 디렉토리 생성
        try:
            os.makedirs(settings['output_dir'], exist_ok=True)
        except Exception as e:
            QMessageBox.critical(
                self,
                "오류",
                f"출력 폴더를 생성할 수 없습니다:\n{str(e)}"
            )
            return
        
        # 진행 상황 탭으로 전환
        self.switch_tab(1)
        
        # 진행 상황 패널 초기화
        self.progress_panel.start_processing(len(self.video_queue))
        
        # Worker 생성 및 시작
        self.worker = SubtitleWorker(self.video_queue.copy(), settings)
        
        # 시그널 연결
        self.worker.progress.connect(self.progress_panel.update_file_progress)
        self.worker.workerProgress.connect(self.progress_panel.update_worker_progress)
        self.worker.initWorkerProgress.connect(self.progress_panel.init_worker_progress)
        self.worker.fileCompleted.connect(self.on_file_completed)
        self.worker.finished.connect(self.on_all_completed)
        self.worker.error.connect(self.on_error)
        self.worker.log.connect(self.progress_panel.add_log)
        
        # 실시간 번역 시그널 연결 (새로 추가)
        self.worker.realtimeTranslation.connect(self.progress_panel.show_realtime_translation)
        
        # Worker 종료 시 정리
        self.worker.finished.connect(self.cleanup_worker)
        
        # 시작
        self.worker.start()
        
        # UI 상태 변경
        self.start_btn.setEnabled(False)
        self.drop_zone.setEnabled(False)
        self.settings_panel.setEnabled(False)  # 처리 중 설정 변경 방지
        
    def on_file_completed(self, filename, output_path):
        """파일 처리 완료"""
        if not self.is_closing:
            self.progress_panel.file_completed(filename, output_path)
        
    def on_all_completed(self):
        """모든 파일 처리 완료"""
        if self.is_closing:
            return
            
        self.progress_panel.finish_processing()
        
        # 완료 메시지
        QMessageBox.information(
            self,
            "처리 완료",
            f"모든 파일의 처리가 완료되었습니다!\n\n"
            f"출력 폴더: {self.settings_panel.get_settings()['output_dir']}"
        )
        
        # 큐 초기화
        self.video_queue.clear()
        self.update_queue_label()
        
        # UI 상태 복원
        self.restore_ui_state()
        
    def on_error(self, error_msg):
        """에러 발생"""
        if self.is_closing:
            return
            
        QMessageBox.critical(
            self,
            "오류 발생",
            f"처리 중 오류가 발생했습니다:\n\n{error_msg}"
        )
        
        # UI 상태 복원
        self.restore_ui_state()
        
    def restore_ui_state(self):
        """UI 상태 복원"""
        self.start_btn.setEnabled(len(self.video_queue) > 0)
        self.drop_zone.setEnabled(True)
        self.drop_zone.reset()
        self.settings_panel.setEnabled(True)
        
    def cancel_processing(self):
        """처리 취소"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "처리 취소",
                "정말로 처리를 취소하시겠습니까?\n\n"
                "현재 처리 중인 파일은 중단됩니다.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # 취소 상태 설정
                self.progress_panel.add_log("취소 처리 중...")
                
                # Worker 취소
                if self.worker:
                    self.worker.cancel()
                    
                    # Worker가 완전히 종료될 때까지 대기 (최대 5초)
                    if not self.worker.wait(5000):
                        # 강제 종료
                        self.worker.terminate()
                        self.worker.wait()
                        self.progress_panel.add_log("작업이 강제 종료되었습니다.")
                    else:
                        self.progress_panel.add_log("작업이 취소되었습니다.")
                
                # UI 상태 복원
                self.restore_ui_state()
                self.progress_panel.reset()
                
                # Worker 정리
                self.cleanup_worker()
                
    def cleanup_worker(self):
        """Worker 정리"""
        if self.worker:
            # 시그널 연결 해제
            try:
                self.worker.progress.disconnect()
                self.worker.workerProgress.disconnect()
                self.worker.initWorkerProgress.disconnect()
                self.worker.fileCompleted.disconnect()
                self.worker.finished.disconnect()
                self.worker.error.disconnect()
                self.worker.log.disconnect()
                self.worker.realtimeTranslation.disconnect()  # 추가
            except:
                pass
            
            # Worker 삭제
            self.worker.deleteLater()
            self.worker = None
            
    def on_settings_changed(self, settings):
        """설정 변경"""
        # 설정이 변경될 때마다 자동 저장됨
        print(f"설정 변경됨: {settings}")
        
    def closeEvent(self, event):
        """윈도우 종료 이벤트"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "종료 확인",
                "처리가 진행 중입니다.\n정말로 종료하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.is_closing = True
                
                # Worker 취소 및 종료
                if self.worker:
                    self.worker.cancel()
                    
                    # 종료 대기 (최대 3초)
                    if not self.worker.wait(3000):
                        self.worker.terminate()
                        self.worker.wait()
                        
                    self.cleanup_worker()
                
                event.accept()

                # 번역 모델 정리
                try:
                    from auto_subtitle_llama.utils import TranslatorManager
                    manager = TranslatorManager()
                    manager._model = None
                    manager._tokenizer = None
                except:
                    pass
            else:
                event.ignore()
        else:
            event.accept()