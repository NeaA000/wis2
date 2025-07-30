#!/usr/bin/env python
"""
GUI 애플리케이션 실행 스크립트
"""
import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
import multiprocessing

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui import MainWindow

def main():
    """메인 함수"""
    # Windows multiprocessing 지원
    if sys.platform.startswith('win'):
        multiprocessing.freeze_support()
    
    # 고해상도 디스플레이 지원
    try:
        from PyQt6.QtCore import Qt
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    except:
        pass  # 구버전 호환성
    
    # 애플리케이션 생성
    app = QApplication(sys.argv)
    app.setApplicationName("Auto Subtitle & Translate")
    app.setOrganizationName("YJ-20")
    
    # 스타일 설정
    app.setStyle("Fusion")  # 크로스 플랫폼 일관성
    
    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()
    
    # 화면 중앙에 위치
    screen = app.primaryScreen()
    if screen:
        screen_rect = screen.availableGeometry()
        window_rect = window.frameGeometry()
        center_point = screen_rect.center()
        window_rect.moveCenter(center_point)
        window.move(window_rect.topLeft())
    
    # 애플리케이션 실행
    sys.exit(app.exec())

if __name__ == "__main__":
    main()