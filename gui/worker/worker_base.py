"""
Worker 기본 클래스와 공통 유틸리티
"""
from PyQt6.QtCore import QThread, pyqtSignal
import sys
import queue
import threading
import os
import hashlib
import re
import time

class BaseSubtitleWorker(QThread):
    """Worker 기본 클래스 - 공통 기능 제공"""
    
    # 시그널 정의
    progress = pyqtSignal(str, int, str)  # 파일명, 퍼센트, 상태메시지
    fileCompleted = pyqtSignal(str, str)  # 파일명, 출력경로
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    
    def __init__(self, video_paths, settings):
        super().__init__()
        self.video_paths = video_paths
        self.settings = settings
        self.is_cancelled = False
        self.console_queue = queue.Queue()
        self.console_thread = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def safe_emit(self, signal, *args):
        """안전하게 시그널 발생"""
        if not self.is_cancelled:
            try:
                signal.emit(*args)
            except:
                pass
                
    def get_cache_path(self, video_path, lang_code=None):
        """비디오별 캐시 파일 경로"""
        os.makedirs("cache", exist_ok=True)
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        
        if lang_code:
            return f"cache/{video_hash}_{lang_code}.pkl"
        else:
            return f"cache/{video_hash}_original.pkl"
            
    def cancel(self):
        """작업 취소"""
        self.is_cancelled = True
        self.safe_emit(self.log, "작업 취소 중...")
        
    # 콘솔 출력 관련 메서드들
    def capture_console_output(self):
        """콘솔 출력 캡처 시작"""
        class ConsoleCapture:
            def __init__(self, queue, original):
                self.queue = queue
                self.original = original
                
            def write(self, text):
                try:
                    self.queue.put(text)
                    self.original.write(text)
                except:
                    pass
                    
            def flush(self):
                try:
                    self.original.flush()
                except:
                    pass
        
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        sys.stdout = ConsoleCapture(self.console_queue, self.original_stdout)
        sys.stderr = ConsoleCapture(self.console_queue, self.original_stderr)
        
    def restore_console_output(self):
        """콘솔 출력 복원"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
    def cleanup(self):
        """정리 작업"""
        try:
            self.restore_console_output()
            
            while not self.console_queue.empty():
                try:
                    self.console_queue.get_nowait()
                except:
                    break
                    
        except Exception as e:
            print(f"정리 중 오류: {e}")


class ProgressParser:
    """진행률 파싱 전용 클래스"""
    
    def __init__(self, worker):
        self.worker = worker
        self.is_downloading_model = False
        self.is_transcribing = False
        self.transcribe_start_time = None
        
    def parse_download_progress(self, line):
        """Whisper 모델 다운로드 진행률 파싱"""
        # 다운로드 진행률 패턴 매칭
        progress_pattern = r'(\d+)%\|.*?\|\s*(\d+\.?\d*[KMG]?)\/(\d+\.?\d*[KMG]?)\s*\[([^\]]+)\]'
        match = re.search(progress_pattern, line)
        
        if match:
            percent = int(match.group(1))
            downloaded = match.group(2)
            total = match.group(3)
            time_info = match.group(4)
            
            # 속도 정보 추출
            speed_match = re.search(r'(\d+\.?\d*[KMG]?B/s)', time_info)
            speed = speed_match.group(1) if speed_match else ""
            
            # 남은 시간 추출
            remaining_match = re.search(r'<(\d+:\d+)', time_info)
            remaining = remaining_match.group(1) if remaining_match else ""
            
            if not self.is_downloading_model:
                self.is_downloading_model = True
                self.worker.safe_emit(self.worker.log, f"Whisper 모델 다운로드 시작... (총 {total})")
            
            # 상태 메시지 구성
            status = f"다운로드 중... {downloaded}/{total}"
            if speed:
                status += f" ({speed})"
            if remaining:
                status += f" - 남은 시간: {remaining}"
                
            self.worker.safe_emit(self.worker.progress, "모델 다운로드", percent, status)
            
            if percent >= 100:
                self.is_downloading_model = False
                self.worker.safe_emit(self.worker.log, "✓ 모델 다운로드 완료")
                
            return True
            
        # 다운로드 시작 메시지 감지
        if "Downloading" in line and ".pt" in line:
            self.is_downloading_model = True
            model_match = re.search(r'Downloading\s+(\S+\.pt)', line)
            if model_match:
                model_file = model_match.group(1)
                self.worker.safe_emit(self.worker.log, f"Whisper 모델 다운로드 준비 중... ({model_file})")
            else:
                self.worker.safe_emit(self.worker.log, "Whisper 모델 다운로드 준비 중...")
            return True
            
        # 다운로드 오류 감지
        if self.is_downloading_model and ("error" in line.lower() or "failed" in line.lower()):
            self.is_downloading_model = False
            self.worker.safe_emit(self.worker.log, f"❌ 모델 다운로드 오류: {line}")
            return True
            
        return False
    
    def parse_transcribe_progress(self, line, video_name):
        """Whisper 음성 인식 진행률 파싱"""
        if not self.is_transcribing:
            return False
            
        # Whisper 진행률 패턴들
        timestamp_pattern = r'\[(\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}\.\d{3})\]'
        match = re.search(timestamp_pattern, line)
        
        if match:
            current_time = match.group(2)
            
            # 시간을 초 단위로 변환
            time_parts = current_time.split(':')
            minutes = int(time_parts[0])
            seconds = float(time_parts[1])
            total_seconds = minutes * 60 + seconds
            
            # 오디오 길이를 기반으로 진행률 계산
            estimated_total = 600  # 10분
            percent = min(int((total_seconds / estimated_total) * 100), 95)
            
            status = f"음성 인식 중... [{current_time}]"
            
            self.worker.safe_emit(self.worker.progress, video_name, 40 + int(percent * 0.5), status)
            
            return True
            
        if "Detecting language" in line:
            self.worker.safe_emit(self.worker.progress, video_name, 35, "언어 감지 중...")
            return True
            
        if "Transcription complete" in line or "Done" in line:
            self.worker.safe_emit(self.worker.progress, video_name, 90, "음성 인식 완료")
            return True
            
        return False