"""
백그라운드 작업 처리를 위한 Worker 스레드
"""
from PyQt6.QtCore import QThread, pyqtSignal
from auto_subtitle_llama.cli import get_audio
from auto_subtitle_llama.utils import filename, format_timestamp, write_srt, load_translator, get_text_batch, replace_text_batch
import whisper
import os
import pickle
import hashlib
import ffmpeg
import traceback
import shutil
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import threading
import queue
import time
import tempfile

class SubtitleWorker(QThread):
    """자막 생성 워커 스레드"""
    
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
        self.model = None
        self.console_queue = queue.Queue()
        self.console_thread = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.current_video_path = None
        
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
        
        # 원본 저장
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # stdout과 stderr 캡처
        sys.stdout = ConsoleCapture(self.console_queue, self.original_stdout)
        sys.stderr = ConsoleCapture(self.console_queue, self.original_stderr)
        
    def restore_console_output(self):
        """콘솔 출력 복원"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
    def process_console_queue(self):
        """콘솔 큐 처리"""
        buffer = ""
        while not self.is_cancelled:
            try:
                text = self.console_queue.get(timeout=0.1)
                buffer += text
                
                # 줄바꿈이 있으면 로그로 전송
                if '\n' in buffer or '\r' in buffer:
                    lines = buffer.replace('\r\n', '\n').replace('\r', '\n').split('\n')
                    for line in lines[:-1]:
                        if line.strip() and not self.is_cancelled:
                            try:
                                self.log.emit(line.strip())
                            except:
                                pass
                    buffer = lines[-1]
                    
            except queue.Empty:
                continue
            except Exception:
                break
                
    def get_cache_path(self, video_path, lang_code=None):
        """비디오별 캐시 파일 경로"""
        os.makedirs("cache", exist_ok=True)
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        
        if lang_code:
            return f"cache/{video_hash}_{lang_code}.pkl"
        else:
            return f"cache/{video_hash}_original.pkl"
            
    def safe_emit(self, signal, *args):
        """안전하게 시그널 발생"""
        if not self.is_cancelled:
            try:
                signal.emit(*args)
            except:
                pass
                
    def custom_transcribe(self, model, audio_path, args):
        """진행률 콜백이 있는 커스텀 transcribe"""
        if self.is_cancelled:
            return None
            
        video_name = filename(self.current_video_path)
        
        # Whisper의 transcribe 함수 호출
        args_with_verbose = args.copy()
        args_with_verbose['verbose'] = True
        args_with_verbose['fp16'] = False  # 안정성을 위해
        
        # 취소 체크를 위한 래퍼
        result = None
        
        def transcribe_with_cancel_check():
            nonlocal result
            if not self.is_cancelled:
                result = model.transcribe(audio_path, **args_with_verbose)
                
        # 별도 스레드에서 실행하여 취소 가능하게 만듦
        transcribe_thread = threading.Thread(target=transcribe_with_cancel_check)
        transcribe_thread.start()
        
        # 진행률 업데이트 (가상)
        start_time = time.time()
        while transcribe_thread.is_alive() and not self.is_cancelled:
            elapsed = time.time() - start_time
            # 예상 시간에 따른 진행률 (10초당 10%)
            percent = min(int(elapsed * 10), 90)
            self.safe_emit(self.progress, video_name, 40 + int(percent * 0.4), f"음성 인식 중... {percent}%")
            time.sleep(0.5)
            
        transcribe_thread.join(timeout=1.0)
        
        return result if not self.is_cancelled else None
            
    def run(self):
        """백그라운드 작업 실행"""
        try:
            # 콘솔 출력 캡처 시작
            self.capture_console_output()
            
            # 콘솔 큐 처리 스레드 시작
            self.console_thread = threading.Thread(target=self.process_console_queue)
            self.console_thread.daemon = True
            self.console_thread.start()
            
            # 모델 로드
            if not self.is_cancelled:
                self.safe_emit(self.log, f"Whisper {self.settings['model']} 모델 로드 중...")
                self.model = whisper.load_model(self.settings['model'])
                self.safe_emit(self.log, f"✓ 모델 로드 완료")
            
            # 각 비디오 파일 처리
            for idx, video_path in enumerate(self.video_paths):
                if self.is_cancelled:
                    break
                
                self.current_video_path = video_path
                video_name = filename(video_path)
                self.safe_emit(self.log, f"\n[{idx+1}/{len(self.video_paths)}] {video_name} 처리 시작")
                
                try:
                    # 1. 캐시 확인
                    cache_path = self.get_cache_path(video_path)
                    if os.path.exists(cache_path) and not self.settings.get('force_reprocess', False):
                        self.safe_emit(self.progress, video_name, 10, "캐시에서 불러오는 중...")
                        with open(cache_path, 'rb') as f:
                            cached_data = pickle.load(f)
                            
                        # 캐시된 데이터 사용
                        if not self.is_cancelled:
                            self.process_cached_video(video_path, cached_data)
                        continue
                    
                    if self.is_cancelled:
                        break
                        
                    # 2. 오디오 추출
                    self.safe_emit(self.progress, video_name, 20, "오디오 추출 중...")
                    self.safe_emit(self.log, "오디오 스트림 추출 중...")
                    audio_paths = get_audio([video_path])
                    self.safe_emit(self.log, "✓ 오디오 추출 완료")
                    
                    if self.is_cancelled:
                        break
                        
                    # 3. 음성 인식 (원본 언어)
                    self.safe_emit(self.progress, video_name, 30, "언어 감지 중...")
                    
                    # 언어 감지
                    audio = whisper.load_audio(list(audio_paths.values())[0])
                    audio = whisper.pad_or_trim(audio)
                    mel = whisper.log_mel_spectrogram(audio, self.model.dims.n_mels).to(self.model.device)
                    _, probs = self.model.detect_language(mel)
                    detected_lang = max(probs, key=probs.get)
                    self.safe_emit(self.log, f"감지된 언어: {detected_lang}")
                    
                    if self.is_cancelled:
                        break
                        
                    # 음성 인식 수행
                    self.safe_emit(self.progress, video_name, 40, "음성 인식 시작...")
                    
                    # 커스텀 transcribe 사용
                    result = self.custom_transcribe(
                        self.model,
                        list(audio_paths.values())[0],
                        {"task": "transcribe", "language": detected_lang}
                    )
                    
                    if result is None or self.is_cancelled:
                        break
                        
                    # SRT 파일 저장
                    srt_path = os.path.join(self.settings['output_dir'], f"{video_name}.srt")
                    with open(srt_path, "w", encoding="utf-8") as srt_file:
                        write_srt(result["segments"], srt_file)
                    
                    subtitles = {video_path: srt_path}
                    
                    # 캐시 저장
                    cache_data = {
                        'subtitles': subtitles,
                        'detected_lang': detected_lang,
                        'audio_paths': audio_paths,
                        'segments': result["segments"]
                    }
                    with open(cache_path, 'wb') as f:
                        pickle.dump(cache_data, f)
                    
                    self.safe_emit(self.log, f"✓ 원본 자막 생성 완료: {srt_path}")
                    
                    if self.is_cancelled:
                        break
                        
                    # 4. 번역 처리
                    if self.settings['translate'] and self.settings['languages']:
                        self.process_translations(
                            video_path, audio_paths, detected_lang, result["segments"]
                        )
                    
                    if self.is_cancelled:
                        break
                        
                    # 5. 비디오에 자막 임베딩 (옵션)
                    if not self.settings['srt_only']:
                        self.embed_subtitles(video_path, subtitles, detected_lang)
                        
                    # 완료
                    self.safe_emit(self.fileCompleted, video_name, srt_path)
                    self.safe_emit(self.progress, video_name, 100, "완료!")
                    
                except Exception as e:
                    if not self.is_cancelled:
                        self.safe_emit(self.log, f"❌ {video_name} 처리 중 오류: {str(e)}")
                        traceback.print_exc()
                    continue
                    
            if not self.is_cancelled:
                self.safe_emit(self.finished)
            
        except Exception as e:
            if not self.is_cancelled:
                self.safe_emit(self.error, str(e))
                traceback.print_exc()
        finally:
            # 정리 작업
            self.cleanup()
            
    def process_cached_video(self, video_path, cached_data):
        """캐시된 비디오 처리"""
        if self.is_cancelled:
            return
            
        video_name = filename(video_path)
        self.safe_emit(self.progress, video_name, 50, "캐시 데이터 로드 완료")
        
        # 번역 처리
        if self.settings['translate'] and self.settings['languages'] and not self.is_cancelled:
            segments = cached_data.get('segments', [])
            self.process_translations(
                video_path, 
                cached_data['audio_paths'],
                cached_data['detected_lang'],
                segments
            )
            
        # 비디오 임베딩
        if not self.settings['srt_only'] and not self.is_cancelled:
            self.embed_subtitles(
                video_path,
                cached_data['subtitles'],
                cached_data['detected_lang']
            )
            
        if not self.is_cancelled:
            output_path = list(cached_data['subtitles'].values())[0]
            self.safe_emit(self.fileCompleted, video_name, output_path)
            self.safe_emit(self.progress, video_name, 100, "완료!")
        
    def process_translations(self, video_path, audio_paths, detected_lang, segments):
        """번역 처리"""
        if self.is_cancelled:
            return
            
        video_name = filename(video_path)
        total_langs = len(self.settings['languages'])
        
        for i, target_lang in enumerate(self.settings['languages']):
            if self.is_cancelled:
                break
                
            # 같은 언어는 건너뛰기
            if target_lang.startswith(detected_lang):
                continue
                
            # 캐시 확인
            cache_path = self.get_cache_path(video_path, target_lang)
            if os.path.exists(cache_path):
                self.safe_emit(
                    self.progress,
                    video_name,
                    60 + (i * 30 // total_langs),
                    f"{target_lang} 캐시 로드 중..."
                )
                continue
                
            # 번역 수행
            progress = 60 + (i * 30 // total_langs)
            self.safe_emit(self.progress, video_name, progress, f"{target_lang} 번역 중...")
            
            try:
                if self.is_cancelled:
                    break
                    
                # LLaMA 번역 수행
                self.safe_emit(self.log, f"{target_lang} 번역 시작...")
                
                # 텍스트 배치 추출
                text_batch = get_text_batch(segments)
                
                # 번역 모델 로드 및 번역
                from auto_subtitle_llama.cli import translates
                translated_batch = translates(target_lang, text_batch)
                
                if self.is_cancelled:
                    break
                    
                # 번역된 텍스트로 segments 업데이트
                translated_segments = replace_text_batch(segments.copy(), translated_batch)
                
                # 번역된 SRT 저장
                translated_srt_path = os.path.join(
                    self.settings['output_dir'], 
                    f"{video_name}_{target_lang.split('_')[0]}.srt"
                )
                
                with open(translated_srt_path, "w", encoding="utf-8") as srt_file:
                    write_srt(translated_segments, srt_file)
                
                translated_subtitles = {video_path: translated_srt_path}
                
                # 번역 결과 캐시
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'subtitles': translated_subtitles,
                        'segments': translated_segments
                    }, f)
                    
                self.safe_emit(self.log, f"✓ {target_lang} 번역 완료")
                
                # 비디오 임베딩
                if not self.settings['srt_only'] and not self.is_cancelled:
                    self.embed_subtitles(
                        video_path, translated_subtitles, detected_lang, target_lang
                    )
                    
            except Exception as e:
                if not self.is_cancelled:
                    self.safe_emit(self.log, f"❌ {target_lang} 번역 실패: {str(e)}")
                
    def embed_subtitles(self, video_path, srt_paths, detected_lang, target_lang=None):
        """비디오에 자막 임베딩"""
        if self.is_cancelled:
            return
            
        video_name = filename(video_path)
        self.safe_emit(self.progress, video_name, 90, "비디오에 자막 추가 중...")
        
        # 출력 파일명 생성
        lang_suffix = f"_{detected_lang}"
        if target_lang:
            lang_suffix += f"2{target_lang.split('_')[0]}"
            
        out_filename = f"{video_name}_subtitled{lang_suffix}.mp4"
        out_path = os.path.join(self.settings['output_dir'], out_filename)
        
        try:
            if self.is_cancelled:
                return
                
            # ffmpeg로 자막 임베딩
            srt_path = list(srt_paths.values())[0]
            
            video = ffmpeg.input(video_path)
            audio = video.audio
            
            # 자막 스타일 적용
            subtitle_style = (
                "FallbackName=NanumGothic,"
                "OutlineColour=&H40000000,"
                "BorderStyle=3,"
                "Fontsize=24,"
                "MarginV=20"
            )
            
            ffmpeg.concat(
                video.filter('subtitles', srt_path, 
                           force_style=subtitle_style, 
                           charenc="UTF-8"),
                audio,
                v=1,
                a=1
            ).output(out_path).run(quiet=True, overwrite_output=True)
            
            self.safe_emit(self.log, f"✓ 자막 임베딩 완료: {out_filename}")
            
        except Exception as e:
            if not self.is_cancelled:
                self.safe_emit(self.log, f"❌ 자막 임베딩 실패: {str(e)}")
            
    def cancel(self):
        """작업 취소"""
        self.is_cancelled = True
        self.safe_emit(self.log, "작업 취소 중...")
        
    def cleanup(self):
        """정리 작업"""
        try:
            # 콘솔 출력 복원
            self.restore_console_output()
            
            # 콘솔 큐 비우기
            while not self.console_queue.empty():
                try:
                    self.console_queue.get_nowait()
                except:
                    break
                    
            # 임시 파일 정리
            self.cleanup_temp_files()
            
        except Exception as e:
            print(f"정리 중 오류: {e}")
        
    def cleanup_temp_files(self):
        """임시 파일 정리"""
        temp_dir = tempfile.gettempdir()
        
        # 오디오 파일 정리
        for video_path in self.video_paths:
            audio_filename = f"{filename(video_path)}.wav"
            audio_path = os.path.join(temp_dir, audio_filename)
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"임시 파일 삭제: {audio_path}")
                except Exception as e:
                    print(f"임시 파일 삭제 실패: {e}")