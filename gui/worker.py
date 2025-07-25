"""
백그라운드 작업 처리를 위한 Worker 스레드
"""
from PyQt6.QtCore import QThread, pyqtSignal
from auto_subtitle_llama.cli import get_audio, get_subtitles
from auto_subtitle_llama.utils import filename
import whisper
import os
import pickle
import hashlib
import ffmpeg
import traceback
import shutil

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
        
    def get_cache_path(self, video_path, lang_code=None):
        """비디오별 캐시 파일 경로"""
        os.makedirs("cache", exist_ok=True)
        video_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        
        if lang_code:
            return f"cache/{video_hash}_{lang_code}.pkl"
        else:
            return f"cache/{video_hash}_original.pkl"
            
    def run(self):
        """백그라운드 작업 실행"""
        try:
            # 모델 로드
            self.log.emit(f"Whisper {self.settings['model']} 모델 로드 중...")
            self.model = whisper.load_model(self.settings['model'])
            
            # 각 비디오 파일 처리
            for idx, video_path in enumerate(self.video_paths):
                if self.is_cancelled:
                    break
                    
                video_name = filename(video_path)
                self.log.emit(f"\n[{idx+1}/{len(self.video_paths)}] {video_name} 처리 시작")
                
                try:
                    # 1. 캐시 확인
                    cache_path = self.get_cache_path(video_path)
                    if os.path.exists(cache_path) and not self.settings.get('force_reprocess', False):
                        self.progress.emit(video_name, 10, "캐시에서 불러오는 중...")
                        with open(cache_path, 'rb') as f:
                            cached_data = pickle.load(f)
                            
                        # 캐시된 데이터 사용
                        self.process_cached_video(video_path, cached_data)
                        continue
                    
                    # 2. 오디오 추출
                    self.progress.emit(video_name, 20, "오디오 추출 중...")
                    audio_paths = get_audio([video_path])
                    
                    # 3. 음성 인식 (원본 언어)
                    self.progress.emit(video_name, 40, "음성 인식 중... (시간이 걸립니다)")
                    subtitles, detected_lang = get_subtitles(
                        audio_paths,
                        output_srt=True,
                        output_dir=self.settings['output_dir'],
                        model=self.model,
                        args={"task": "transcribe", "verbose": False},
                        translate_to=None
                    )
                    
                    # 캐시 저장
                    cache_data = {
                        'subtitles': subtitles,
                        'detected_lang': detected_lang,
                        'audio_paths': audio_paths
                    }
                    with open(cache_path, 'wb') as f:
                        pickle.dump(cache_data, f)
                    
                    self.log.emit(f"원본 언어 감지: {detected_lang}")
                    
                    # 4. 번역 처리
                    if self.settings['translate'] and self.settings['languages']:
                        self.process_translations(
                            video_path, audio_paths, detected_lang
                        )
                    
                    # 5. 비디오에 자막 임베딩 (옵션)
                    if not self.settings['srt_only']:
                        self.embed_subtitles(video_path, subtitles, detected_lang)
                        
                    # 완료
                    output_path = list(subtitles.values())[0]
                    self.fileCompleted.emit(video_name, output_path)
                    
                except Exception as e:
                    self.log.emit(f"❌ {video_name} 처리 중 오류: {str(e)}")
                    traceback.print_exc()
                    continue
                    
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()
            
    def process_cached_video(self, video_path, cached_data):
        """캐시된 비디오 처리"""
        video_name = filename(video_path)
        
        # 번역 처리
        if self.settings['translate'] and self.settings['languages']:
            self.process_translations(
                video_path, 
                cached_data['audio_paths'],
                cached_data['detected_lang']
            )
            
        # 비디오 임베딩
        if not self.settings['srt_only']:
            self.embed_subtitles(
                video_path,
                cached_data['subtitles'],
                cached_data['detected_lang']
            )
            
        output_path = list(cached_data['subtitles'].values())[0]
        self.fileCompleted.emit(video_name, output_path)
        
    def process_translations(self, video_path, audio_paths, detected_lang):
        """번역 처리"""
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
                self.progress.emit(
                    video_name,
                    60 + (i * 30 // total_langs),
                    f"{target_lang} 캐시 로드 중..."
                )
                continue
                
            # 번역 수행
            progress = 60 + (i * 30 // total_langs)
            self.progress.emit(video_name, progress, f"{target_lang} 번역 중...")
            
            try:
                translated, _ = get_subtitles(
                    audio_paths,
                    output_srt=True,
                    output_dir=self.settings['output_dir'],
                    model=self.model,
                    args={"task": "transcribe", "verbose": False},
                    translate_to=target_lang
                )
                
                # 번역 결과 캐시
                with open(cache_path, 'wb') as f:
                    pickle.dump(translated, f)
                    
                self.log.emit(f"✓ {target_lang} 번역 완료")
                
                # 비디오 임베딩
                if not self.settings['srt_only']:
                    self.embed_subtitles(
                        video_path, translated, detected_lang, target_lang
                    )
                    
            except Exception as e:
                self.log.emit(f"❌ {target_lang} 번역 실패: {str(e)}")
                
    def embed_subtitles(self, video_path, srt_paths, detected_lang, target_lang=None):
        """비디오에 자막 임베딩"""
        if self.is_cancelled:
            return
            
        video_name = filename(video_path)
        self.progress.emit(video_name, 90, "비디오에 자막 추가 중...")
        
        # 출력 파일명 생성
        lang_suffix = f"_{detected_lang}"
        if target_lang:
            lang_suffix += f"2{target_lang.split('_')[0]}"
            
        out_filename = f"{video_name}_subtitled{lang_suffix}.mp4"
        out_path = os.path.join(self.settings['output_dir'], out_filename)
        
        try:
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
            
            self.log.emit(f"✓ 자막 임베딩 완료: {out_filename}")
            
        except Exception as e:
            self.log.emit(f"❌ 자막 임베딩 실패: {str(e)}")
            
    def cancel(self):
        """작업 취소"""
        self.is_cancelled = True
        self.log.emit("사용자에 의해 작업이 취소되었습니다.")
        
    def cleanup_temp_files(self):
        """임시 파일 정리"""
        import tempfile
        temp_dir = tempfile.gettempdir()
        
        # 오디오 파일 정리
        for video_path in self.video_paths:
            audio_filename = f"{filename(video_path)}.wav"
            audio_path = os.path.join(temp_dir, audio_filename)
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass