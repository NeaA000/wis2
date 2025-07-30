"""
진행률 추적 및 관리를 위한 클래스
"""
import time
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from threading import Lock

@dataclass
class Stage:
    """작업 단계 정보"""
    name: str
    weight: float  # 전체에서 차지하는 비중 (0~1)
    progress: float = 0.0  # 현재 진행률 (0~100)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    sub_stages: Dict[str, float] = field(default_factory=dict)  # 서브 단계

class ProgressTracker:
    """통합 진행률 추적기"""

    def __init__(self, callback: Callable[[str, int, str], None]):
        """
        Args:
            callback: 진행률 업데이트 콜백 (filename, percent, status)
        """
        self.callback = callback
        self.lock = Lock()

        # 기본 단계 정의 (가중치 합 = 1.0)
        self.stages = {
            'prepare': Stage('준비', 0.02),
            'model_load': Stage('모델 로드', 0.03),
            'audio_extract': Stage('오디오 추출', 0.05),
            'transcribe': Stage('음성 인식', 0.65),
            'translate': Stage('번역', 0.20),
            'save': Stage('저장 및 임베딩', 0.05)
        }

        # 번역 서브 단계 (언어별)
        self.translation_languages = []

        # 현재 상태
        self.current_stage = None
        self.current_file = ""
        self.start_time = None

        # 통계
        self.stage_times = {}  # 각 단계별 실제 소요 시간

    def start(self, filename: str):
        """작업 시작"""
        with self.lock:
            self.current_file = filename
            self.start_time = time.time()
            # 모든 단계 초기화
            for stage in self.stages.values():
                stage.progress = 0.0
                stage.start_time = None
                stage.end_time = None

    def enter_stage(self, stage_name: str, status: str = None):
        """새 단계 진입"""
        with self.lock:
            if self.current_stage:
                # 이전 단계 완료 처리
                self.stages[self.current_stage].progress = 100.0
                self.stages[self.current_stage].end_time = time.time()

            self.current_stage = stage_name
            stage = self.stages[stage_name]
            stage.start_time = time.time()

            if status is None:
                status = f"{stage.name} 시작..."

            # 전체 진행률 계산 및 콜백
            total_progress = self._calculate_total_progress()
            self.callback(self.current_file, total_progress, status)

    def update_stage_progress(self, progress: float, status: str = None,
                              sub_stage: str = None, sub_progress: float = None):
        """현재 단계의 진행률 업데이트"""
        with self.lock:
            if not self.current_stage:
                return

            stage = self.stages[self.current_stage]
            stage.progress = min(progress, 100.0)

            # 서브 단계 업데이트
            if sub_stage and sub_progress is not None:
                stage.sub_stages[sub_stage] = sub_progress

            # 상태 메시지 생성
            if status is None:
                status = f"{stage.name} 진행 중... {int(progress)}%"

            # 전체 진행률 계산
            total_progress = self._calculate_total_progress()
            self.callback(self.current_file, total_progress, status)

    def update_transcribe_progress(self, current_time: float, total_time: float,
                                   text_preview: str = None):
        """음성 인식 진행률 업데이트 (시간 기반)"""
        if total_time <= 0:
            return

        progress = min((current_time / total_time) * 100, 99)

        # 상태 메시지
        remaining = total_time - current_time
        if remaining > 0:
            if remaining > 60:
                status = f"음성 인식 중... [{self._format_time(current_time)}/{self._format_time(total_time)}] - 남은 시간: {int(remaining//60)}분 {int(remaining%60)}초"
            else:
                status = f"음성 인식 중... [{self._format_time(current_time)}/{self._format_time(total_time)}] - 남은 시간: {int(remaining)}초"
        else:
            status = f"음성 인식 중... [{self._format_time(current_time)}/{self._format_time(total_time)}]"

        if text_preview:
            preview = text_preview[:30] + "..." if len(text_preview) > 30 else text_preview
            status += f" - {preview}"

        self.update_stage_progress(progress, status)

    def update_translation_progress(self, language: str, lang_index: int,
                                    total_languages: int, lang_progress: float):
        """번역 진행률 업데이트 (언어별)"""
        # 전체 번역 진행률 계산
        base_progress = (lang_index / total_languages) * 100
        current_lang_progress = (lang_progress / total_languages)
        total_progress = base_progress + current_lang_progress

        status = f"번역 중... {language}: {int(lang_progress)}%"
        self.update_stage_progress(total_progress, status,
                                   sub_stage=language, sub_progress=lang_progress)

    def update_parallel_progress(self, worker_progresses: Dict[int, float]):
        """병렬 처리 진행률 업데이트"""
        if not worker_progresses:
            return

        # 모든 워커의 평균 진행률
        avg_progress = sum(worker_progresses.values()) / len(worker_progresses)
        active_workers = sum(1 for p in worker_progresses.values() if p < 100)

        status = f"병렬 처리 중... (활성 워커: {active_workers}개)"
        self.update_stage_progress(avg_progress, status)

    def set_translation_languages(self, languages: list):
        """번역 대상 언어 설정 (가중치 재계산)"""
        with self.lock:
            self.translation_languages = languages

            if not languages:
                # 번역 없으면 번역 단계 가중치를 음성 인식에 추가
                self.stages['transcribe'].weight = 0.85
                self.stages['translate'].weight = 0.0
            else:
                # 번역 있으면 기본 가중치 사용
                self.stages['transcribe'].weight = 0.65
                self.stages['translate'].weight = 0.20

    def complete(self):
        """작업 완료"""
        with self.lock:
            if self.current_stage:
                self.stages[self.current_stage].progress = 100.0
                self.stages[self.current_stage].end_time = time.time()

            # 통계 수집
            if self.start_time:
                total_time = time.time() - self.start_time
                for name, stage in self.stages.items():
                    if stage.start_time and stage.end_time:
                        self.stage_times[name] = stage.end_time - stage.start_time

            self.callback(self.current_file, 100, "완료!")

    def get_statistics(self) -> Dict:
        """작업 통계 반환"""
        with self.lock:
            return {
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'stage_times': self.stage_times.copy(),
                'stages': {name: {
                    'progress': stage.progress,
                    'time': stage.end_time - stage.start_time
                    if stage.start_time and stage.end_time else 0
                } for name, stage in self.stages.items()}
            }

    def _calculate_total_progress(self) -> int:
        """전체 진행률 계산"""
        total = 0.0

        for name, stage in self.stages.items():
            # 완료된 단계는 100%
            if stage.end_time:
                total += stage.weight * 100
            # 현재 단계는 진행률 반영
            elif name == self.current_stage:
                total += stage.weight * stage.progress
            # 아직 시작 안 한 단계는 0%

        return min(int(total), 100)

    def _format_time(self, seconds: float) -> str:
        """시간 포맷팅"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def adjust_weights_from_history(self, history: Dict[str, float]):
        """과거 실행 데이터를 기반으로 가중치 자동 조정"""
        with self.lock:
            total_time = sum(history.values())
            if total_time > 0:
                for stage_name, stage_time in history.items():
                    if stage_name in self.stages:
                        self.stages[stage_name].weight = stage_time / total_time