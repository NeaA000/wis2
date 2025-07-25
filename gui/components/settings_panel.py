"""
설정 패널 컴포넌트
"""
from PyQt6.QtWidgets import (
   QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
   QComboBox, QPushButton, QCheckBox, QGroupBox,
   QListWidget, QListWidgetItem, QFileDialog,
   QAbstractItemView, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
import json
import os
import multiprocessing

class SettingsPanel(QWidget):
   """설정 패널"""
   
   # 설정이 변경되었을 때 발생하는 시그널
   settingsChanged = pyqtSignal(dict)
   
   def __init__(self, parent=None):
       super().__init__(parent)
       self.is_loading = False  # 설정 로드 중 플래그
       self.save_timer = QTimer()  # 디바운싱을 위한 타이머
       self.save_timer.setSingleShot(True)
       self.save_timer.timeout.connect(self.save_settings)
       
       self.init_ui()
       self.load_settings()
       self.connect_signals()  # 시그널 연결을 로드 후에 수행
       
   def init_ui(self):
       """UI 초기화"""
       layout = QVBoxLayout()
       layout.setSpacing(20)
       
       # Whisper 모델 설정
       model_group = QGroupBox("Whisper 모델")
       model_layout = QVBoxLayout()
       
       model_label = QLabel("모델 크기:")
       self.model_combo = QComboBox()
       self.model_combo.addItems([
           "tiny (39MB)",
           "base (74MB)",
           "small (244MB)",
           "medium (769MB)",
           "large (1550MB)",
           "turbo (809MB)"
       ])
       self.model_combo.setCurrentIndex(5)  # turbo 기본값
       
       model_info = QLabel("💡 큰 모델일수록 정확도가 높지만 처리 시간이 길어집니다")
       model_info.setWordWrap(True)
       model_info.setStyleSheet("font-size: 12px; color: #888;")
       
       model_layout.addWidget(model_label)
       model_layout.addWidget(self.model_combo)
       model_layout.addWidget(model_info)
       model_group.setLayout(model_layout)
       
       # 번역 설정
       translate_group = QGroupBox("번역 설정")
       translate_layout = QVBoxLayout()
       
       self.translate_check = QCheckBox("자막 번역 활성화")
       self.translate_check.setChecked(True)
       
       lang_label = QLabel("번역 대상 언어 (다중 선택 가능):")
       self.lang_list = QListWidget()
       self.lang_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
       self.lang_list.setMaximumHeight(200)
       
       # 주요 언어 목록
       languages = [
           ("영어", "en_XX"),
           ("스페인어", "es_XX"),
           ("프랑스어", "fr_XX"),
           ("독일어", "de_DE"),
           ("일본어", "ja_XX"),
           ("중국어", "zh_CN"),
           ("러시아어", "ru_RU"),
           ("포르투갈어", "pt_XX"),
           ("이탈리아어", "it_IT"),
           ("네덜란드어", "nl_XX"),
           ("아랍어", "ar_AR"),
           ("힌디어", "hi_IN"),
           ("터키어", "tr_TR"),
           ("베트남어", "vi_VN"),
           ("태국어", "th_TH"),
           ("인도네시아어", "id_ID"),
           ("말레이어", "ms_MY"),
           ("스웨덴어", "sv_SE"),
           ("폴란드어", "pl_PL"),
           ("체코어", "cs_CZ")
       ]
       
       for name, code in languages:
           item = QListWidgetItem(name)
           item.setData(Qt.ItemDataRole.UserRole, code)
           self.lang_list.addItem(item)
           
       # 기본값: 영어, 스페인어 선택
       self.lang_list.item(0).setSelected(True)
       self.lang_list.item(1).setSelected(True)
       
       translate_layout.addWidget(self.translate_check)
       translate_layout.addWidget(lang_label)
       translate_layout.addWidget(self.lang_list)
       translate_group.setLayout(translate_layout)
       
       # 출력 설정
       output_group = QGroupBox("출력 설정")
       output_layout = QVBoxLayout()
       
       output_dir_layout = QHBoxLayout()
       output_label = QLabel("출력 폴더:")
       self.output_path = QLabel("output/")
       self.output_path.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
       self.browse_output_btn = QPushButton("변경")
       self.browse_output_btn.clicked.connect(self.browse_output_dir)
       
       output_dir_layout.addWidget(output_label)
       output_dir_layout.addWidget(self.output_path, 1)
       output_dir_layout.addWidget(self.browse_output_btn)
       
       self.srt_only_check = QCheckBox("자막 파일(.srt)만 생성 (비디오 임베딩 안함)")
       
       output_layout.addLayout(output_dir_layout)
       output_layout.addWidget(self.srt_only_check)
       output_group.setLayout(output_layout)
       
       # 병렬 처리 설정 그룹 (새로 추가)
       parallel_group = QGroupBox("병렬 처리 설정")
       parallel_layout = QVBoxLayout()
       
       self.parallel_check = QCheckBox("긴 영상 병렬 처리 활성화")
       self.parallel_check.setChecked(True)
       self.parallel_check.setToolTip(
           "30분 이상의 영상을 여러 부분으로 나누어 동시에 처리합니다.\n"
           "처리 속도가 크게 향상되지만 더 많은 메모리를 사용합니다."
       )
       
       # 병렬 처리 최소 영상 길이
       min_duration_layout = QHBoxLayout()
       min_duration_label = QLabel("병렬 처리 최소 길이:")
       self.min_duration_spin = QSpinBox()
       self.min_duration_spin.setRange(10, 120)  # 10분 ~ 120분
       self.min_duration_spin.setValue(30)  # 기본값 30분
       self.min_duration_spin.setSuffix(" 분")
       self.min_duration_spin.setToolTip("이 길이 이상의 영상에만 병렬 처리를 적용합니다")
       
       min_duration_layout.addWidget(min_duration_label)
       min_duration_layout.addWidget(self.min_duration_spin)
       min_duration_layout.addStretch()
       
       # 워커 수 설정
       workers_layout = QHBoxLayout()
       workers_label = QLabel("동시 작업 수:")
       self.workers_spin = QSpinBox()
       self.workers_spin.setRange(1, 16)
       self.workers_spin.setValue(min(4, multiprocessing.cpu_count()))  # CPU 코어 수 기준
       self.workers_spin.setToolTip(f"동시에 처리할 청크 수 (현재 CPU 코어: {multiprocessing.cpu_count()}개)")
       
       workers_layout.addWidget(workers_label)
       workers_layout.addWidget(self.workers_spin)
       workers_layout.addStretch()
       
       # 청크 크기 설정
       chunk_layout = QHBoxLayout()
       chunk_label = QLabel("청크 크기:")
       self.chunk_spin = QSpinBox()
       self.chunk_spin.setRange(10, 60)  # 10분 ~ 60분
       self.chunk_spin.setValue(30)  # 기본값 30분
       self.chunk_spin.setSuffix(" 분")
       self.chunk_spin.setToolTip("영상을 나눌 단위 크기 (작을수록 더 많이 분할됩니다)")
       
       chunk_layout.addWidget(chunk_label)
       chunk_layout.addWidget(self.chunk_spin)
       chunk_layout.addStretch()
       
       parallel_info = QLabel(
           "⚡ 병렬 처리 정보:\n"
           "• 2시간 영상: 4개 워커로 약 30-40분에 처리\n"
           "• 메모리 사용량: 워커당 약 2-3GB\n"
           "• 권장: CPU 코어 수의 절반 사용"
       )
       parallel_info.setWordWrap(True)
       parallel_info.setStyleSheet("font-size: 12px; color: #888; background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
       
       parallel_layout.addWidget(self.parallel_check)
       parallel_layout.addLayout(min_duration_layout)
       parallel_layout.addLayout(workers_layout)
       parallel_layout.addLayout(chunk_layout)
       parallel_layout.addWidget(parallel_info)
       parallel_group.setLayout(parallel_layout)
       
       # 실시간 번역 설정 그룹
       realtime_group = QGroupBox("실시간 번역 설정")
       realtime_layout = QVBoxLayout()
       
       self.realtime_check = QCheckBox("실시간 스트리밍 번역 활성화")
       self.realtime_check.setChecked(True)
       self.realtime_check.setToolTip(
           "Whisper가 음성을 인식하는 동시에 번역을 수행합니다.\n"
           "전체 처리 시간을 30-40% 단축할 수 있습니다."
       )
       
       realtime_info = QLabel(
           "⚡ 실시간 번역 장점:\n"
           "• 음성 인식과 번역 동시 진행\n"
           "• GPU(Whisper)와 CPU(번역) 동시 활용\n"
           "• 진행 상황을 실시간으로 확인 가능\n"
           "• 전체 처리 시간 대폭 단축"
       )
       realtime_info.setWordWrap(True)
       realtime_info.setStyleSheet("font-size: 12px; color: #888; background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
       
       # 실시간 로그 표시 옵션
       self.realtime_log_check = QCheckBox("실시간 번역 로그 표시")
       self.realtime_log_check.setChecked(True)
       self.realtime_log_check.setToolTip("처리 중인 자막과 번역을 실시간으로 로그에 표시합니다")
       
       realtime_layout.addWidget(self.realtime_check)
       realtime_layout.addWidget(self.realtime_log_check)
       realtime_layout.addWidget(realtime_info)
       realtime_group.setLayout(realtime_layout)
       
       # 캐시 설정
       cache_group = QGroupBox("캐시")
       cache_layout = QHBoxLayout()
       
       cache_label = QLabel("처리된 비디오를 캐시에 저장하여 재처리 시간을 단축합니다")
       self.clear_cache_btn = QPushButton("캐시 삭제")
       self.clear_cache_btn.clicked.connect(self.clear_cache)
       
       cache_layout.addWidget(cache_label, 1)
       cache_layout.addWidget(self.clear_cache_btn)
       cache_group.setLayout(cache_layout)
       
       # 레이아웃에 추가
       layout.addWidget(model_group)
       layout.addWidget(translate_group)
       layout.addWidget(output_group)
       layout.addWidget(parallel_group)  # 병렬 처리 설정 추가
       layout.addWidget(realtime_group)  # 실시간 번역 설정 추가
       layout.addWidget(cache_group)
       layout.addStretch()
       
       self.setLayout(layout)
       
   def connect_signals(self):
       """시그널 연결 (로드 완료 후 호출)"""
       # 모델 변경 시 자동 저장
       self.model_combo.currentIndexChanged.connect(self.on_settings_changed)
       
       # 번역 체크박스 변경 시
       self.translate_check.stateChanged.connect(self.on_translate_toggled)
       
       # 언어 선택 변경 시 자동 저장
       self.lang_list.itemSelectionChanged.connect(self.on_settings_changed)
       
       # SRT only 체크박스 변경 시 자동 저장
       self.srt_only_check.stateChanged.connect(self.on_settings_changed)
       
       # 병렬 처리 설정 변경 시
       self.parallel_check.stateChanged.connect(self.on_parallel_toggled)
       self.min_duration_spin.valueChanged.connect(self.on_settings_changed)
       self.workers_spin.valueChanged.connect(self.on_settings_changed)
       self.chunk_spin.valueChanged.connect(self.on_settings_changed)
       
       # 실시간 번역 설정 변경 시
       self.realtime_check.stateChanged.connect(self.on_realtime_toggled)
       self.realtime_log_check.stateChanged.connect(self.on_settings_changed)
       
   def on_translate_toggled(self, checked):
       """번역 활성화/비활성화"""
       self.lang_list.setEnabled(checked)
       self.on_settings_changed()  # 설정 저장
       
   def on_parallel_toggled(self, checked):
       """병렬 처리 활성화/비활성화"""
       self.min_duration_spin.setEnabled(checked)
       self.workers_spin.setEnabled(checked)
       self.chunk_spin.setEnabled(checked)
       self.on_settings_changed()  # 설정 저장
       
   def on_realtime_toggled(self, checked):
       """실시간 번역 활성화/비활성화"""
       self.realtime_log_check.setEnabled(checked)
       self.on_settings_changed()  # 설정 저장
       
   def on_settings_changed(self):
       """설정 변경 시 호출 (디바운싱 적용)"""
       if not self.is_loading:  # 로드 중이 아닐 때만 저장
           # 기존 타이머 취소하고 새로 시작 (500ms 디바운싱)
           self.save_timer.stop()
           self.save_timer.start(500)
       
   def browse_output_dir(self):
       """출력 디렉토리 선택"""
       dir_path = QFileDialog.getExistingDirectory(
           self,
           "출력 폴더 선택",
           self.output_path.text()
       )
       if dir_path:
           self.output_path.setText(dir_path)
           self.on_settings_changed()  # save_settings() 대신 디바운싱 적용
           
   def clear_cache(self):
       """캐시 삭제"""
       import shutil
       cache_dir = "cache"
       if os.path.exists(cache_dir):
           try:
               shutil.rmtree(cache_dir)
               os.makedirs(cache_dir)
               self.clear_cache_btn.setText("✓ 캐시 삭제됨")
               # 2초 후 원래 텍스트로 복원
               QTimer.singleShot(2000, lambda: self.clear_cache_btn.setText("캐시 삭제"))
           except Exception as e:
               print(f"캐시 삭제 실패: {e}")
               
   def get_settings(self):
       """현재 설정 반환"""
       # 선택된 언어 코드 가져오기
       selected_languages = []
       for i in range(self.lang_list.count()):
           item = self.lang_list.item(i)
           if item.isSelected():
               selected_languages.append(item.data(Qt.ItemDataRole.UserRole))
       
       return {
           "model": self.model_combo.currentText().split()[0],  # "turbo (809MB)" -> "turbo"
           "translate": self.translate_check.isChecked(),
           "languages": selected_languages,
           "output_dir": self.output_path.text(),
           "srt_only": self.srt_only_check.isChecked(),
           # 병렬 처리 설정
           "parallel_processing": self.parallel_check.isChecked(),
           "parallel_min_duration": self.min_duration_spin.value() * 60,  # 분 -> 초
           "parallel_workers": self.workers_spin.value(),
           "chunk_duration": self.chunk_spin.value() * 60,  # 분 -> 초
           # 실시간 번역 설정
           "realtime_translation": self.realtime_check.isChecked(),
           "realtime_log": self.realtime_log_check.isChecked(),
       }
       
   def save_settings(self):
       """설정 저장"""
       try:
           settings = self.get_settings()
           os.makedirs("config", exist_ok=True)
           with open("config/settings.json", "w", encoding="utf-8") as f:
               json.dump(settings, f, indent=2, ensure_ascii=False)
           self.settingsChanged.emit(settings)
           print(f"설정 저장됨: {settings}")  # 디버깅용
       except Exception as e:
           print(f"설정 저장 실패: {e}")
       
   def load_settings(self):
       """설정 불러오기"""
       self.is_loading = True  # 로드 시작
       try:
           with open("config/settings.json", "r", encoding="utf-8") as f:
               settings = json.load(f)
               
           # 모델 설정
           for i in range(self.model_combo.count()):
               if settings.get("model") in self.model_combo.itemText(i):
                   self.model_combo.setCurrentIndex(i)
                   break
                   
           # 번역 설정
           self.translate_check.setChecked(settings.get("translate", True))
           
           # 언어 선택
           saved_languages = settings.get("languages", ["en_XX", "es_XX"])
           for i in range(self.lang_list.count()):
               item = self.lang_list.item(i)
               code = item.data(Qt.ItemDataRole.UserRole)
               item.setSelected(code in saved_languages)
               
           # 출력 설정
           self.output_path.setText(settings.get("output_dir", "output/"))
           self.srt_only_check.setChecked(settings.get("srt_only", False))
           
           # 병렬 처리 설정
           self.parallel_check.setChecked(settings.get("parallel_processing", True))
           self.min_duration_spin.setValue(settings.get("parallel_min_duration", 1800) // 60)  # 초 -> 분
           self.workers_spin.setValue(settings.get("parallel_workers", min(4, multiprocessing.cpu_count())))
           self.chunk_spin.setValue(settings.get("chunk_duration", 1800) // 60)  # 초 -> 분
           
           # 병렬 처리 관련 위젯 활성화/비활성화
           parallel_enabled = self.parallel_check.isChecked()
           self.min_duration_spin.setEnabled(parallel_enabled)
           self.workers_spin.setEnabled(parallel_enabled)
           self.chunk_spin.setEnabled(parallel_enabled)
           
           # 실시간 번역 설정
           self.realtime_check.setChecked(settings.get("realtime_translation", True))
           self.realtime_log_check.setChecked(settings.get("realtime_log", True))
           self.realtime_log_check.setEnabled(self.realtime_check.isChecked())
           
           print(f"설정 로드됨: {settings}")  # 디버깅용
           
       except FileNotFoundError:
           # 설정 파일이 없으면 기본값 사용
           print("설정 파일 없음, 기본값 사용")
       except Exception as e:
           print(f"설정 로드 실패: {e}")
       finally:
           self.is_loading = False  # 로드 완료