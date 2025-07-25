"""
설정 패널 컴포넌트
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QCheckBox, QGroupBox,
    QListWidget, QListWidgetItem, QFileDialog,
    QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal
import json
import os

class SettingsPanel(QWidget):
    """설정 패널"""
    
    # 설정이 변경되었을 때 발생하는 시그널
    settingsChanged = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.load_settings()
        
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
        self.translate_check.stateChanged.connect(self.on_translate_toggled)
        
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
        layout.addWidget(cache_group)
        layout.addStretch()
        
        self.setLayout(layout)
        
    def on_translate_toggled(self, checked):
        """번역 활성화/비활성화"""
        self.lang_list.setEnabled(checked)
        
    def browse_output_dir(self):
        """출력 디렉토리 선택"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "출력 폴더 선택",
            self.output_path.text()
        )
        if dir_path:
            self.output_path.setText(dir_path)
            self.save_settings()
            
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
                from PyQt6.QtCore import QTimer
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
            "srt_only": self.srt_only_check.isChecked()
        }
        
    def save_settings(self):
        """설정 저장"""
        settings = self.get_settings()
        os.makedirs("config", exist_ok=True)
        with open("config/settings.json", "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        self.settingsChanged.emit(settings)
        
    def load_settings(self):
        """설정 불러오기"""
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
            
        except FileNotFoundError:
            # 설정 파일이 없으면 기본값 사용
            pass